"""Capture a real warm CPU workload (--capture), or redraw its saved measurements.
Optional plotting dependency: matplotlib. Runtime application requirements are unchanged.
AI PromQL prompt: measure mode throughput, active uploads, CPU idle-rate utilization,
host RAM and process RSS at 5-second intervals; preserve raw samples and explain scope.
"""
import argparse
import io
import json
import time
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from prometheus_client.parser import text_string_to_metric_families

HERE = Path(__file__).resolve().parent
SOURCE = 'https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/images/flower.jpg'
QUERIES = {
    'throughput': 'sum by (mode) (rate(caption_images_total[15s]))',
    'active': 'sum by (mode) (caption_active_requests)',
    'cpu': '100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle",job="node"}[15s])))',
    'ram': '100 * (1 - node_memory_MemAvailable_bytes{job="node"} / node_memory_MemTotal_bytes{job="node"})',
    'rss': 'caption_process_memory_bytes',
}


def metrics():
    response = requests.get('http://127.0.0.1:8000/metrics', timeout=10)
    response.raise_for_status()
    return [s for f in text_string_to_metric_families(response.text) for s in f.samples]


def value(samples, name, **labels):
    return sum(s.value for s in samples if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()))


def capture():
    from PIL import Image, ImageEnhance
    response = requests.get(SOURCE, timeout=30)
    response.raise_for_status()
    photograph = response.content
    image = Image.open(io.BytesIO(photograph)).convert('RGB')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for i in range(10):
            output = io.BytesIO()
            ImageEnhance.Brightness(image).enhance(.8 + .04*i).save(output, format='JPEG', quality=90)
            archive.writestr(f'flower-{i+1:02}.jpg', output.getvalue())
    health = requests.get('http://127.0.0.1:8001/health', timeout=10).json()
    if not health.get('model_ready'):
        # Warm-up is real inference and is excluded from the measured counter deltas.
        response = requests.post('http://127.0.0.1:8001/single', data=photograph, timeout=300)
        response.raise_for_status()
        assert not response.json()['results'][0]['error'], response.json()
    data = {'started': time.time(), 'source': SOURCE, 'warm_model': True, 'scrape_seconds': 5, 'phases': {}}
    print('Capturing 30 seconds of idle baseline...', flush=True)
    time.sleep(30)
    before = metrics()
    headers = {'X-Client-Category': 'demonstration'}
    for mode, path, body in [('single', '/single', photograph), ('bulk', '/bulk', buffer.getvalue())]:
        print(f'Running real {mode} CPU workload...', flush=True)
        start = time.time()
        response = requests.post('http://127.0.0.1:8001'+path, data=body, headers=headers, timeout=300)
        response.raise_for_status()
        results = response.json()['results']
        data['phases'][mode] = {'started': start, 'finished': time.time(), 'results': results}
        assert len(results) == (1 if mode == 'single' else 10)
        assert all(r['caption'] and not r['error'] for r in results), results
        if mode == 'single':
            time.sleep(10)
    print('Capturing 30 seconds of recovery...', flush=True)
    time.sleep(30)
    after = metrics()
    data['finished'] = time.time()
    data['counts'] = {}
    data['mean_inference_seconds'] = {}
    for mode in ('single', 'bulk'):
        data['counts'][mode] = {outcome: value(after, 'caption_images_total', mode=mode, outcome=outcome)-value(before, 'caption_images_total', mode=mode, outcome=outcome) for outcome in ('success','error')}
        count = value(after,'caption_inference_duration_seconds_count',mode=mode)-value(before,'caption_inference_duration_seconds_count',mode=mode)
        duration = value(after,'caption_inference_duration_seconds_sum',mode=mode)-value(before,'caption_inference_duration_seconds_sum',mode=mode)
        data['mean_inference_seconds'][mode] = duration/count
    assert data['counts'] == {'single': {'success': 1, 'error': 0}, 'bulk': {'success': 10, 'error': 0}}, 'Other workloads overlapped; repeat this capture without concurrent requests.'
    data['recovery'] = {name: value(after,name) for name in ('caption_active_requests','caption_active_inferences','caption_bulk_active')}
    assert all(v == 0 for v in data['recovery'].values())
    data['series'] = {}
    for name, query in QUERIES.items():
        response = requests.get('http://127.0.0.1:9090/api/v1/query_range', params={'query':query,'start':data['started'],'end':data['finished'],'step':'5s'}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        assert payload['status'] == 'success' and payload['data']['result'], (name,payload)
        data['series'][name] = {'query': query, 'result': payload['data']['result']}
    (HERE/'measurements.json').write_text(json.dumps(data, indent=2)+'\n')
    print('Saved actual captions, counters, timestamps and Prometheus samples.', flush=True)
    return data


def render(data):
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "da3408-matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "da3408-font-cache"))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    blue, orange, teal = '#0072B2', '#D55E00', '#009E73'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'figure.facecolor':'white','savefig.facecolor':'white'})
    date = datetime.fromtimestamp(data['started'], ZoneInfo('Asia/Kolkata')).strftime('%d %b %Y · %H:%M IST')
    def finish(fig, filename, title, subtitle, note):
        fig.suptitle(title, x=.08, ha='left', fontsize=19, fontweight='bold', y=.97)
        fig.text(.08,.91,subtitle,ha='left',fontsize=11,color='#444444')
        fig.text(.08,.045,note+'\nMeasured '+date+' · CPU-only BLIP · 5-second Prometheus samples · Linear axes',ha='left',fontsize=9,color='#444444')
        fig.subplots_adjust(left=.11,right=.96,bottom=.22,top=.82,hspace=.55)
        fig.savefig(HERE/filename,dpi=180)
        plt.close(fig)
    def shade(ax):
        start = data['phases']['bulk']['started']-data['started']
        end = data['phases']['bulk']['finished']-data['started']
        ax.axvspan(start,end,color='#E69F00',alpha=.17,label='Bulk request interval')
        ax.set_xlim(0,data['finished']-data['started'])
        ax.set_xlabel('Elapsed time from idle baseline (seconds)')
    def series(ax,key,scale=1,steps=False):
        for i,item in enumerate(data['series'][key]['result']):
            points=[(float(t)-data['started'],float(v)/scale) for t,v in item['values'] if v not in ('NaN','+Inf','-Inf')]
            assert points,key
            label=item['metric'].get('mode',{'cpu':'Host CPU','ram':'Host RAM used','rss':'Caption worker RSS'}.get(key,key))
            style = 0 if label == 'single' else (1 if label == 'bulk' else i%3)
            ax.plot([p[0] for p in points],[p[1] for p in points],label=label,color=[blue,orange,teal][style],linestyle=['-','--',':'][style],marker=['o','s','^'][style],markersize=3.5,linewidth=1.8,drawstyle='steps-post' if steps else 'default')
        shade(ax)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper left',frameon=True,fontsize=9)
    # 1. Isolate this run using counter deltas, not lifetime totals.
    fig,ax=plt.subplots(figsize=(10,6))
    x=np.arange(2)
    for j,outcome in enumerate(('success','error')):
        values=[data['counts'][m][outcome] for m in ('single','bulk')]
        bars=ax.bar(x+(j-.5)*.3,values,.3,label=outcome.title(),color=[blue,orange][j],hatch=['','///'][j],edgecolor='white')
        ax.bar_label(bars,labels=[str(int(v)) for v in values],padding=4)
    ax.set_xticks(x,['Single image','Bulk ZIP (10 images)']);ax.set_ylabel('Images processed (count)');ax.set_ylim(0,12);ax.legend()
    finish(fig,'01_processing_results.png','11 real captions. Zero image failures.','Counter changes from this isolated run: 1 single image + 10 ZIP images.','Counts exclude earlier workloads and model warm-up. All recovery active-work gauges returned to zero.')
    # 2. Differentiate model-only timing from end-to-end per-image processing.
    fig,axes=plt.subplots(1,2,figsize=(11,6))
    means=[data['mean_inference_seconds'][m] for m in ('single','bulk')]
    bars=axes[0].bar(['Single','Bulk'],means,color=[blue,orange],edgecolor='white',hatch=['','///'])
    axes[0].bar_label(bars,labels=[f'{v:.2f} s' for v in means],padding=4)
    axes[0].set_ylabel('Mean model generation time (seconds)');axes[0].set_ylim(0,max(means)*1.3);axes[0].set_title('Inference summary: Δsum / Δcount')
    for mode,color,marker in [('single',blue,'o'),('bulk',orange,'s')]:
        times=[r['seconds'] for r in data['phases'][mode]['results']]
        positions=[1] if mode=='single' else list(range(2,12))
        axes[1].plot(positions,times,marker=marker,color=color,label=mode,linestyle='None',markersize=7)
    axes[1].set_xlabel('Image order (1 single, then 10 bulk)');axes[1].set_ylabel('Per-image processing time (seconds)');axes[1].set_ylim(bottom=0);axes[1].set_title('Decode + model generation');axes[1].legend()
    finish(fig,'02_caption_latency.png','Warm CPU inference, measured per image','Two timings: model-only mean versus complete per-image processing.','The model was loaded before baseline. Points are actual returned image timings; no synthetic observations.')
    # 3. Pair workload rate and active-request recovery.
    fig,axes=plt.subplots(2,1,figsize=(11,8))
    series(axes[0],'throughput');axes[0].set_ylabel('Throughput (images/second)');axes[0].set_title('15-second rolling counter rate, grouped by mode')
    series(axes[1],'active',steps=True);axes[1].set_ylabel('Active uploads (count)');axes[1].set_title('Uploads complete and active requests return to zero')
    finish(fig,'03_throughput_and_recovery.png','Bulk work raises throughput, then completes','Orange shading marks the exact bulk request start and finish.','Rates lag events because of the 15-second window. A short single request may fall between 5-second scrapes.')
    # 4. Host CPU correlation, with honest attribution limits.
    fig,ax=plt.subplots(figsize=(11,6));series(ax,'cpu');ax.set_ylabel('Host CPU utilization (%)')
    ax.set_ylim(0,min(100,max(20,ax.get_ylim()[1]*1.1)))
    finish(fig,'04_cpu_and_bulk_activity.png','Host CPU during a real bulk workload','CPU = 100 × (1 − mean idle CPU rate), over a 15-second window.','Shading shows workload timing, not proof of causation. Host CPU includes monitoring and other machine activity.')
    # 5. Different scopes use separate axes and explicit units.
    fig,axes=plt.subplots(2,1,figsize=(11,8))
    series(axes[0],'rss',scale=1024**2,steps=True);axes[0].set_ylabel('Worker RSS (MiB)');axes[0].set_title('Process RSS: Python runtime + loaded model + working tensors')
    series(axes[1],'ram');axes[1].set_ylabel('Host RAM used (%)');axes[1].set_title('Host RAM = 100 × (1 − available / total)')
    finish(fig,'05_memory_and_bulk_activity.png','A warm model stays resident after processing','Worker memory and whole-host RAM measure different resource scopes.','Zero active requests does not mean unloaded weights. Plots show measured values without forcing a memory spike.')
    print('Rendered five plots from saved measurements.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',action='store_true',help='Run a real workload against the local stack (~90 seconds)')
    args=parser.parse_args()
    data=capture() if args.capture else json.loads((HERE/'measurements.json').read_text())
    render(data)
