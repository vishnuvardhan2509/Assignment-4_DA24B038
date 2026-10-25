"""CPU-only captioning, bounded uploads, and process-wide Prometheus instrumentation."""
from __future__ import annotations
import io
import os
import threading
import time
import warnings
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable

import psutil
from PIL import Image, ImageOps, UnidentifiedImageError
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Summary, start_http_server

MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_ZIP_BYTES = 50 * 1024 * 1024
MAX_BATCH = 100
MAX_EXPANDED_BYTES = 150 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
MODES = ('single', 'bulk')
CLIENTS = ('interactive', 'demonstration', 'test')

class InputError(ValueError):
    pass

class Metrics:
    def __init__(self, registry=None):
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.images = Counter('caption_images', 'Attempted images by result.', ['mode', 'outcome', 'client'], registry=r)
        self.errors = Counter('caption_errors', 'Failures by bounded operation.', ['mode', 'operation', 'client'], registry=r)
        self.requests = Counter('caption_requests', 'Completed upload requests.', ['mode', 'outcome', 'client'], registry=r)
        self.active = Gauge('caption_active_requests', 'Uploads being processed.', ['mode'], registry=r)
        self.inflight = Gauge('caption_active_inferences', 'CPU inferences in progress.', registry=r)
        self.bulk = Gauge('caption_bulk_active', 'Bulk requests in progress.', registry=r)
        self.batch = Gauge('caption_last_batch_size', 'Number of candidate images in the last bulk request.', registry=r)
        self.memory = Gauge('caption_process_memory_bytes', 'Caption process RSS.', registry=r)
        self.model_ready = Gauge('caption_model_ready', 'Whether pretrained CPU model is loaded.', registry=r)
        self.inference = Histogram('caption_inference_seconds', 'Model generation latency.', ['mode'], buckets=(.1,.25,.5,1,2,4,8,16,32,64,128), registry=r)
        self.processing = Histogram('caption_processing_seconds', 'Decode plus generation time per image.', ['mode'], buckets=(.1,.5,1,2,4,8,16,32,64,128), registry=r)
        self.summary = Summary('caption_inference_duration_seconds', 'Generation duration sum and count.', ['mode'], registry=r)
        self.upload_duration = Summary('caption_request_duration_seconds', 'Whole upload duration sum and count.', ['mode'], registry=r)
        self.batch_sizes = Histogram('caption_batch_images', 'Candidate images per bulk upload.', buckets=(1,5,10,20,50,100), registry=r)
        self.started = Gauge('caption_started_timestamp_seconds', 'Metrics service start time.', registry=r)
        self.started.set(time.time())
        for mode in MODES:
            self.active.labels(mode).set(0)
            self.inference.labels(mode)
            self.processing.labels(mode)
            self.summary.labels(mode)
            self.upload_duration.labels(mode)
            for client in CLIENTS:
                for outcome in ('success','error'):
                    self.images.labels(mode,outcome,client)
                    self.requests.labels(mode,outcome,client)
                for op in ('validation','decode','inference','model'):
                    self.errors.labels(mode,op,client)
        self.update_memory()

    def update_memory(self):
        self.memory.set(psutil.Process().memory_info().rss)

    def serve(self, port=8000):
        # Bind to all interfaces only when explicitly configured (e.g. Docker).
        self.server, self.thread = start_http_server(port, addr=os.getenv('METRICS_HOST','127.0.0.1'), registry=self.registry)
        def sample():
            while self.thread.is_alive():
                self.update_memory()
                time.sleep(1)
        threading.Thread(target=sample, daemon=True, name='caption-rss').start()


def decode_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise InputError('Image must contain 1 byte to 15 MiB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as check:
                check.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                rgb = ImageOps.exif_transpose(image).convert('RGB')
                rgb.thumbnail((2048,2048))
                return rgb.copy()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise InputError('Invalid, corrupt, or excessively large image.') from e


def zip_members(data: bytes):
    """Validate all archive metadata before reading; never extract to the filesystem."""
    if not data or len(data) > MAX_ZIP_BYTES:
        raise InputError('ZIP must contain 1 byte to 50 MiB.')
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        entries = archive.infolist()
        if len(entries) > 1000:
            raise InputError('Archive contains too many entries.')
        candidates=[]
        expanded=0
        for item in entries:
            path=PurePosixPath(item.filename.replace('\\','/'))
            if item.is_dir():
                continue
            if path.is_absolute() or '..' in path.parts:
                raise InputError('Unsafe archive member path.')
            expanded += item.file_size
            if expanded > MAX_EXPANDED_BYTES:
                raise InputError('ZIP expanded size exceeds 150 MiB.')
            if item.flag_bits & 1:
                raise InputError('Encrypted ZIP entries are not supported.')
            if path.suffix.lower() in ('.png','.jpg','.jpeg','.webp','.bmp'):
                if item.file_size > MAX_IMAGE_BYTES:
                    raise InputError('An image exceeds 15 MiB.')
                candidates.append(item)
        if not candidates:
            raise InputError('ZIP contains no supported images.')
        if len(candidates) > MAX_BATCH:
            raise InputError('Maximum batch size is 100 images.')
        return archive, candidates
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
        raise InputError('Invalid or corrupt ZIP.') from e

class CaptionModel:
    def __init__(self):
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        torch.set_num_threads(max(1,min(int(os.getenv('CPU_THREADS','2')),4)))
        self.torch = torch
        model = os.getenv('CAPTION_MODEL','Salesforce/blip-image-captioning-base')
        revision = None if os.path.isdir(model) else os.getenv('CAPTION_MODEL_REVISION','82a37760796d32b1411fe092ab5d4e227313294b')
        self.processor=BlipProcessor.from_pretrained(model, revision=revision)
        self.model=BlipForConditionalGeneration.from_pretrained(model, revision=revision, use_safetensors=False).to('cpu').eval()

    def __call__(self, image):
        inputs=self.processor(images=image,return_tensors='pt').to('cpu')
        with self.torch.inference_mode():
            output=self.model.generate(**inputs,max_new_tokens=30,num_beams=1,do_sample=False)
        text=self.processor.decode(output[0],skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError('Model generated an empty caption.')
        return text

@dataclass
class Result:
    filename: str
    caption: str = ''
    seconds: float = 0.0
    error: str = ''

class CaptionService:
    def __init__(self, metrics=None, model_factory: Callable=CaptionModel):
        self.metrics=metrics or Metrics()
        self.model_factory=model_factory
        self.model=None
        self.lock=threading.Lock()

    def load(self):
        with self.lock:
            if self.model is None:
                self.model=self.model_factory()
                self.metrics.model_ready.set(1)
                self.metrics.update_memory()

    @contextmanager
    def request(self, mode, client):
        if mode not in MODES or client not in CLIENTS:
            raise ValueError('Unknown mode or client category.')
        m=self.metrics
        start=time.perf_counter()
        m.active.labels(mode).inc()
        if mode=='bulk': m.bulk.inc()
        try:
            yield
        finally:
            m.active.labels(mode).dec()
            if mode=='bulk': m.bulk.dec()
            m.upload_duration.labels(mode).observe(time.perf_counter()-start)
            m.update_memory()

    def _image(self, data, name, mode, client):
        m=self.metrics
        start=time.perf_counter()
        operation='decode'
        try:
            image=decode_image(data)
            operation='model'
            self.load()
            operation='inference'
            # Serialize inference across users to bound CPU/RAM.
            with self.lock:
                m.inflight.inc()
                infer=time.perf_counter()
                try:
                    caption=self.model(image)
                finally:
                    elapsed=time.perf_counter()-infer
                    m.inflight.dec()
                    m.inference.labels(mode).observe(elapsed)
                    m.summary.labels(mode).observe(elapsed)
            m.images.labels(mode,'success',client).inc()
            return Result(name,caption,time.perf_counter()-start)
        except Exception as e:
            m.errors.labels(mode,operation,client).inc()
            m.images.labels(mode,'error',client).inc()
            # Detailed type is useful locally; never send file paths/client ids as labels.
            if isinstance(e,InputError): message=str(e)
            elif isinstance(e,MemoryError): message='Insufficient memory; reduce image or batch size.'
            else: message=f'{operation.capitalize()} failed ({type(e).__name__}). See server logs.'
            import logging
            logging.getLogger(__name__).exception('Caption failure: %s', operation)
            return Result(name,seconds=time.perf_counter()-start,error=message)
        finally:
            m.processing.labels(mode).observe(time.perf_counter()-start)
            m.update_memory()

    def single(self,data,name='image',client='interactive'):
        with self.request('single',client):
            result=self._image(data,name,'single',client)
            self.metrics.requests.labels('single','error' if result.error else 'success',client).inc()
            return result

    def bulk(self,data,client='interactive',progress=None):
        with self.request('bulk',client):
            try:
                archive,entries=zip_members(data)
            except (InputError,MemoryError):
                self.metrics.errors.labels('bulk','validation',client).inc()
                self.metrics.requests.labels('bulk','error',client).inc()
                raise
            self.metrics.batch.set(len(entries))
            self.metrics.batch_sizes.observe(len(entries))
            results=[]
            with archive:
                for i,item in enumerate(entries):
                    try:
                        raw=archive.read(item)
                        result=self._image(raw,item.filename,'bulk',client)
                    except Exception as e:
                        self.metrics.errors.labels('bulk','decode',client).inc()
                        self.metrics.images.labels('bulk','error',client).inc()
                        result=Result(item.filename,error=f'ZIP member unreadable ({type(e).__name__}).')
                    results.append(result)
                    if progress: progress(i+1,len(entries),result)
            self.metrics.requests.labels('bulk','error' if any(r.error for r in results) else 'success',client).inc()
            return results

import dataclasses
import json
import logging
import subprocess
import sys
import signal
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
import requests

class WorkerServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,service):
        super().__init__(address,Handler)
        self.service=service
        self.slots=threading.BoundedSemaphore(2)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,fmt,*args): logging.info(fmt,*args)
    def reply(self,status,data):
        raw=json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
    def do_GET(self):
        if self.path=='/health':
            self.reply(200,{'status':'ok','device':'cpu','model_ready':self.server.service.model is not None})
        else: self.reply(404,{'error':'Unknown endpoint'})
    def do_POST(self):
        path=urlsplit(self.path)
        if path.path not in ('/single','/bulk'):
            self.reply(404,{'error':'Unknown endpoint'}); return
        client=self.headers.get('X-Client-Category','interactive')
        if client not in ('interactive','demonstration','test'):
            self.reply(400,{'error':'Unknown client category'}); return
        try: size=int(self.headers.get('Content-Length','0'))
        except ValueError: size=0
        limit=MAX_IMAGE_BYTES if path.path=='/single' else MAX_ZIP_BYTES
        if not 0<size<=limit:
            self.reply(413,{'error':'Missing, empty, or oversized request body'}); return
        if not self.server.slots.acquire(blocking=False):
            self.reply(429,{'error':'Worker busy. Retry after current uploads complete.'}); return
        try:
            self.connection.settimeout(30)
            data=self.rfile.read(size)
            if len(data)!=size: raise InputError('Incomplete upload')
            if path.path=='/single':
                name=parse_qs(path.query).get('filename',['image'])[0][:255]
                results=[self.server.service.single(data,name,client)]
            else: results=self.server.service.bulk(data,client)
            self.reply(200,{'results':[dataclasses.asdict(r) for r in results]})
        except InputError as e: self.reply(400,{'error':str(e)})
        except (MemoryError,TimeoutError): self.reply(503,{'error':'Worker resource limit or upload timeout'})
        finally: self.server.slots.release()


class CaptionClient:
    def __init__(self): self.url=os.getenv('CAPTION_WORKER_URL','http://127.0.0.1:8001')
    def _request(self,path,data,client,name=None):
        try:
            response=requests.post(self.url+path,data=data,params={'filename':name} if name else None,
                headers={'X-Client-Category':client,'Content-Type':'application/octet-stream'},timeout=(10,600))
            payload=response.json()
            if not response.ok: raise InputError(payload.get('error',f'Worker HTTP {response.status_code}'))
            return [Result(**r) for r in payload['results']]
        except requests.RequestException as e:
            raise InputError('CPU worker unavailable or timed out. Check backend logs and retry.') from e
    def single(self,data,name='image',client='interactive'):
        return self._request('/single',data,client,name)[0]
    def bulk(self,data,client='interactive',progress=None):
        results=self._request('/bulk',data,client)
        if progress:
            for i,r in enumerate(results): progress(i+1,len(results),r)
        return results

def run_ui():
    import pandas as pd
    import streamlit as st
    st.set_page_config(page_title='Caption Observatory', page_icon='🔭', layout='wide')

    @st.cache_resource
    def service():
        return CaptionClient()

    svc=service()
    st.title('Caption Observatory')
    st.caption('CPU image captioning · Single images and sequential ZIP batches · Live Prometheus monitoring')
    with st.sidebar:
        st.header('Workload settings')
        mode=st.radio('Mode',['Single','Bulk'])
        client=st.selectbox('Client category',['interactive','demonstration'])
        st.caption('Client categories are bounded. IP addresses and session IDs are never exported as labels.')
        st.info('CPU only · 15 MiB/image · 50 MiB/ZIP · 100 images/batch')
        st.caption('Model weights load on first inference. Monitoring uses port 8000.')
    if mode=='Single':
        upload=st.file_uploader('Upload one image',type=['png','jpg','jpeg','webp','bmp'])
    else:
        upload=st.file_uploader('Upload a ZIP of images',type=['zip'])
    run=st.button('Generate captions',type='primary',disabled=upload is None)
    if run and upload:
        raw=upload.getvalue()
        with st.status('Processing on CPU…',expanded=True) as status:
            try:
                if mode=='Single':
                    result=svc.single(raw,upload.name,client)
                    results=[result]
                    if not result.error:
                        st.image(io.BytesIO(raw),width=420)
                        st.subheader(result.caption)
                else:
                    st.write('The worker processes ZIP images sequentially; results appear when the batch completes.')
                    bar=st.progress(0)
                    line=st.empty()
                    def update(i,n,result):
                        bar.progress(i/n)
                        line.write(f'{i}/{n}: {result.filename} — {result.error or result.caption}')
                    results=svc.bulk(raw,client,update)
                errors=sum(bool(r.error) for r in results)
                status.update(label=f'Finished: {len(results)-errors} captions, {errors} failures',state='error' if errors else 'complete')
                st.session_state['results']=[dataclasses.asdict(r) for r in results]
            except (InputError,MemoryError) as e:
                st.error(str(e) if isinstance(e,InputError) else 'Not enough memory. Try a smaller upload.')
                status.update(label='Upload rejected',state='error')
    if 'results' in st.session_state:
        frame=pd.DataFrame(st.session_state['results'])
        st.dataframe(frame,hide_index=True,width='stretch')
        st.download_button('Download results CSV',frame.to_csv(index=False),'captions.csv','text/csv')
    st.caption('Inference time measures model.generate; per-image processing includes decode and initial model loading. Failures appear in the results table.')

def run_worker():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx2').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    metrics = Metrics()
    metrics.serve(8000)
    WorkerServer(('127.0.0.1', 8001), CaptionService(metrics)).serve_forever()


def launch():
    """One command starts a metrics worker and the Streamlit interface."""
    os.environ.setdefault('HF_HOME', str(Path(__file__).parent / '.cache/huggingface'))
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    worker = subprocess.Popen([sys.executable, __file__, '--worker'])
    ui = None
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        for _ in range(30):
            if worker.poll() is not None:
                raise RuntimeError('CPU worker failed to start; inspect the error above.')
            try:
                requests.get('http://127.0.0.1:8001/health', timeout=1).raise_for_status()
                break
            except requests.RequestException:
                time.sleep(1)
        else:
            raise RuntimeError('CPU worker readiness timed out.')
        ui = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', __file__,
            '--server.address', '127.0.0.1', '--server.port', '8501',
            '--server.headless', 'true', '--server.maxUploadSize', '50',
            '--browser.gatherUsageStats', 'false', '--', '--ui'])
        while ui.poll() is None:
            if worker.poll() is not None:
                raise RuntimeError('CPU worker stopped unexpectedly.')
            time.sleep(1)
        if ui.returncode:
            raise SystemExit(ui.returncode)
    except KeyboardInterrupt:
        pass
    finally:
        for child in [ui, worker]:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


if __name__ == '__main__':
    if '--worker' in sys.argv:
        run_worker()
    elif '--ui' in sys.argv:
        run_ui()
    else:
        launch()
