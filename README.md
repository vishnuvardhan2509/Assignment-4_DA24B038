# DA3408 Assignment 4 — Image Captioning & Monitoring

CPU-only BLIP image captioning with single-image and ZIP uploads, Prometheus
metrics, node_exporter, a 20-panel Grafana dashboard, and Alertmanager.

## Submission files

```text
Assignment-4/
├── app.py
├── requirements.txt
├── prometheus.yml
├── compose.yml
├── alert_rules.yml
├── alertmanager.yml
├── README.md
├── demo.mp4
├── monitoring/
│   └── grafana-dashboard.json
└── plots/
    └── five measured charts + data and plotting instructions
```

`app.py` contains the UI, CPU model, input validation, instrumentation and local
inference worker. No other Python files are needed. Compose generates Grafana's
small provisioning files at runtime, so its datasource and dashboard load automatically.

## Run

Requires Linux/Ubuntu x86-64, Docker Engine, Docker Compose v2, internet access,
and at least 4 GB available RAM. From the repository directory:

```bash
docker compose up -d
docker compose logs -f app
```

The first start installs the pinned Python packages into a persistent Docker
volume. On first caption generation, the public BLIP model downloads about 1 GB
of weights. Allow several minutes for this initial setup. Later starts reuse
installed packages and cached weights. Required download sources are PyPI,
PyTorch CPU wheels and Hugging Face (including its artifact CDN).

| Service | Local address |
|---|---|
| Streamlit | `http://127.0.0.1:8501` |
| Grafana | `http://127.0.0.1:3000` |
| Prometheus | `http://127.0.0.1:9090` |
| Application metrics | `http://127.0.0.1:8000/metrics` |
| Alertmanager | `http://127.0.0.1:9093` |

These are local addresses on the machine running Docker. The cloud onboarding
interface does not provide a localhost preview. For a remote Ubuntu server, use
an SSH port tunnel. Listeners intentionally bind to loopback.

Grafana initially uses **admin / admin**; change its password when prompted.
Open **Dashboards → DA3408 → Caption Observatory · CPU Image Captioning**.
The dashboard refreshes every **5 seconds**. Its exported JSON is in `monitoring/`.

Upload a JPG/PNG/WebP/BMP in **Single** mode, or a ZIP containing images in **Bulk**
mode, then select **Generate captions**. Bulk inference is sequential on CPU.
The UI shows captions, durations and failures, with a CSV download. Limits:
15 MiB/image, 50 MiB/ZIP, 100 images and 150 MiB expanded archive data. Corrupt
images fail individually; empty/unsafe archives are rejected. Generation is
serialized to bound CPU/RAM. The model remains loaded between requests.

```bash
# Confirm readiness, scrape targets and stop without deleting data:
docker compose ps
curl -fsS http://127.0.0.1:8000/metrics
curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=up'
docker compose stop
```

The app, node, Prometheus and Alertmanager targets should all report `1`.
Avoid `docker compose down -v` unless you intend to delete monitoring history
and the installed dependency volume. Node metrics describe the available Docker
host/cloud namespace; overlay filesystem statistics are enabled explicitly.

## Metrics and dashboard

- **Counters:** attempted images, successful/failed captions, upload requests and
  errors. Labels use single/bulk mode, outcome, operation and bounded client
  categories; raw IPs/session IDs are not exported.
- **Gauges:** active uploads/inferences/bulk work, last batch size, model readiness
  and process memory.
- **Histograms:** inference/processing latency and batch sizes.
- **Summaries:** inference/request durations with `_sum` and `_count`.
  Counters use `_total`; summaries do not replace `_count` with `_total`.

The dashboard covers health, outcomes, mode throughput, mean/P50/P95/P99 latency,
active work, bulk size, CPU/RAM/disk and bulk/resource correlation. It uses named
axes, explicit units and linear scales, legends, accessible colors with line
patterns/markers, descriptive titles and explanations. Latency quantiles need
actual inference observations; run a workload before evaluating those panels.

Manual calculation examples:

- Throughput: 12 images over 60 seconds ≈ **0.2 images/s** (`rate` extrapolates
  scrape boundaries and handles counter resets).
- Mean inference: 22 generation seconds / 11 images = **2 seconds/image**.
- CPU: average idle rate 0.75 → `100 × (1 − 0.75)` = **25%**.
- RAM: 24 GiB available / 32 GiB total → **25% used**.
- Histogram P95: if cumulative counts are 80 at 2 s and 98 at 4 s out of 100,
  interpolation gives `2 + (95−80)/(98−80) × 2` ≈ **3.67 s**. This is an estimate.

Alerts cover application downtime, mean inference >10 s, image error ratio >20%
with at least five attempts, CPU >85%, and RAM >85%. Prometheus evaluates the rules
and sends alerts to Alertmanager. The local receiver has no external email/webhook;
alerts can be inspected in Alertmanager. Rule files include their thresholds and delays.

**AI attribution prompt for dashboard PromQL and filesystem regex:**
“Build a professional DA3408 dashboard using caption_* metrics and node_exporter;
include health, mode throughput, errors, mean and histogram quantiles, bulk size,
CPU/RAM/disk and bulk correlation. Explain every query, use bounded labels,
accessible colors, explicit units, linear axes, and 5s refresh. Include overlay
filesystems for a container-based cloud machine while excluding pseudo filesystems.”
Alert-expression attribution is included as comments in `alert_rules.yml`.

## Plots

Open [plots/](plots/README.md) for five self-contained charts covering processing
results, latency, throughput/recovery, CPU and memory during a real single-plus-10-image
CPU workload. Each chart includes titles, units, legends and explanatory context.
The measured data and optional plotting script are included to reproduce the figures;
plotting packages are not needed to run the application.

## Demonstration and validation

[demo.mp4](demo.mp4) is the genuine approximately 60-second dashboard recording,
including a real single-plus-10-image CPU workload and five-second refreshes.
It was recorded before repository cleanup; the same UI, model, metrics and
panels are retained. The simplified Docker version was retested: 11/11 CPU captions, both browser
upload modes, all four scrape targets and all 26 dashboard queries passed.

Original measured workload: **11/11 captions**, zero failures; warm 10-image bulk
processing took **15.06 s**, with **1.50 s/image** mean generation. The first
caption was “two orange flowers.” These are measurements from the recorded
validation, not promises about performance on another machine. The consolidated app passed the existing 19 regression tests, and its monitoring
configuration checks passed. The original validation also verified a live downtime
alert sent to Alertmanager. Extra test scripts and raw evidence were removed from this
minimal submission; Git history retains the earlier version.

To make your own recording, show Grafana idle, upload one image and then a ZIP
of at least 10 images in Streamlit, show throughput/latency and hardware charts,
and finish with active requests returning to zero. Record about one minute with
the dashboard refresh set to five seconds.
