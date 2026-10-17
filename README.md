# SMS Fraud Detection

Proof of concept that flags fraudulent SMS messages with a DistilBERT model fine-tuned with
LoRA, served through a FastAPI API and a static web page, deployed on **Vercel**.

![Screenshot of the web interface](docs/screenshot.png)

> **Status: proof of concept.** The model can be wrong in both directions. See
> [Model behaviour](#-model-behaviour) before relying on a verdict.

## 📋 Contents

- [Architecture](#-architecture)
- [Model behaviour](#-model-behaviour)
- [API](#-api)
- [Deploy on Vercel](#-deploy-on-vercel)
- [Configuration](#️-configuration)
- [Local development](#-local-development)
- [Project structure](#-project-structure)
- [Quality and CI](#-quality-and-ci)

## 🏗 Architecture

```
Vercel
├── CDN ─────────────── public/  (index.html, styles.css, i18n.js, app.js)
└── Python function ─── sms_fraud.main:app  (FastAPI)
        /v1/predict, /v1/predict/batch, /health, /ready, /docs
        └── ONNX Runtime + tokenizers
                └── model.onnx (fp32), downloaded on cold start to /tmp
                    from the GitHub release model-onnx-2cf2ba0, checksum verified
```

- **Model**: [`MrNWARMeilleur/sms_fraud_detection`](https://huggingface.co/MrNWARMeilleur/sms_fraud_detection),
  a LoRA adapter (r=8, `q_lin`/`v_lin`) on `distilbert-base-uncased`, revision `2cf2ba0` pinned.
- **Serving**: the adapter is merged into the base model and exported to ONNX in full precision.
  The weights are unchanged (no quantization), and the export is rejected if its probabilities
  differ from PyTorch by more than `1e-4` (measured gap on the real model: `3e-7`). Serving needs
  neither PyTorch nor transformers: ~190 MB of dependencies, well under Vercel's 500 MB limit.
- **Frontend and API share one origin**: no CORS configuration is needed.

## 🔎 Model behaviour

Every export scores a small hand-labelled set of English messages and records the result in the
release's `metadata.json` (`sanity_report`). It is a smoke check of 12 messages, not an evaluation.

| Message (abridged) | Expected | Fraud score |
|---|---|---|
| You've won a $1000 gift card… reply with your PIN | fraud | 1.000 |
| FREE entry in 2 a wkly comp to win FA Cup final tkts… | fraud | 1.000 |
| Mom, I lost my phone… send me $200 for rent | fraud | 0.998 |
| URGENT: your bank account has been suspended… | fraud | 0.000 |
| Your package could not be delivered. Pay $1.99… | fraud | 0.000 |
| IRS notice: you owe back taxes… | fraud | 0.002 |
| Your Netflix payment failed. Update your card… | fraud | 0.000 |
| Hey, are we still on for lunch tomorrow at 12? | legitimate | 0.053 |
| Reminder: your dentist appointment is on Monday | legitimate | 0.000 |
| Your Uber is arriving in 3 minutes. | legitimate | 0.000 |
| Can you pick up milk on your way home? | legitimate | 0.996 |
| Happy birthday! Have a great day, see you tonight | legitimate | 1.000 |

The model agrees with 6 of the 12 labels. It catches prize and "family emergency" scams, but
misses phishing styles that are probably absent from its training data, and it flags some
ordinary personal messages. Its scores are almost always close to 0 or 1, so the score should
not be read as a calibrated probability. The training data, the code and metrics are not part
of this repository, so this behaviour cannot be traced back to them here.

## 📡 API

Interactive documentation: `/docs`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (model loaded), returns the model version |
| `POST` | `/v1/predict` | Classify one message (1–2000 characters) |
| `POST` | `/v1/predict/batch` | Classify up to 32 messages |

```http
POST /v1/predict
Content-Type: application/json

{"message": "URGENT: your account is locked, verify at http://bit.ly/x"}
```

```json
{
  "label": "fraud",
  "is_fraud": true,
  "fraud_score": 0.982,
  "threshold": 0.5,
  "model_version": "MrNWARMeilleur/sms_fraud_detection@2cf2ba0+onnx",
  "latency_ms": 12.4
}
```

| Status | Meaning |
|---|---|
| `422` | Invalid input (empty, longer than 2000 characters, more than 32 messages) |
| `429` | Rate limit exceeded; see the `Retry-After` header |
| `503` | Model not loaded yet (cold start) |

Messages are never logged; only counts and latency are.

## 🚀 Deploy on Vercel

1. **Publish the ONNX model** (once per model revision): *Actions → Export ONNX model → Run
   workflow* on `main`. It exports the model, checks parity with PyTorch, smoke-tests the API and
   publishes the `model-onnx-2cf2ba0` release the API downloads. Releases are never overwritten.
2. **Import the repository** in Vercel (*Add New → Project*). The FastAPI preset is detected from
   `pyproject.toml` (`[tool.vercel] entrypoint = "sms_fraud.main:app"`). No build command and no
   environment variable are required.
3. **Deploy.** `public/` is served by the CDN, the API by a Python function (`vercel.json`:
   60 s max duration, tests/docs excluded from the bundle, security headers and CSP).
4. Every pull request gets a preview deployment. After each production deployment, the
   *Smoke test deployment* workflow checks the live site.

Recommended hardening:
- set `SMS_FRAUD_ONNX_SHA256` to the `model.onnx` checksum from the release's `metadata.json`;
- add a rate limiting rule in Vercel's firewall (the built-in limiter works per instance).

The first request after a period of inactivity starts a new instance, which downloads the model
(~270 MB). The page shows "Model warming up" and retries automatically during that time.

## ⚙️ Configuration

Environment variables, all optional:

| Variable | Default | Description |
|---|---|---|
| `SMS_FRAUD_BACKEND` | `onnx` | `onnx` (production) or `transformers` (PyTorch reference) |
| `SMS_FRAUD_ONNX_DIR` | `artifacts/onnx` (`/tmp/…` on Vercel) | Directory of the ONNX artifacts |
| `SMS_FRAUD_ONNX_URL` | `model-onnx-2cf2ba0` release | Where to download the artifacts when missing |
| `SMS_FRAUD_ONNX_SHA256` | — | Expected checksum of `model.onnx` |
| `SMS_FRAUD_THRESHOLD` | `0.5` | Minimum fraud score to flag a message |
| `SMS_FRAUD_MAX_TOKENS` | `512` | Truncation length |
| `SMS_FRAUD_RATE_LIMIT_PER_MINUTE` | `60` | Prediction requests per client per minute (`0` disables) |
| `SMS_FRAUD_TRUST_PROXY_HEADERS` | `true` on Vercel | Identify clients by `X-Real-IP` / `X-Forwarded-For` |
| `SMS_FRAUD_CORS_ORIGINS` | `[]` | Allowed origins, JSON list (empty = same origin only) |
| `SMS_FRAUD_DOCS_ENABLED` | `true` | Expose `/docs` and `/openapi.json` |
| `SMS_FRAUD_FRONTEND_DIR` | — | Serve the static frontend from the API (local development) |
| `SMS_FRAUD_LOG_LEVEL` | `INFO` | Log level |
| `SMS_FRAUD_BASE_MODEL`, `SMS_FRAUD_ADAPTER_REPO`, `SMS_FRAUD_ADAPTER_REVISION` | pinned | Model source (export and `transformers` backend) |

## 💻 Local development

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
python -m playwright install chromium        # for the browser tests

python -m sms_fraud.export                   # export + parity check -> artifacts/onnx
SMS_FRAUD_FRONTEND_DIR=public python -m sms_fraud   # http://127.0.0.1:8080
python scripts/smoke_test.py                 # smoke test a running instance
```

Without the export step, the API downloads the published release on first start.

| File | Contents |
|---|---|
| `requirements.txt` (= `pyproject.toml` dependencies) | Runtime: ONNX Runtime, tokenizers, FastAPI |
| `requirements-ml.txt` | + PyTorch, transformers, PEFT: export and reference backend |
| `requirements-dev.txt` | + pytest, ruff, Playwright |

## 📁 Project structure

```
.
├── sms_fraud/               # API package
│   ├── main.py              # FastAPI app factory, routes, rate limiting hook
│   ├── schemas.py           # request/response models
│   ├── config.py            # environment-based settings
│   ├── onnx_predictor.py    # ONNX Runtime backend (production)
│   ├── predictor.py         # PyTorch reference backend
│   ├── artifacts.py         # download and checksum of ONNX artifacts
│   ├── export.py            # fp32 ONNX export, parity check, sanity report
│   └── ratelimit.py         # per-client rate limiter
├── public/                  # static frontend (served by Vercel's CDN)
├── tests/                   # pytest suite; tests/e2e: Playwright browser tests
├── scripts/smoke_test.py    # end-to-end check of a running instance
├── vercel.json              # function settings and security headers
├── .github/workflows/       # CI, ONNX export/release, deployment smoke test
└── docs/AUDIT.md            # technical audit and roadmap
```

## ✅ Quality and CI

```bash
ruff check . && ruff format --check .
pytest -m "not e2e"     # unit and API tests, no network needed
pytest -m e2e           # browser tests under the production CSP
```

GitHub Actions runs lint, unit tests, browser tests, a runtime size guard (no torch, < 300 MB),
the ONNX export of the real model with its parity check on pull requests touching inference
code, and a smoke test of each production deployment.

## 📄 License

No license has been chosen yet.
