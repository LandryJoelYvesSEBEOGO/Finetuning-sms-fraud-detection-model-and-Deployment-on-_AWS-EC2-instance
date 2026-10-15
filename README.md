# SMS Fraud Detection - Machine Learning Project

<img src="sms_fraud_detection.png" alt="Texte alternatif" width="1000">

This project implements an SMS fraud detection system using a Machine Learning model based on DistilBERT. The system is deployed via Docker and can be easily put into production on AWS.
 <img src="project_Pipepline.jpg" alt="Texte alternatif" width="1000">
 
## 🌟 Context and Motivation

In an increasingly connected world, SMS fraud has become a major problem:

- **Increase in Fraud**: In 2023, more than 60% of financial fraud starts with a fraudulent SMS
- **Economic Cost**: Financial losses due to SMS fraud amount to several billion euros per year
- **Social Impact**: Elderly and vulnerable people are particularly affected by these scams

Our system addresses these challenges by:
- Analyzing suspicious SMS messages in real-time
- Using artificial intelligence to detect fraudulent patterns
- Providing a simple interface for non-technical users
- Offering a robust API for integration with other systems

### Technological Innovation

The project uses cutting-edge technologies:
- **DistilBERT**: A powerful and lightweight language model
- **PEFT (Parameter-Efficient Fine-Tuning)**: Model optimization for specific fraud cases
- **Cloud Native Architecture**: Flexible and scalable deployment

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Web Interface](#web-interface)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## 🎯 Project Overview

The system analyzes SMS messages to detect fraud attempts using a fine-tuned Machine Learning model. It includes:
- A FastAPI REST API
- A user-friendly web interface
- Nginx reverse proxy
- Containerized architecture with Docker

## 🏗 Architecture

The project consists of three main services:

1. **API (FastAPI)**:
   - Port: 8080
   - Handles ML model and predictions
   - Based on DistilBERT with PEFT fine-tuning

2. **Web Interface**:
   - Port: 8000
   - Responsive user interface
   - Communication with API via fetch

3. **Nginx**:
   - Port: 80
   - Reverse proxy
   - Routing management between API and web interface

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.8 or higher (for local development)
- Git
- At least 4GB of RAM available
- Internet connection (for downloading models)

## 🚀 Installation

1. Clone the repository:
```bash
git clone [REPO_URL]
cd sms-fraud-detection
```

2. Build and start containers:
```bash
docker-compose up --build
```

## ⚙️ Configuration

The API is configured with environment variables prefixed with `SMS_FRAUD_`:

| Variable | Default | Description |
|---|---|---|
| `SMS_FRAUD_BASE_MODEL` | `distilbert-base-uncased` | Base model |
| `SMS_FRAUD_ADAPTER_REPO` | `MrNWARMeilleur/sms_fraud_detection` | LoRA adapter (Hub repo or local path) |
| `SMS_FRAUD_ADAPTER_REVISION` | pinned commit | Adapter revision |
| `SMS_FRAUD_THRESHOLD` | `0.5` | Minimum fraud probability to flag a message |
| `SMS_FRAUD_MAX_TOKENS` | `512` | Truncation length |
| `SMS_FRAUD_CORS_ORIGINS` | `[]` | Allowed origins, JSON list (empty = same-origin only) |
| `SMS_FRAUD_DOCS_ENABLED` | `true` | Expose `/docs` and `/openapi.json` |
| `SMS_FRAUD_LOG_LEVEL` | `INFO` | Log level |

### Local development

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

python -m sms_fraud                     # API on http://127.0.0.1:8080
python scripts/smoke_test.py            # smoke test a running instance
pytest && ruff check . && ruff format --check .
```

## 💻 Usage

1. Access the web interface: `http://your-ip`
2. Enter an SMS message in the text area
3. Click "Analyze"
4. The result will appear in the "Result" section

## 📁 Project Structure

```
.
├── sms_fraud/             # API package
│   ├── main.py            # FastAPI app factory and routes
│   ├── predictor.py       # model loading and inference
│   ├── schemas.py         # request/response models
│   └── config.py          # environment-based settings
├── tests/                 # pytest suite (no network needed)
├── scripts/smoke_test.py  # end-to-end check of a running instance
├── interface-web/         # static web page
├── Dockerfile, docker-compose.yml, nginx.conf
└── docs/AUDIT.md          # technical audit and roadmap
```

## 📡 API Reference

Interactive documentation is available at `/docs` when enabled.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (model loaded) |
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
  "model_version": "MrNWARMeilleur/sms_fraud_detection@2cf2ba0",
  "latency_ms": 41.7
}
```

Invalid input (empty, too long, too many messages) returns `422`; `503` means the model is not loaded yet.

## 🌐 Web Interface

The web interface offers:
- Responsive design
- Input validation
- Visual feedback of results
- Error handling
- Multilingual support (EN/FR)

## 🚢 Deployment

### AWS Deployed Architecture

Our solution is currently deployed on AWS with the following architecture:

#### Infrastructure
- **EC2 Instance**: 
  - Type: t2.medium (2 vCPU, 4GB RAM)
  - OS: Ubuntu 22.04 LTS
  - Region: eu-west-3 (Paris)

#### Security
- **Security Groups**:
  - Inbound: Ports 80 (HTTP), 443 (HTTPS), 22 (SSH)
  - Outbound: All traffic allowed

#### Network
- **Dedicated VPC**:
  - Public and private subnets
  - NAT Gateway for Internet access

#### Monitoring
- **CloudWatch**:
  - Performance metrics
  - Resource consumption alerts
  - Container logs

#### Current Deployment
1. **Server Preparation**:
```bash
# System update
sudo apt-get update && sudo apt-get upgrade -y

# Docker and Docker Compose installation
sudo apt-get install docker.io docker-compose -y
```

2. **SSL/TLS Configuration**:
```bash
# Certbot installation
sudo apt-get install certbot python3-certbot-nginx -y

# Certificate generation
sudo certbot --nginx -d your-domain.com
```

3. **Container Deployment**:
```bash
# Repository cloning
git clone https://github.com/your-repo/sms-fraud-detection.git
cd sms-fraud-detection

# Environment variables configuration
cp .env.example .env
nano .env

# Deployment
docker-compose -f docker-compose.prod.yml up -d
```

### Performance and Scalability

The deployed system offers:
- **Response Time**: < 500ms for SMS analysis
- **Capacity**: Up to 100 simultaneous requests
- **Availability**: 99.9% uptime

### Maintenance

Automated maintenance scripts are in place:
```bash
# Daily log backup
0 0 * * * docker-compose logs > /backup/logs/$(date +\%Y\%m\%d).log

# Weekly container updates
0 0 * * 0 docker-compose pull && docker-compose up -d
```

## 🔧 Troubleshooting

### Common Issues

1. **Failed to fetch**:
   - Check if API is accessible
   - Check Nginx logs
   - Verify CORS configuration

2. **Model Errors**:
   - Check available disk space
   - Check available RAM
   - Consult API logs

### Logs

To view logs:
```bash
# API logs
docker-compose logs api

# Web logs
docker-compose logs web

# Nginx logs
docker-compose logs nginx
```

## 🔒 Security

- CORS configured for production
- Input validation
- No sensitive data storage
- Containers with non-root users

## 📄 License

This project is under MIT license. See the [LICENSE](LICENSE) file for more details.

## 👥 Contact

For any questions or suggestions:
- **Email**: your.email@domain.com
- **LinkedIn**: [Your Profile](https://linkedin.com/in/your-profile)
- **GitHub**: [Your Account](https://github.com/your-account) 
