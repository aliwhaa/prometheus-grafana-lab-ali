

# 🌐 IoT ML-Model Monitoring Stack

This repository contains an end-to-end **MLOps and Monitoring pipeline** for an IoT-based Machine Learning application. It features automated data ingestion, model retraining, real-time metrics, and a full alerting suite.

---

## 🏗️ System Architecture

The system is composed of four main microservices interconnected via a dedicated Docker bridge network:

* **ML App (FastAPI)**: Simulates an IoT gateway that pulls data from a Data Lake, runs inferences, and exports metrics.
* **Prometheus**: Scrapes metrics from the ML App and evaluates alerting rules.
* **Alertmanager**: Routes triggered alerts to **Slack**.
* **Grafana**: Provides a visual dashboard for model performance and system health.

---

## 🚦 Quick Start

### 1. Prerequisites

* **Docker** and **Docker Compose** installed.
* A **Slack Incoming Webhook URL**.

### 2. Configuration

Before launching, update your Slack Webhook in two locations:

1. **`app/python-exporter/main.py`**: Update the `SLACK_WEBHOOK` variable.
2. **`alertmanager/config.yml`**: Update the `api_url` under `slack_configs`.

### 3. Deployment

Run the following command in the root directory:

```bash
docker compose up -d --build

```

---

## 📊 Monitoring Dashboard

Once the stack is running, access **Grafana** at `http://localhost:3000` (Credentials: `admin`/`admin`). The dashboard is **automatically provisioned** and includes:

* **Model Accuracy**: Gauge showing real-time performance.
* **Inference Volume**: Time-series graph of request rates.
* **Data Lake Health**: Tracking 503 errors and ingestion latency.
* **Drift Detection**: Binary indicator for feature distribution shifts.

---

## 🚨 Alerting Rules

The system is configured with the following **Prometheus Alerts**:

| Alert Name | Condition | Severity |
| --- | --- | --- |
| **DataLakeUnavailable** | Increase in 503 errors > 0 | **Critical** |
| **AccuracyLow** | Accuracy drops below **80%** | **Critical** |
| **DistributionDrift** | Feature mean deviates from history | **Warning** |
| **HighResponseDelay** | Ingestion latency > 2s | **Warning** |

---

## ☁️ AWS EC2 Deployment

To deploy this stack on **AWS**, launch an **Ubuntu 24.04** instance and provide the following **User Data script** in the Advanced Details section:

```bash
#!/bin/bash
# Install Docker and Compose
apt-get update -y
apt-get install -y docker.io docker-compose-v2 git
systemctl start docker
usermod -aG docker ubuntu

# Setup Project
cd /home/ubuntu
git clone <YOUR_REPO_URL>
cd <YOUR_PROJECT_FOLDER>
docker compose up -d --build

```

---

## 📂 Project Structure

```text
.
├── app/
│   └── python-exporter/
│       ├── main.py            # IoT Application Logic & Metrics
│       └── Dockerfile         # Python Container Config
├── prometheus/
│   ├── prometheus.yml         # Scrape Configurations
│   └── alert_rules.yml        # Alerting Logic
├── alertmanager/
│   └── config.yml             # Slack Notification Config
├── grafana/
│   ├── provisioning/          # Datasource & Dashboard Setup
│   └── dashboards/            # Dashboard JSON Model
└── docker-compose.yml         # Container Orchestration

```

---
