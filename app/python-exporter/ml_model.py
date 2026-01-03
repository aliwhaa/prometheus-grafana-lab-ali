import requests
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from prometheus_client import Counter, Gauge, start_http_server
import joblib
import threading
import statistics
import os
import smtplib
from email.mime.text import MIMEText  # You can swap for Slack webhook later

# ---------------- Prometheus Metrics ----------------
records_processed_total = Counter('records_processed_total', 'Total number of records processed')
datalake_unavailable = Counter('datalake_unavailable', 'Number of times Data Lake returned 503')
response_delay_seconds = Gauge('response_delay_seconds', 'Response time from Data Lake in seconds')
model_accuracy = Gauge('model_accuracy', 'Current accuracy of the ML model')
retrain_count_total = Gauge('retrain_count_total', 'Total number of retrains performed')
distribution_drift_detected = Gauge('distribution_drift_detected', 'Distribution drift detected')
feature_added = Gauge('feature_added', 'New feature detected')
feature_removed = Gauge('feature_removed', 'Feature removed')

# ---------------- Config ----------------
URL = "http://149.40.228.124:6500/records"
RETRAIN_THRESHOLD = 0.8
MIN_SAMPLES_TO_TRAIN = 4
TRAIN_CYCLES = 50
SAVE_INTERVAL = 5
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/XXXX/XXXX/XXXX"  # Optional

# ---------------- ML Model ----------------
model = RandomForestClassifier()
X, y = [], []
retrain_count = 0
current_cycle = 0
previous_features = set()
feature_stats = {}  # track mean/std for drift detection

# ---------------- Functions ----------------
def send_alert(msg):
    # Placeholder: implement Slack webhook call or email
    print(f"ALERT: {msg}")
    # Example: requests.post(SLACK_WEBHOOK_URL, json={"text": msg})

def fetch_records():
    try:
        start_time = time.time()
        response = requests.get(URL)
        delay = time.time() - start_time
        response_delay_seconds.set(delay)

        if response.status_code == 503:
            datalake_unavailable.inc()
            send_alert("Data Lake unavailable! 503 detected")
            print("Data Lake unavailable! 503 detected.")
            return None

        data = response.json()
        if isinstance(data, list):
            records = data
        else:
            records = data.get('records', [])

        records_processed_total.inc(len(records))
        return records

    except Exception as e:
        datalake_unavailable.inc()
        send_alert(f"Error fetching data: {e}")
        print("Error fetching data:", e)
        return None

def detect_feature_changes(records):
    global previous_features
    new_features_set = set(records[0].get('features', []))
    
    added = new_features_set - previous_features
    removed = previous_features - new_features_set

    if added:
        feature_added.inc(len(added))
        send_alert(f"New feature(s) added: {added}")
    if removed:
        feature_removed.inc(len(removed))
        send_alert(f"Feature(s) removed: {removed}")

    previous_features = new_features_set

def detect_drift(records):
    # Simple distribution drift: check mean changes per feature
    global feature_stats
    drift_detected = False
    for idx, _ in enumerate(records[0].get('features', [])):
        vals = [r['features'][idx] for r in records]
        mean_val = statistics.mean(vals)
        std_val = statistics.stdev(vals) if len(vals) > 1 else 0.0

        if idx in feature_stats:
            old_mean, old_std = feature_stats[idx]
            if abs(mean_val - old_mean) > max(0.1, 0.2 * old_std):  # basic threshold
                drift_detected = True
        feature_stats[idx] = (mean_val, std_val)

    if drift_detected:
        distribution_drift_detected.set(1)
        send_alert("Distribution drift detected")
    else:
        distribution_drift_detected.set(0)

def train_model():
    global model, retrain_count

    if len(X) < MIN_SAMPLES_TO_TRAIN:
        return 0.0

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X), np.array(y), test_size=0.3, random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    model_accuracy.set(acc)
    retrain_count_total.set(retrain_count)

    if acc < RETRAIN_THRESHOLD:
        retrain_count += 1
        send_alert(f"Accuracy dropped below {RETRAIN_THRESHOLD:.2f}: retraining needed")

    return acc

# ---------------- Main Loop ----------------
def ingestion_loop():
    global current_cycle, X, y
    start_http_server(8001)  # Prometheus metrics

    while current_cycle < TRAIN_CYCLES:
        records = fetch_records()
        if records:
            detect_feature_changes(records)
            detect_drift(records)

            for item in records:
                features = item.get('features')
                label = item.get('label')
                if features is not None and label is not None:
                    X.append(features)
                    y.append(label)

            acc = train_model()
            print(f"Cycle {current_cycle + 1}/{TRAIN_CYCLES}: Accuracy={acc:.2f}, Records={len(records)}")

            if (current_cycle + 1) % SAVE_INTERVAL == 0:
                model_file = f"model_cycle_{current_cycle + 1}.joblib"
                joblib.dump(model, model_file)
                print(f"Model saved: {model_file}")

            current_cycle += 1

        time.sleep(10)

    # Final model save
    joblib.dump(model, "ml_model_final.joblib")
    print("Training complete. Final model saved.")

# ---------------- Run ----------------
if __name__ == "__main__":
    ingestion_loop()
