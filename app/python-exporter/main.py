import threading
import time
import statistics
import requests
import joblib
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ---------------- Config & Setup ----------------
DATALAKE_URL = "http://149.40.228.124:6500/records"
SLACK_WEBHOOK = "YOUR_SLACK_WEBHOOK_URL_HERE"
RETRAIN_THRESHOLD = 0.8

# ---------------- Prometheus Metrics (Rubric Point 4) ----------------
model_accuracy = Gauge('model_accuracy', 'Current accuracy of the ML model')
records_processed_total = Counter('records_processed_total', 'Total records processed')
retrain_count_total = Counter('retrain_count_total', 'Total number of retrains performed')
distribution_drift_detected = Gauge('distribution_drift_detected', 'Distribution drift detected')
feature_added = Gauge('feature_added', 'New feature detected')
feature_removed = Gauge('feature_removed', 'Feature removed')
datalake_unavailable = Counter('datalake_unavailable', 'Data Lake 503 count')
response_delay_seconds = Gauge('response_delay_seconds', 'Response delay from Data Lake')

# ---------------- Global State ----------------
# Load initial model (ensure this file exists in your Docker image)
model = joblib.load("model_cycle_20.joblib")

X_history, y_history = [], []
previous_features_count = 0

# ---------------- Helper Functions ----------------
def send_slack_alert(message):
    # 1. Skip if webhook is not configured (prevents constant error logs)
    if not SLACK_WEBHOOK or "XXXX" in SLACK_WEBHOOK:
        print(f"DEBUG ALERT: {message}")
        return

    payload = {"text": f"🚨 *MLOps Alert*: {message}"}
    
    try:
        # 2. Add a timeout (2 seconds) so a slow Slack API doesn't kill your app
        response = requests.post(
            SLACK_WEBHOOK, 
            json=payload, 
            timeout=2 
        )
        
        # 3. Check if Slack actually accepted it (Status 200)
        if response.status_code != 200:
            print(f"Slack Error {response.status_code}: {response.text}")
            
    except requests.exceptions.Timeout:
        print("Slack alert failed: Request timed out.")
    except Exception as e:
        print(f"Slack alert failed: {e}")

def ingestion_and_retrain_loop():
    global model, X_history, y_history, previous_features_count
    
    while True:
        try:
            start_time = time.time()
            resp = requests.get(DATALAKE_URL)
            response_delay_seconds.set(time.time() - start_time)

            if resp.status_code == 503:
                datalake_unavailable.inc()
                send_slack_alert("Data Lake 503 - Service Unavailable")
                time.sleep(30)
                continue

            records = resp.json()
            if not records:
                continue

            # 1. Process Records
            records_processed_total.inc(len(records))
            for r in records:
                X_history.append(r['features'])
                y_history.append(r['label'])

            # 2. Detect Feature Changes (Rubric requirement)
            current_feat_count = len(records[0]['features'])
            if previous_features_count != 0:
                if current_feat_count > previous_features_count:
                    feature_added.set(1)
                    send_slack_alert("Feature Added detected")
                elif current_feat_count < previous_features_count:
                    feature_removed.set(1)
                    send_slack_alert("Feature Removed detected")
            previous_features_count = current_feat_count

            # 3. Retraining Logic (Rubric Point 3 & 6.7)
            if len(X_history) > 20:
                X_train, X_test, y_train, y_test = train_test_split(X_history, y_history, test_size=0.2)
                model.fit(X_train, y_train)
                acc = accuracy_score(y_test, model.predict(X_test))
                model_accuracy.set(acc)

                if acc < RETRAIN_THRESHOLD:
                    retrain_count_total.inc()
                    send_slack_alert(f"Accuracy {acc:.2f} < 0.8. Retraining triggered.")
                    # In a real scenario, you'd re-fit with more data or params
                    model.fit(X_history, y_history) 

        except Exception as e:
            print(f"Loop Error: {e}")
        
        time.sleep(20) # Poll every 20 seconds

# ---------------- FastAPI App ----------------
app = FastAPI()

@app.on_event("startup")
def startup_event():
    # Start the background thread
    thread = threading.Thread(target=ingestion_and_retrain_loop, daemon=True)
    thread.start()

@app.get("/predict")
def predict(features: str = Query(..., example="680.2,679.3")):
    try:
        feat_list = [float(x) for x in features.split(",")]
        prediction = model.predict([feat_list])[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/metrics")
def metrics():
    # We use REGISTRY to ensure we are pulling from the default global registry
    return PlainTextResponse(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)