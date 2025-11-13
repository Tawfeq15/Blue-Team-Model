#!/usr/bin/env python3
"""
🛡️ Simplified Phishing Detection API
====================================
Lightweight API without PyTorch/SHAP dependencies
Perfect for quick deployment and testing!
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Core dependencies only
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# ================== Configuration ==================
API_KEY = os.getenv("API_KEY", "dev-key")
ROOT = Path(__file__).parent.resolve()
ART = ROOT / "PhishingData" / "artifacts"
MODEL_F = ART / "best_model.pkl"
CLEANER_F = ART / "data_cleaner.pkl"
RESULTS_JSON = ART / "results.json"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
log = logging.getLogger("phishing-api-simple")

# Stats tracking
stats = {
    "total_predictions": 0,
    "phishing_detected": 0,
    "safe_emails": 0,
    "start_time": datetime.now(),
}

# ================== FastAPI App ==================
app = FastAPI(
    title="🛡️ Phishing Detection API (Simple)",
    description="Lightweight phishing detection without heavy dependencies",
    version="1.0.0"
)

# Global variables
model = None
preprocessor = None
config = {}
threshold = 0.5

# ================== Pydantic Models ==================
class PredictionRequest(BaseModel):
    url: str = Field(..., description="URL to analyze")
    subject: str = Field(default="", description="Email subject (optional)")
    body: str = Field(default="", description="Email body (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "url": "http://login-secure-check.com/verify",
                "subject": "Verify your account",
                "body": "Click here to verify your account immediately"
            }
        }

class PredictionResponse(BaseModel):
    url: str
    prediction: int  # 0=safe, 1=phishing
    probability: float
    is_phishing: bool
    confidence: str  # low, medium, high
    threshold: float

class BatchRequest(BaseModel):
    items: List[PredictionRequest]

class StatsResponse(BaseModel):
    total_predictions: int
    phishing_detected: int
    safe_emails: int
    uptime_seconds: float
    phishing_rate: float

# ================== Helper Functions ==================
def load_artifacts():
    """Load model, preprocessor, and config"""
    global model, preprocessor, config, threshold

    log.info(f"Loading artifacts from: {ART}")

    try:
        # Load model
        if not MODEL_F.exists():
            log.error(f"Model file not found: {MODEL_F}")
            log.warning("⚠️  Train the model first: python App.py")
            return False

        log.info("Loading model...")
        model = joblib.load(MODEL_F)
        log.info(f"✓ Model loaded: {type(model).__name__}")

        # Load preprocessor
        log.info("Loading preprocessor...")
        preprocessor = joblib.load(CLEANER_F)
        log.info("✓ Preprocessor loaded")

        # Load config
        if RESULTS_JSON.exists():
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                config = json.load(f)
            threshold = float(config.get("best_threshold", 0.5))
            log.info(f"✓ Config loaded (threshold: {threshold:.3f})")
        else:
            log.warning("⚠️  results.json not found, using default threshold")
            threshold = 0.5

        log.info("✅ All artifacts loaded successfully!")
        return True

    except Exception as e:
        log.error(f"❌ Failed to load artifacts: {e}", exc_info=True)
        return False

def predict_sample(url: str, subject: str = "", body: str = "") -> Dict[str, Any]:
    """Make prediction on a single sample"""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Create dataframe
        df = pd.DataFrame([{
            "url": url,
            "subject": subject if subject else "",
            "body": body if body else ""
        }])

        # Preprocess
        processed = preprocessor.clean_data(df.copy(), is_train=False)

        # Extract numeric features
        X = processed.select_dtypes(include=[np.number, bool])
        if "label" in X.columns:
            X = X.drop(columns=["label"])

        # Predict probability
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0, 1])
        elif hasattr(model, "decision_function"):
            z = float(model.decision_function(X)[0])
            prob = 1.0 / (1.0 + np.exp(-z))  # sigmoid
        else:
            prob = float(model.predict(X)[0])

        # Make prediction
        prediction = 1 if prob >= threshold else 0

        # Determine confidence
        dist_from_threshold = abs(prob - threshold)
        if dist_from_threshold > 0.3:
            confidence = "high"
        elif dist_from_threshold > 0.15:
            confidence = "medium"
        else:
            confidence = "low"

        # Update stats
        stats["total_predictions"] += 1
        if prediction == 1:
            stats["phishing_detected"] += 1
        else:
            stats["safe_emails"] += 1

        return {
            "url": url,
            "prediction": prediction,
            "probability": prob,
            "is_phishing": bool(prediction == 1),
            "confidence": confidence,
            "threshold": threshold
        }

    except Exception as e:
        log.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ================== API Endpoints ==================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_artifacts()
    if not success:
        log.warning("⚠️  API started but model not loaded!")
        log.warning("   Train the model first: python App.py")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Web interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🛡️ Phishing Detection API</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            h1 { margin-top: 0; }
            .button {
                display: inline-block;
                padding: 12px 24px;
                margin: 10px 5px;
                background: white;
                color: #667eea;
                text-decoration: none;
                border-radius: 5px;
                font-weight: bold;
                transition: transform 0.2s;
            }
            .button:hover {
                transform: translateY(-2px);
            }
            .feature {
                background: rgba(255,255,255,0.1);
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
            }
            code {
                background: rgba(0,0,0,0.3);
                padding: 2px 6px;
                border-radius: 3px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛡️ Phishing Detection API</h1>
            <p><strong>Simplified Version</strong> - No PyTorch/SHAP dependencies</p>

            <div class="feature">
                <h3>✅ Status</h3>
                <p>API is running and ready!</p>
            </div>

            <div class="feature">
                <h3>📚 Documentation</h3>
                <a href="/docs" class="button">Interactive API Docs</a>
                <a href="/health" class="button">Health Check</a>
                <a href="/stats" class="button">Statistics</a>
            </div>

            <div class="feature">
                <h3>🧪 Quick Test</h3>
                <p>Test the API with curl:</p>
                <pre><code>curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "url": "http://login-secure-check.com/verify",
    "subject": "Verify your account",
    "body": "Click here now"
  }'</code></pre>
            </div>

            <div class="feature">
                <h3>📖 Endpoints</h3>
                <ul>
                    <li><code>POST /predict</code> - Single prediction</li>
                    <li><code>POST /predict/batch</code> - Batch predictions</li>
                    <li><code>GET /health</code> - Health check</li>
                    <li><code>GET /stats</code> - Usage statistics</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "threshold": threshold,
        "model_type": type(model).__name__ if model else None,
        "artifacts_dir": str(ART),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats", response_model=StatsResponse)
def get_stats():
    """Get usage statistics"""
    uptime = (datetime.now() - stats["start_time"]).total_seconds()
    total = stats["total_predictions"]
    phishing_rate = stats["phishing_detected"] / total if total > 0 else 0.0

    return {
        "total_predictions": total,
        "phishing_detected": stats["phishing_detected"],
        "safe_emails": stats["safe_emails"],
        "uptime_seconds": uptime,
        "phishing_rate": phishing_rate
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_single(request: PredictionRequest):
    """
    Predict if a URL/email is phishing

    - **url**: URL to analyze (required)
    - **subject**: Email subject (optional)
    - **body**: Email body (optional)
    """
    return predict_sample(request.url, request.subject, request.body)

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    """
    Batch prediction for multiple URLs/emails
    """
    results = []
    for item in request.items:
        try:
            result = predict_sample(item.url, item.subject, item.body)
            results.append(result)
        except Exception as e:
            results.append({
                "url": item.url,
                "error": str(e)
            })

    return {
        "total": len(request.items),
        "results": results
    }

@app.get("/config")
def get_config():
    """Get model configuration"""
    if not config:
        raise HTTPException(status_code=503, detail="Config not loaded")

    return {
        "best_model": config.get("best_model_name", "Unknown"),
        "threshold": threshold,
        "test_metrics": config.get("test_metrics", {}),
        "training_time": config.get("training_time_seconds", 0),
        "timestamp": config.get("timestamp", "Unknown")
    }

# ================== Run Server ==================
if __name__ == "__main__":
    log.info("="*60)
    log.info("🛡️  Starting Phishing Detection API (Simplified)")
    log.info("="*60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
