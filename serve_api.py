# serve_api_ENHANCED.py
"""
🛡️ Enhanced Phishing Detection API with Web UI
===============================================
Features:
- ✅ Simple & beautiful web interface
- ✅ Batch prediction support
- ✅ Real-time statistics
- ✅ Explanation endpoint
- ✅ Health monitoring
"""
import requests
import os
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import time

# ================== Configuration ==================
API_KEY = os.getenv("API_KEY", "dev-key")
ROOT = Path(__file__).parent.resolve()
ART = ROOT / "PhishingData" / "artifacts"
MODEL_F = ART / "best_model.pkl"
PIPELINE_F = ART / "feature_pipeline.pkl"
CLEANER_F = ART / "data_cleaner.pkl"
DRIFT_F = ART / "drift_monitor.pkl"
RESULTS_JSON = ART / "results.json"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
log = logging.getLogger("phishing-api")

# Stats tracking
stats = {
    "total_predictions": 0,
    "phishing_detected": 0,
    "safe_emails": 0,
    "start_time": datetime.now(),
    "avg_response_time_ms": 0,
    "total_response_time": 0
}


# ================== Unpickling Shim ==================
def _shim_classes_for_unpickling():
    """Fix unpickling issues for custom classes"""
    try:
        from App import DataPreprocessor, DriftMonitor, ModelConfig
        
        try:
            from enhanced_trainer import EnhancedPhishingTrainer, EnhancedPhishingTrainerV2
        except ImportError:
            EnhancedPhishingTrainer = None
            EnhancedPhishingTrainerV2 = None
        
        main_module = sys.modules["__main__"]
        main_module.DataPreprocessor = DataPreprocessor
        main_module.DriftMonitor = DriftMonitor
        main_module.ModelConfig = ModelConfig
        if EnhancedPhishingTrainer:
            main_module.EnhancedPhishingTrainer = EnhancedPhishingTrainer
        if EnhancedPhishingTrainerV2:
            main_module.EnhancedPhishingTrainerV2 = EnhancedPhishingTrainerV2
        
        try:
            from sklearn.pipeline import Pipeline, FeatureUnion
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import StandardScaler
            
            main_module.Pipeline = Pipeline
            main_module.FeatureUnion = FeatureUnion
            main_module.TfidfVectorizer = TfidfVectorizer
            main_module.StandardScaler = StandardScaler
        except ImportError:
            pass
        
        log.info("✓ Shimmed __main__ with all required classes")
        return True
    except Exception as e:
        log.error("Shim setup failed: %s", e, exc_info=True)
        return False


_shim_classes_for_unpickling()


# ================== Helper Functions ==================
def _load_bin(path: Path):
    """Load binary artifact"""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


def _safe_json(path: Path, default: dict):
    """Safely load JSON file"""
    try:
        if path.exists():
            return json.load(open(path, "r", encoding="utf-8"))
    except Exception as e:
        log.warning("Failed to read %s: %s", path, e)
    return default


def _to_df_text(text: str) -> pd.DataFrame:
    """Convert text to DataFrame"""
    return pd.DataFrame([{"text": text or ""}])


def _predict_proba(txt: str) -> float:
    """Get phishing probability for text"""
    try:
        df = _to_df_text(txt)
        
        # Apply cleaner if available
        if cleaner is not None:
            try:
                if hasattr(cleaner, "transform"):
                    df = cleaner.transform(df)
                elif callable(cleaner):
                    df = cleaner(df)
            except Exception as e:
                log.warning("Cleaner failed: %s", e)
        
        # Transform and predict
        X = feature_pipeline.transform(df)
        proba = model.predict_proba(X)[:, 1][0]
        return float(proba)
    except Exception as e:
        log.error("Prediction failed: %s", e, exc_info=True)
        raise


def _get_vectorizer_from_pipeline(pipeline):
    """Extract vectorizer from pipeline"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        if hasattr(pipeline, "steps"):
            for _, step in pipeline.steps:
                if isinstance(step, TfidfVectorizer):
                    return step
        if "TfidfVectorizer" in type(pipeline).__name__:
            return pipeline
    except Exception:
        pass
    return None


def _explain_text(txt: str, top_k: int = 10) -> Dict[str, Any]:
    """Get explanation for prediction"""
    info = {
        "available": False,
        "method": "feature_importance",
        "top_positive": [],
        "top_negative": [],
    }
    
    try:
        vec = _get_vectorizer_from_pipeline(feature_pipeline)
        if vec is None or not hasattr(model, "coef_"):
            return info
        
        X = vec.transform([txt])
        coef = model.coef_.reshape(-1)
        contrib = X.multiply(coef).toarray().reshape(-1)
        feats = vec.get_feature_names_out()
        
        # Get top contributors
        idx = np.argsort(np.abs(contrib))[::-1][:top_k * 2]
        ranked = [(feats[i], float(contrib[i])) for i in idx if contrib[i] != 0.0]
        
        pos = [(t, v) for (t, v) in ranked if v > 0][:top_k]
        neg = [(t, v) for (t, v) in ranked if v < 0][:top_k]
        
        info.update({
            "available": True,
            "top_positive": pos,
            "top_negative": neg
        })
    except Exception as e:
        log.warning("Explanation failed: %s", e)
    
    return info


def _update_stats(is_phishing: bool, response_time: float):
    """Update statistics"""
    stats["total_predictions"] += 1
    if is_phishing:
        stats["phishing_detected"] += 1
    else:
        stats["safe_emails"] += 1
    
    stats["total_response_time"] += response_time
    stats["avg_response_time_ms"] = (
        stats["total_response_time"] / stats["total_predictions"] * 1000
    )


# ================== Load Artifacts ==================
log.info("Loading artifacts from: %s", ART.as_posix())

try:
    model = _load_bin(MODEL_F)
    log.info("✓ Model loaded: %s", type(model).__name__)
except Exception as e:
    log.error("Failed to load model: %s", e, exc_info=True)
    raise

try:
    feature_pipeline = _load_bin(PIPELINE_F)
    log.info("✓ Pipeline loaded")
except Exception as e:
    log.error("Failed to load pipeline: %s", e, exc_info=True)
    raise

try:
    cleaner = _load_bin(CLEANER_F) if CLEANER_F.exists() else None
    log.info("✓ Cleaner loaded: %s", cleaner is not None)
except Exception as e:
    log.warning("Failed to load cleaner: %s", e)
    cleaner = None

try:
    drift_monitor = _load_bin(DRIFT_F) if DRIFT_F.exists() else None
    log.info("✓ Drift monitor loaded: %s", drift_monitor is not None)
except Exception as e:
    log.warning("Failed to load drift monitor: %s", e)
    drift_monitor = None

results_meta = _safe_json(RESULTS_JSON, default={})
best_threshold = float(results_meta.get("best_threshold", 0.5))
model_metrics = {
    "accuracy": results_meta.get("test_accuracy", 0),
    "precision": results_meta.get("test_precision", 0),
    "recall": results_meta.get("test_recall", 0),
    "f1": results_meta.get("test_f1", 0),
    "roc_auc": results_meta.get("test_roc_auc", 0)
}

log.info("✓ All artifacts loaded. Threshold=%.4f", best_threshold)


# ================== FastAPI App ==================
app = FastAPI(
    title="Phishing Detection API",
    version="2.0",
    description="Enhanced phishing email detection with web UI"
)


# ================== Middleware ==================
@app.middleware("http")
async def auth_and_timing_mw(request: Request, call_next):
    """Authentication and timing middleware"""
    # Public endpoints
    if request.url.path in ("/", "/health", "/healthz", "/stats", "/docs", "/openapi.json"):
        return await call_next(request)
    
    # Check API key for protected endpoints
    if request.headers.get("x-api-key") != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid API key"}
        )
    
    # Process request with timing
    start_time = time.time()
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        log.exception("Unhandled error in request: %s", e)
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )


# ================== Pydantic Models ==================
class TextIn(BaseModel):
    text: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Congratulations! You've won $1,000,000! Click here to claim."
            }
        }


class BatchTextIn(BaseModel):
    texts: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Meeting at 10 AM tomorrow",
                    "URGENT! Verify your account NOW!"
                ]
            }
        }


class PredictionOut(BaseModel):
    text: str
    probability: float
    label: str
    is_phishing: bool
    confidence: str
    threshold: float
    response_time_ms: float


# ================== Web UI ==================
@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Serve web interface"""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛡️ Phishing Email Detector</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 18px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            width: 100%;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 8px;
            display: none;
        }
        
        .result.phishing {
            background: #fee;
            border: 2px solid #f44336;
        }
        
        .result.safe {
            background: #efe;
            border: 2px solid #4caf50;
        }
        
        .result-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .result-icon {
            font-size: 3em;
            margin-right: 15px;
        }
        
        .result-title {
            font-size: 1.8em;
            font-weight: 700;
        }
        
        .result.phishing .result-title {
            color: #c62828;
        }
        
        .result.safe .result-title {
            color: #2e7d32;
        }
        
        .result-details {
            margin-top: 15px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #ddd;
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            font-weight: 600;
            color: #555;
        }
        
        .metric-value {
            color: #333;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .examples {
            margin-top: 20px;
        }
        
        .example-btn {
            display: inline-block;
            padding: 8px 15px;
            margin: 5px;
            background: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }
        
        .example-btn:hover {
            background: #e0e0e0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Phishing Email Detector</h1>
            <p>Powered by Machine Learning - Detect phishing attempts instantly</p>
        </div>
        
        <div class="card">
            <div class="input-group">
                <label for="emailText">Enter Email Text:</label>
                <textarea 
                    id="emailText" 
                    rows="8" 
                    placeholder="Paste the email content here..."
                ></textarea>
            </div>
            
            <button class="btn" onclick="checkEmail()" id="checkBtn">
                <span id="btnText">Check Email</span>
            </button>
            
            <div class="examples">
                <strong>Try examples:</strong><br>
                <span class="example-btn" onclick="setExample(0)">Safe Email</span>
                <span class="example-btn" onclick="setExample(1)">Phishing Attempt</span>
                <span class="example-btn" onclick="setExample(2)">Urgent Scam</span>
            </div>
            
            <div id="result" class="result"></div>
        </div>
        
        <div class="card">
            <h2 style="margin-bottom: 15px;">📊 Statistics</h2>
            <div class="stats-grid" id="statsGrid">
                <div class="stat-card">
                    <div class="stat-value" id="totalPredictions">0</div>
                    <div class="stat-label">Total Checks</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="phishingDetected">0</div>
                    <div class="stat-label">Phishing Detected</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="safeEmails">0</div>
                    <div class="stat-label">Safe Emails</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="avgTime">0</div>
                    <div class="stat-label">Avg Response (ms)</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const examples = [
            "Thank you for the update. I'll review the document and get back to you by tomorrow. Best regards.",
            "CONGRATULATIONS! You've been selected to receive $1,000,000! Click this link immediately to claim your prize: http://suspicious-site.com/claim",
            "URGENT: Your account has been compromised! Verify your identity NOW by clicking here and entering your password or your account will be deleted in 24 hours!"
        ];
        
        function setExample(index) {
            document.getElementById('emailText').value = examples[index];
        }
        
        async function checkEmail() {
            const text = document.getElementById('emailText').value.trim();
            
            if (!text) {
                alert('Please enter some email text!');
                return;
            }
            
            const btn = document.getElementById('checkBtn');
            const btnText = document.getElementById('btnText');
            const result = document.getElementById('result');
            
            // Show loading state
            btn.disabled = true;
            btnText.innerHTML = '<span class="loading"></span> Analyzing...';
            result.style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'x-api-key': 'dev-key'
                    },
                    body: JSON.stringify({ text: text })
                });
                
                if (!response.ok) {
                    throw new Error('Prediction failed');
                }
                
                const data = await response.json();
                displayResult(data);
                
                // Update stats
                updateStats();
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                btn.disabled = false;
                btnText.textContent = 'Check Email';
            }
        }
        
        function displayResult(data) {
            const result = document.getElementById('result');
            const isPhishing = data.is_phishing;
            const prob = (data.probability * 100).toFixed(1);
            
            result.className = 'result ' + (isPhishing ? 'phishing' : 'safe');
            result.innerHTML = `
                <div class="result-header">
                    <div class="result-icon">${isPhishing ? '⚠️' : '✅'}</div>
                    <div>
                        <div class="result-title">${isPhishing ? 'PHISHING DETECTED!' : 'Safe Email'}</div>
                        <div>${data.confidence} confidence</div>
                    </div>
                </div>
                <div class="result-details">
                    <div class="metric">
                        <span class="metric-label">Probability:</span>
                        <span class="metric-value">${prob}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Label:</span>
                        <span class="metric-value">${data.label}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Threshold:</span>
                        <span class="metric-value">${(data.threshold * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Response Time:</span>
                        <span class="metric-value">${data.response_time_ms.toFixed(1)} ms</span>
                    </div>
                </div>
            `;
            result.style.display = 'block';
        }
        
        async function updateStats() {
            try {
                const response = await fetch('/stats');
                const data = await response.json();
                
                document.getElementById('totalPredictions').textContent = data.total_predictions;
                document.getElementById('phishingDetected').textContent = data.phishing_detected;
                document.getElementById('safeEmails').textContent = data.safe_emails;
                document.getElementById('avgTime').textContent = data.avg_response_time_ms.toFixed(1);
            } catch (error) {
                console.error('Failed to update stats:', error);
            }
        }
        
        // Update stats on page load
        updateStats();
        
        // Allow Enter to submit
        document.getElementById('emailText').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                checkEmail();
            }
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html)


# ================== API Endpoints ==================
@app.get("/health")
@app.get("/healthz")
def health() -> Dict[str, Any]:
    """Health check endpoint"""
    uptime = (datetime.now() - stats["start_time"]).total_seconds()
    
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "model": type(model).__name__,
        "threshold": best_threshold,
        "artifacts_loaded": {
            "model": True,
            "pipeline": True,
            "cleaner": cleaner is not None,
            "drift_monitor": drift_monitor is not None
        },
        "metrics": model_metrics
    }


@app.get("/stats")
def get_stats() -> Dict[str, Any]:
    """Get API statistics"""
    uptime = (datetime.now() - stats["start_time"]).total_seconds()
    
    return {
        **stats,
        "uptime_seconds": uptime,
        "phishing_rate": (
            stats["phishing_detected"] / max(1, stats["total_predictions"]) * 100
        )
    }


@app.post("/predict", response_model=PredictionOut)
def predict(
    inp: TextIn,
    x_api_key: Optional[str] = Header(default=None)
) -> Dict[str, Any]:
    """Predict if email is phishing"""
    start_time = time.time()
    
    try:
        proba = _predict_proba(inp.text)
        is_phishing = proba >= best_threshold
        
        # Determine confidence level
        confidence_score = abs(proba - 0.5) * 2  # 0 to 1
        if confidence_score > 0.8:
            confidence = "Very High"
        elif confidence_score > 0.6:
            confidence = "High"
        elif confidence_score > 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        response_time = (time.time() - start_time)
        
        # Update stats
        _update_stats(is_phishing, response_time)
        
        return {
            "text": inp.text[:100] + ("..." if len(inp.text) > 100 else ""),
            "probability": proba,
            "label": "Phishing" if is_phishing else "Safe",
            "is_phishing": is_phishing,
            "confidence": confidence,
            "threshold": best_threshold,
            "response_time_ms": response_time * 1000
        }
    except Exception as e:
        log.exception("Prediction endpoint error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(
    inp: BatchTextIn,
    x_api_key: Optional[str] = Header(default=None)
) -> Dict[str, Any]:
    """Batch prediction endpoint"""
    start_time = time.time()
    
    try:
        results = []
        for text in inp.texts:
            proba = _predict_proba(text)
            is_phishing = proba >= best_threshold
            
            results.append({
                "text": text[:100] + ("..." if len(text) > 100 else ""),
                "probability": proba,
                "label": "Phishing" if is_phishing else "Safe",
                "is_phishing": is_phishing
            })
            
            _update_stats(is_phishing, 0)
        
        response_time = (time.time() - start_time)
        
        return {
            "total": len(results),
            "phishing_count": sum(1 for r in results if r["is_phishing"]),
            "safe_count": sum(1 for r in results if not r["is_phishing"]),
            "results": results,
            "response_time_ms": response_time * 1000
        }
    except Exception as e:
        log.exception("Batch prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
def explain(
    inp: TextIn,
    x_api_key: Optional[str] = Header(default=None)
) -> Dict[str, Any]:
    """Get prediction explanation"""
    try:
        explanation = _explain_text(inp.text, top_k=10)
        proba = _predict_proba(inp.text)
        
        return {
            "text": inp.text[:200],
            "probability": proba,
            "is_phishing": proba >= best_threshold,
            "explanation": explanation
        }
    except Exception as e:
        log.exception("Explanation endpoint error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ================== Run Instructions ==================
if __name__ == "__main__":
    import uvicorn
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Phishing Detection API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    args = parser.parse_args()

    print("=" * 70)
    print("🛡️  Phishing Detection API Server")
    print("=" * 70)
    print(f"Model: {type(model).__name__}")
    print(f"Threshold: {best_threshold:.4f}")
    print(f"Accuracy: {model_metrics.get('accuracy', 0):.2%}")
    print(f"Recall: {model_metrics.get('recall', 0):.2%}")
    print("=" * 70)
    print("\n🌐 Starting server...")
    print(f"   Web UI: http://localhost:{args.port}")
    print(f"   API Docs: http://localhost:{args.port}/docs")
    print(f"   Health: http://localhost:{args.port}/health")
    print("\n📝 API Key: dev-key")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(app, host=args.host, port=args.port)