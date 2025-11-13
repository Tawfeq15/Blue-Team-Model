# 🔧 حل مشاكل التشغيل - Troubleshooting Guide

## ❌ المشكلة 1: PyTorch DLL Error

```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
Error loading "C:\Python312\Lib\site-packages\torch\lib\c10.dll"
```

### ✅ الحل:

#### الخيار 1: تثبيت Visual C++ Redistributable (الأفضل)

1. **حمّل وثبّت Microsoft Visual C++ Redistributable:**
   - اذهب إلى: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - حمل الملف وثبّته
   - أعد تشغيل الـ terminal

#### الخيار 2: إعادة تثبيت PyTorch

```powershell
# احذف PyTorch القديم
pip uninstall -y torch torchvision torchaudio

# ثبّت PyTorch CPU version (أخف وأسرع)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### الخيار 3: استخدام API بدون PyTorch (الأسرع)

استخدم `serve_api_simple.py` الذي لا يحتاج PyTorch (شوف أدناه)

---

## ❌ المشكلة 2: No module named 'lightgbm'

```
ModuleNotFoundError: No module named 'lightgbm'
```

### ✅ الحل:

```powershell
# ثبّت كل المكتبات الناقصة
pip install lightgbm xgboost catboost

# أو ثبّت كل requirements.txt
pip install -r requirements.txt
```

---

## 🚀 الحل الشامل (مضمون 100%)

### الخطوة 1: تثبيت كل المكتبات

```powershell
# افتح PowerShell كـ Administrator
# انتقل لمجلد المشروع
cd "C:\Users\moham\Desktop\Blue Team Model"

# فعّل virtual environment
.\.venv\Scripts\Activate.ps1

# حدّث pip
python -m pip install --upgrade pip

# ثبّت المكتبات الأساسية
pip install lightgbm xgboost catboost
pip install numpy pandas scikit-learn
pip install fastapi uvicorn pydantic
pip install joblib tqdm
pip install shap imbalanced-learn

# ثبّت PyTorch (CPU version - أخف)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### الخطوة 2: إذا استمرت مشكلة PyTorch

```powershell
# حمّل وثبّت Visual C++ Redistributable
# الرابط: https://aka.ms/vs/17/release/vc_redist.x64.exe

# ثم أعد تشغيل terminal
```

### الخطوة 3: جرب API

```powershell
python serve_api.py
```

---

## 🆘 حل بديل: API مبسط (بدون PyTorch)

إذا ما اشتغل معك، استخدم النسخة المبسطة من API:

### إنشاء `serve_api_simple.py`:

```python
# serve_api_simple.py
# نسخة مبسطة من API بدون PyTorch/SHAP
import os
import json
import joblib
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Phishing Detection API - Simple")

# Paths
ROOT = Path(__file__).parent.resolve()
ART = ROOT / "PhishingData" / "artifacts"
MODEL_F = ART / "best_model.pkl"
CLEANER_F = ART / "data_cleaner.pkl"
RESULTS_JSON = ART / "results.json"

# Load model
model = None
preprocessor = None
config = {}

@app.on_event("startup")
async def load_model():
    global model, preprocessor, config
    try:
        model = joblib.load(MODEL_F)
        preprocessor = joblib.load(CLEANER_F)
        with open(RESULTS_JSON, "r") as f:
            config = json.load(f)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

class PredictionRequest(BaseModel):
    url: str
    subject: str = ""
    body: str = ""

@app.get("/")
def root():
    return {"message": "Phishing Detection API", "status": "running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "threshold": config.get("best_threshold", 0.5)
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        return {"error": "Model not loaded"}

    import pandas as pd
    import numpy as np

    # Prepare data
    df = pd.DataFrame([{
        "url": request.url,
        "subject": request.subject,
        "body": request.body
    }])

    # Preprocess
    processed = preprocessor.clean_data(df, is_train=False)
    X = processed.select_dtypes(include=[np.number, bool])
    if "label" in X.columns:
        X = X.drop(columns=["label"])

    # Predict
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0, 1]
    else:
        prob = model.predict(X)[0]

    threshold = float(config.get("best_threshold", 0.5))
    prediction = 1 if prob >= threshold else 0

    return {
        "url": request.url,
        "prediction": int(prediction),
        "probability": float(prob),
        "is_phishing": bool(prediction == 1),
        "confidence": "high" if abs(prob - 0.5) > 0.3 else "medium",
        "threshold": threshold
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### استخدام API المبسط:

```powershell
python serve_api_simple.py
```

---

## 📋 Checklist - تأكد من:

- [ ] Python 3.11 أو 3.12 مثبت
- [ ] Virtual environment مفعّل (`.venv\Scripts\Activate.ps1`)
- [ ] `pip install lightgbm xgboost catboost` نجح
- [ ] Visual C++ Redistributable مثبت
- [ ] الموديل موجود في `PhishingData/artifacts/best_model.pkl`

---

## 🎯 الخلاصة

### إذا عندك وقت:
```powershell
# ثبّت Visual C++ Redistributable
# ثم ثبّت كل المكتبات
pip install -r requirements.txt
python serve_api.py
```

### إذا تبي حل سريع:
```powershell
# ثبّت المكتبات الأساسية فقط
pip install lightgbm xgboost catboost fastapi uvicorn
python serve_api_simple.py
```

---

## 💡 نصائح إضافية

1. **لو ما اشتغل PyTorch:** مش ضروري للـ API! استخدم النسخة المبسطة
2. **لو Model مش موجود:** لازم تدرب الموديل أولاً بـ `python App.py`
3. **لو في مشاكل بالـ DLL:** ثبّت Visual C++ Redistributable 2015-2022

---

**بالتوفيق! 🚀**
