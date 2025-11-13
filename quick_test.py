#!/usr/bin/env python3
"""
Quick Model Testing Script
==========================
Test your trained phishing detection model quickly!
"""

import sys
from pathlib import Path
import pandas as pd
import joblib
import json

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

def test_model():
    """Test the trained model"""

    artifacts_dir = Path("PhishingData/artifacts")

    # Check if model exists
    model_path = artifacts_dir / "best_model.pkl"
    if not model_path.exists():
        print("❌ Model not found! Train the model first:")
        print("   python App.py")
        return

    print("🔍 Loading model and artifacts...")

    # Load model and preprocessor
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(artifacts_dir / "data_cleaner.pkl")

        # Load results
        with open(artifacts_dir / "results.json", "r") as f:
            results = json.load(f)

        print("✅ Model loaded successfully!")
        print(f"\n📊 Model: {results.get('best_model_name', 'Unknown')}")
        print(f"🎯 Threshold: {results.get('best_threshold', 0.5):.3f}")

        # Display test metrics
        test_metrics = results.get("test_metrics", {})
        print("\n" + "="*50)
        print("TEST SET PERFORMANCE")
        print("="*50)
        print(f"Accuracy : {test_metrics.get('Accuracy', 0):.4f}")
        print(f"Precision: {test_metrics.get('Precision', 0):.4f}")
        print(f"Recall   : {test_metrics.get('Recall', 0):.4f}")
        print(f"F1 Score : {test_metrics.get('F1', 0):.4f}")
        print(f"ROC-AUC  : {test_metrics.get('ROC-AUC', 0):.4f}")
        print(f"PR-AUC   : {test_metrics.get('PR-AUC', 0):.4f}")

        # Confusion matrix
        print("\nConfusion Matrix:")
        print(f"TN: {test_metrics.get('TN', 0):<6} FP: {test_metrics.get('FP', 0)}")
        print(f"FN: {test_metrics.get('FN', 0):<6} TP: {test_metrics.get('TP', 0)}")

        # Evaluate quality
        print("\n" + "="*50)
        print("MODEL QUALITY ASSESSMENT")
        print("="*50)

        recall = test_metrics.get('Recall', 0)
        precision = test_metrics.get('Precision', 0)
        roc_auc = test_metrics.get('ROC-AUC', 0)

        quality_score = 0

        # Recall check (most important)
        if recall >= 0.95:
            print("✅ Recall: EXCELLENT (≥95%)")
            quality_score += 3
        elif recall >= 0.90:
            print("✅ Recall: GOOD (≥90%)")
            quality_score += 2
        elif recall >= 0.85:
            print("⚠️  Recall: ACCEPTABLE (≥85%)")
            quality_score += 1
        else:
            print("❌ Recall: POOR (<85%) - Need retraining!")

        # Precision check
        if precision >= 0.90:
            print("✅ Precision: EXCELLENT (≥90%)")
            quality_score += 2
        elif precision >= 0.85:
            print("✅ Precision: GOOD (≥85%)")
            quality_score += 1
        else:
            print("⚠️  Precision: NEEDS IMPROVEMENT (<85%)")

        # ROC-AUC check
        if roc_auc >= 0.95:
            print("✅ ROC-AUC: EXCELLENT (≥0.95)")
            quality_score += 2
        elif roc_auc >= 0.90:
            print("✅ ROC-AUC: GOOD (≥0.90)")
            quality_score += 1
        else:
            print("⚠️  ROC-AUC: NEEDS IMPROVEMENT (<0.90)")

        # Overall assessment
        print("\n" + "="*50)
        if quality_score >= 6:
            print("🏆 OVERALL: PRODUCTION READY!")
            print("   Your model is excellent and ready for deployment.")
        elif quality_score >= 4:
            print("✅ OVERALL: GOOD MODEL")
            print("   Your model performs well. Consider fine-tuning.")
        elif quality_score >= 2:
            print("⚠️  OVERALL: NEEDS IMPROVEMENT")
            print("   Consider retraining with more data or tuning.")
        else:
            print("❌ OVERALL: POOR PERFORMANCE")
            print("   Retrain the model with better hyperparameters!")

        print("="*50)

        # Test on sample URLs
        print("\n🧪 Testing on sample URLs...")
        test_urls = [
            {"url": "http://login-secure-check.com/verify", "subject": "Account verification", "body": "Click to verify", "expected": "PHISHING"},
            {"url": "https://www.google.com", "subject": "Welcome", "body": "Hello", "expected": "SAFE"},
            {"url": "http://apple-support.tk/login", "subject": "Security alert", "body": "Verify now", "expected": "PHISHING"},
            {"url": "https://www.microsoft.com", "subject": "Newsletter", "body": "Updates", "expected": "SAFE"},
        ]

        from App import PhishingDetectionAPI
        api = PhishingDetectionAPI.from_artifacts(str(artifacts_dir))

        print("\n" + "-"*80)
        for i, test in enumerate(test_urls, 1):
            df_test = pd.DataFrame([{
                "url": test["url"],
                "subject": test.get("subject", ""),
                "body": test.get("body", "")
            }])

            result = api.predict(df_test)
            pred = result["predictions"][0]
            prob = result["probabilities"][0]

            pred_label = "🚨 PHISHING" if pred == 1 else "✅ SAFE" if pred == 0 else "❓ UNCERTAIN"
            expected = test["expected"]
            match = "✓" if (pred == 1 and expected == "PHISHING") or (pred == 0 and expected == "SAFE") else "✗"

            print(f"{i}. {test['url'][:50]}")
            print(f"   Prediction: {pred_label} (prob: {prob:.3f}) [{match}]")
            print(f"   Expected: {expected}")
            print()

        print("-"*80)
        print("\n✅ Testing complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
