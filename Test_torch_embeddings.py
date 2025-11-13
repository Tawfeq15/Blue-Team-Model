# -*- coding: utf-8 -*-
"""
Interactive Model Tester
اختبر النموذج بشكل تفاعلي على URLs حقيقية
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_model_and_pipeline():
    """تحميل النموذج والـpipeline"""
    print("=" * 70)
    print("🔄 Loading model and pipeline...")
    print("=" * 70)
    
    try:
        model = joblib.load("best_model.pkl")
        print("✅ Model loaded: best_model.pkl")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\nTip: Make sure best_model.pkl is in the current directory")
        return None, None
    
    # محاولة تحميل pipeline (إذا موجود)
    pipeline = None
    pipeline_paths = [
        "feature_pipeline.pkl",
        "pipeline.pkl",
        "PhishingData/artifacts/pipeline.pkl"
    ]
    
    for path in pipeline_paths:
        if Path(path).exists():
            try:
                pipeline = joblib.load(path)
                print(f"✅ Pipeline loaded: {path}")
                break
            except:
                continue
    
    if pipeline is None:
        print("⚠️ No pipeline found, will use raw features")
    
    print()
    return model, pipeline

def extract_simple_features(df):
    """استخراج features بسيطة للاختبار"""
    features = []
    
    for idx, row in df.iterrows():
        url = str(row.get('url', ''))
        subject = str(row.get('email_subject', ''))
        body = str(row.get('email_body', ''))
        
        # URL features
        url_len = len(url)
        num_dots = url.count('.')
        num_slashes = url.count('/')
        num_digits = sum(c.isdigit() for c in url)
        has_ip = any(c.isdigit() for c in url.split('//')[-1].split('/')[0].split(':')[0])
        has_https = int(url.startswith('https'))
        has_www = int('www.' in url)
        
        # Domain features
        try:
            domain = url.split('//')[-1].split('/')[0]
            domain_len = len(domain)
            domain_dots = domain.count('.')
        except:
            domain_len = 0
            domain_dots = 0
        
        # Email features
        subject_len = len(subject)
        body_len = len(body)
        
        # Suspicious keywords
        suspicious_words = ['verify', 'urgent', 'confirm', 'account', 'suspend', 
                           'security', 'update', 'click', 'login', 'password']
        num_suspicious = sum(1 for word in suspicious_words if word in subject.lower() or word in body.lower())
        
        feat = {
            'url_length': url_len,
            'num_dots': num_dots,
            'num_slashes': num_slashes,
            'num_digits': num_digits,
            'has_ip': int(has_ip),
            'has_https': has_https,
            'has_www': has_www,
            'domain_length': domain_len,
            'domain_dots': domain_dots,
            'subject_length': subject_len,
            'body_length': body_len,
            'num_suspicious_words': num_suspicious
        }
        
        features.append(feat)
    
    return pd.DataFrame(features)

def predict_url(model, pipeline, url, email_subject="", email_body=""):
    """تنبؤ لـURL واحد"""
    # إنشاء DataFrame
    df = pd.DataFrame([{
        'url': url,
        'email_subject': email_subject,
        'email_body': email_body
    }])
    
    try:
        # استخدام pipeline إذا موجود
        if pipeline is not None:
            X = pipeline.transform(df)
        else:
            X = extract_simple_features(df)
        
        # Prediction
        proba = model.predict_proba(X)[0, 1]
        is_phishing = proba > 0.5
        
        return {
            'probability': proba,
            'is_phishing': is_phishing,
            'confidence': 'high' if proba > 0.9 or proba < 0.1 else 'medium'
        }
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

def print_result(url, result):
    """طباعة النتيجة بشكل جميل"""
    print("\n" + "=" * 70)
    print("📊 RESULT")
    print("=" * 70)
    print(f"URL: {url[:60]}...")
    print()
    
    proba = result['probability']
    is_phishing = result['is_phishing']
    
    # Risk level
    if proba > 0.9:
        risk = "🔴 CRITICAL"
        emoji = "🚨"
    elif proba > 0.7:
        risk = "🟠 HIGH"
        emoji = "⚠️"
    elif proba > 0.5:
        risk = "🟡 MEDIUM"
        emoji = "⚡"
    elif proba > 0.3:
        risk = "🟢 LOW"
        emoji = "✓"
    else:
        risk = "🔵 SAFE"
        emoji = "✅"
    
    print(f"Classification: {emoji} {'PHISHING' if is_phishing else 'LEGITIMATE'}")
    print(f"Probability:    {proba:.4f} ({proba*100:.2f}%)")
    print(f"Risk Level:     {risk}")
    print(f"Confidence:     {result['confidence'].upper()}")
    print("=" * 70)
    
    # Interpretation
    if is_phishing:
        print("\n⚠️ RECOMMENDATION: This URL appears to be PHISHING!")
        print("   - Do not click on this link")
        print("   - Do not enter any credentials")
        print("   - Report this to security team")
    else:
        print("\n✅ RECOMMENDATION: This URL appears to be legitimate")
        if proba > 0.3:
            print("   ⚠️ However, probability is not very low")
            print("   - Still exercise caution")
            print("   - Verify the domain")

def test_samples():
    """اختبار على أمثلة محددة مسبقاً"""
    samples = [
        {
            "url": "http://login-secure-check.com/verify",
            "subject": "Urgent: Verify your account",
            "description": "Suspicious login page"
        },
        {
            "url": "https://www.google.com",
            "subject": "",
            "description": "Legitimate website"
        },
        {
            "url": "http://apple-support-verify.tk/login",
            "subject": "Your Apple ID has been suspended",
            "description": "Fake Apple phishing"
        },
        {
            "url": "https://www.microsoft.com",
            "subject": "",
            "description": "Legitimate Microsoft"
        },
        {
            "url": "http://192.168.1.1/admin",
            "subject": "",
            "description": "IP-based URL"
        }
    ]
    
    print("\n" + "=" * 70)
    print("🧪 Testing on predefined samples...")
    print("=" * 70)
    
    model, pipeline = load_model_and_pipeline()
    if model is None:
        return
    
    for i, sample in enumerate(samples, 1):
        print(f"\n{'─' * 70}")
        print(f"Sample {i}/5: {sample['description']}")
        print(f"{'─' * 70}")
        
        result = predict_url(
            model, pipeline,
            sample['url'],
            sample['subject']
        )
        
        if result:
            print_result(sample['url'], result)
        
        if i < len(samples):
            input("\nPress Enter to continue...")

def interactive_mode():
    """الوضع التفاعلي"""
    print("\n" + "=" * 70)
    print("🎮 Interactive Mode")
    print("=" * 70)
    print("Enter URLs to test (or 'quit' to exit)")
    print()
    
    model, pipeline = load_model_and_pipeline()
    if model is None:
        return
    
    while True:
        print("\n" + "─" * 70)
        url = input("Enter URL: ").strip()
        
        if url.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not url:
            continue
        
        # اختياري: email context
        add_context = input("Add email context? (y/n): ").strip().lower()
        
        if add_context == 'y':
            subject = input("Email subject: ").strip()
            body = input("Email body: ").strip()
        else:
            subject = ""
            body = ""
        
        # Predict
        result = predict_url(model, pipeline, url, subject, body)
        
        if result:
            print_result(url, result)

def batch_test_file(filepath):
    """اختبار batch من ملف CSV"""
    print("\n" + "=" * 70)
    print(f"📁 Batch testing from file: {filepath}")
    print("=" * 70)
    
    model, pipeline = load_model_and_pipeline()
    if model is None:
        return
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df)} samples")
        
        if 'url' not in df.columns:
            print("❌ File must have 'url' column")
            return
        
        # Predictions
        results = []
        for idx, row in df.iterrows():
            url = row['url']
            subject = row.get('email_subject', '')
            body = row.get('email_body', '')
            
            result = predict_url(model, pipeline, url, subject, body)
            if result:
                results.append({
                    'url': url,
                    'probability': result['probability'],
                    'is_phishing': result['is_phishing'],
                    'confidence': result['confidence']
                })
            
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx+1}/{len(df)}...")
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        
        results_df = pd.DataFrame(results)
        num_phishing = results_df['is_phishing'].sum()
        num_legitimate = len(results_df) - num_phishing
        avg_proba = results_df['probability'].mean()
        
        print(f"Total samples:    {len(results_df)}")
        print(f"Phishing:         {num_phishing} ({num_phishing/len(results_df)*100:.1f}%)")
        print(f"Legitimate:       {num_legitimate} ({num_legitimate/len(results_df)*100:.1f}%)")
        print(f"Avg probability:  {avg_proba:.4f}")
        
        # حفظ النتائج
        output_file = filepath.replace('.csv', '_predictions.csv')
        results_df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                   🔍 Phishing Detection Model Tester               ║
║                         Interactive Testing Tool                   ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    print("Choose testing mode:")
    print("  1. Test predefined samples")
    print("  2. Interactive mode (enter URLs manually)")
    print("  3. Batch test from CSV file")
    print("  4. Exit")
    
    while True:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            test_samples()
        elif choice == '2':
            interactive_mode()
        elif choice == '3':
            filepath = input("Enter CSV file path: ").strip()
            if Path(filepath).exists():
                batch_test_file(filepath)
            else:
                print(f"❌ File not found: {filepath}")
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)