# -*- coding: utf-8 -*-
"""
Enhanced Split Leakage Auditor
يفحص التسريب بين train/val/test ويعطي تحليل مفصّل
"""
import os, sys, hashlib
import pandas as pd
from collections import Counter

LABEL_COL = os.environ.get("LABEL_COL", "label")

def _sha(x: str) -> str:
    if pd.isna(x): x = ""
    return hashlib.sha1(str(x).encode("utf-8", "ignore")).hexdigest()

def _load_csv(p):
    return pd.read_csv(p, engine="c", low_memory=False)

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "url_canon" not in df.columns:
        if "url" in df.columns:
            df["url_canon"] = df["url"].astype(str).str.strip()
        else:
            df["url_canon"] = ""
    if "domain" not in df.columns:
        df["domain"] = df["url_canon"].str.extract(r"^(?:https?://)?([^/]+)")[0]
    
    subj = df["email_subject"].astype(str) if "email_subject" in df.columns else ""
    body = df["email_body"].astype(str) if "email_body" in df.columns else ""
    df["_text_joined"] = (subj + " " + body).astype(str)
    return df

def _fingerprints(df: pd.DataFrame):
    df = _ensure_cols(df.copy())
    f_url    = df["url_canon"].fillna("").map(_sha)
    f_domain = df["domain"].fillna("").map(_sha)
    f_text   = df["_text_joined"].fillna("").map(_sha)
    return set(f_url), set(f_domain), set(f_text)

def _detailed_analysis(df1, df2, name1, name2):
    """تحليل مفصّل للـoverlap"""
    df1 = _ensure_cols(df1.copy())
    df2 = _ensure_cols(df2.copy())
    
    # البحث عن domains مشتركة
    domains1 = set(df1["domain"].fillna(""))
    domains2 = set(df2["domain"].fillna(""))
    common_domains = domains1 & domains2
    
    if common_domains:
        print(f"\n📊 Detailed Analysis: {name1} ∩ {name2}")
        print(f"   Common domains: {len(common_domains)}")
        
        # أكثر domains تكراراً
        domain_counts = Counter()
        for dom in common_domains:
            count1 = len(df1[df1["domain"] == dom])
            count2 = len(df2[df2["domain"] == dom])
            domain_counts[dom] = count1 + count2
        
        print(f"   Top 10 overlapping domains:")
        for dom, count in domain_counts.most_common(10):
            count1 = len(df1[df1["domain"] == dom])
            count2 = len(df2[df2["domain"] == dom])
            label1 = "mixed" if LABEL_COL in df1.columns else "N/A"
            if LABEL_COL in df1.columns:
                labels = df1[df1["domain"] == dom][LABEL_COL].unique()
                label1 = f"{len(labels)} classes"
            print(f"      {dom[:50]:50s} | {name1}={count1:5d} | {name2}={count2:5d}")

def audit(train_path, val_path, test_path, verbose=True):
    print("=" * 70)
    print("🔍 Split Leakage Audit - Enhanced Version")
    print("=" * 70)
    
    # تحميل البيانات
    print("\n[1/3] Loading data...")
    tr = _load_csv(train_path)
    va = _load_csv(val_path)
    te = _load_csv(test_path)
    print(f"   ✓ Train: {len(tr):,}  Val: {len(va):,}  Test: {len(te):,}")
    
    # حساب fingerprints
    print("\n[2/3] Computing fingerprints...")
    f_tr = _fingerprints(tr)
    f_va = _fingerprints(va)
    f_te = _fingerprints(te)
    print("   ✓ URL, domain, and text hashes computed")
    
    # تقاطعات
    print("\n[3/3] Checking intersections...")
    uv_tr_va = (len(f_tr[0] & f_va[0]), len(f_tr[1] & f_va[1]), len(f_tr[2] & f_va[2]))
    uv_tr_te = (len(f_tr[0] & f_te[0]), len(f_tr[1] & f_te[1]), len(f_tr[2] & f_te[2]))
    uv_va_te = (len(f_va[0] & f_te[0]), len(f_va[1] & f_te[1]), len(f_va[2] & f_te[2]))
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Train: {len(tr):,}  Val: {len(va):,}  Test: {len(te):,}")
    print("\nIntersections (url, domain, text):")
    print(f"Train ∩ Val : {uv_tr_va}")
    print(f"Train ∩ Test: {uv_tr_te}")
    print(f"Val   ∩ Test: {uv_va_te}")
    
    # تحليل النتائج
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    alerts = []
    critical = False
    
    # فحص Train ∩ Test (الأهم!)
    url_tr_te, dom_tr_te, txt_tr_te = uv_tr_te
    if url_tr_te > 0 or dom_tr_te > 0 or txt_tr_te > 0:
        alerts.append(("Train ∩ Test", uv_tr_te))
        if dom_tr_te > 100 or url_tr_te > 50:
            critical = True
    
    # فحص Train ∩ Val
    url_tr_va, dom_tr_va, txt_tr_va = uv_tr_va
    if url_tr_va > 0 or dom_tr_va > 0 or txt_tr_va > 0:
        alerts.append(("Train ∩ Val", uv_tr_va))
    
    # فحص Val ∩ Test
    url_va_te, dom_va_te, txt_va_te = uv_va_te
    if url_va_te > 0 or dom_va_te > 0 or txt_va_te > 0:
        alerts.append(("Val ∩ Test", uv_va_te))
    
    if not alerts:
        print("✅ EXCELLENT: No overlap detected!")
        print("\n   Your splits are clean:")
        print("   - No URL overlap")
        print("   - No domain overlap")
        print("   - No text overlap")
        print("\n   ✅ Model evaluation is reliable")
        print("   ✅ Results can be trusted")
        print("   ✅ Ready for production!")
        return 0
    
    if critical:
        print("❌ CRITICAL: Significant Train/Test overlap detected!")
        print("\n   This is a serious problem:")
        print(f"   - {dom_tr_te} overlapping domains between Train/Test")
        print(f"   - {url_tr_te} overlapping URLs between Train/Test")
        print("\n   ⚠️ Your model may be memorizing instead of learning!")
        print("   ⚠️ Test results are likely overoptimistic")
        print("   ⚠️ Performance will drop in production")
        print("\n   📝 REQUIRED ACTIONS:")
        print("   1. Re-split data using domain-based or time-based splits")
        print("   2. Ensure no domain appears in both train and test")
        print("   3. Re-train model with clean splits")
        print("   4. Re-evaluate performance")
        
        if verbose:
            _detailed_analysis(tr, te, "Train", "Test")
        return 2
    
    # Overlap موجود لكن ليس critical
    print("⚠️ WARNING: Some overlap detected")
    print("\n   Detected overlaps:")
    for name, tup in alerts:
        url_ov, dom_ov, txt_ov = tup
        severity = "HIGH" if dom_ov > 50 else "MEDIUM" if dom_ov > 10 else "LOW"
        print(f"   - {name}: {tup} (Severity: {severity})")
    
    if dom_tr_te > 0:
        print("\n   ⚠️ Train/Test domain overlap exists")
        print("   This may cause overoptimistic results")
        print("   Recommended: Re-split data")
        if verbose:
            _detailed_analysis(tr, te, "Train", "Test")
    
    if dom_tr_va > 0 and dom_tr_te == 0:
        print("\n   ℹ️ Train/Val overlap exists (but Train/Test is clean)")
        print("   This is less critical but not ideal")
        print("   Validation metrics may be overoptimistic")
    
    if dom_va_te > 0 and dom_tr_te == 0:
        print("\n   ℹ️ Val/Test overlap exists (but Train/Test is clean)")
        print("   This is less critical")
        print("   Main concern: Test set may not be fully independent")
    
    print("\n   📝 RECOMMENDED ACTIONS:")
    if dom_tr_te > 0:
        print("   1. ⚠️ PRIORITY: Re-split to eliminate Train/Test overlap")
        print("   2. Use domain-based or time-based splitting")
        print("   3. Re-train and re-evaluate")
    else:
        print("   1. Consider re-splitting to eliminate all overlaps")
        print("   2. Current model is likely still usable")
        print("   3. Be aware of potential optimistic bias in metrics")
    
    return 1

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python audit_splits_enhanced.py <train.csv> <val.csv> <test.csv>")
        print("\nExample:")
        print("  python audit_splits_enhanced.py train.csv val.csv test.csv")
        sys.exit(1)
    
    exit_code = audit(sys.argv[1], sys.argv[2], sys.argv[3], verbose=True)
    sys.exit(exit_code)