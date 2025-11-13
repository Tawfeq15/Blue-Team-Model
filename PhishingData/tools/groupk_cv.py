# -*- coding: utf-8 -*-
"""
groupk_cv.py
============
Cross-Validation محترم للتسرّب: يستخدم StratifiedGroupKFold لمنع تسرب نفس الدومين
بين Train/Val، مع قياس Recall@FPR≤0.3% وتصدير ملفات تشخيصية للـFP/FN/Domains.

الاستخدام:
----------
python groupk_cv.py --train <train.csv> --val <val.csv> --folds 5 --svd 768 --dump-fold 1

الميزات:
--------
- بناء نص موحد من URL/Email
- استخراج دومين قوي للتجميع
- TF-IDF (char 3-5 + word 1-3) + SVD اختياري
- StratifiedGroupKFold (fallback إلى GroupKFold)
- حراسة كاملة من الفولدات أحادية الكلاس
- قياس Recall@FPR≤0.3% + AUC آمن
- تصدير ملفات تشخيص: domains/FP/FN
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.sparse import hstack

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ============================================================================
# Helpers: Domain Extraction & Text Building
# ============================================================================
import re
from urllib.parse import urlparse

try:
    import tldextract
    HAS_TLDEXTRACT = True
except ImportError:
    HAS_TLDEXTRACT = False
    tldextract = None


def _extract_domain(u: str) -> str | None:
    """
    استخراج دومين قوي من URL باستخدام tldextract أو urlparse.
    
    Args:
        u: URL string
        
    Returns:
        domain string أو None إذا فشل الاستخراج
    """
    if not u:
        return None
    u = str(u).strip()
    if not u:
        return None
    
    try:
        if HAS_TLDEXTRACT:
            ex = tldextract.extract(u)
            dom = ".".join([p for p in [ex.domain, ex.suffix] if p])
            return dom or None
        
        # Fallback: استخدام urlparse
        if not re.match(r'^\w+://', u):
            u = f"http://{u}"
        netloc = urlparse(u).netloc
        if not netloc:
            return None
        # إزالة port إن وجد
        return netloc.split(":")[0] or None
    except Exception:
        return None


def build_text_and_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    بناء نص موحد للـTF-IDF ومفاتيح التجميع لمنع التسريب.
    
    يُنشئ:
    - text: نص موحد من url_canon/url أو email_subject+body
    - _group_key: modality:domain (للتجميع في CV)
    - _strata: modality (للطبقات في StratifiedGroupKFold)
    
    Args:
        df: DataFrame يحتوي على أعمدة URL و/أو Email
        
    Returns:
        DataFrame محدّث مع الأعمدة الجديدة (مفلتر لإزالة النصوص الفارغة)
    """
    # 1) بناء نص موحد للفكتورايزر
    url_txt = df.get("url_canon", "").fillna("")
    url_txt = url_txt.mask(lambda s: s.str.len() == 0, df.get("url", "").fillna(""))
    
    email_txt = (
        df.get("email_subject", "").fillna("") + " " +
        df.get("email_body", "").fillna("")
    ).str.strip()
    
    # دمج: أولوية للـURL، ثم الإيميل
    fused = url_txt.copy()
    fused = fused.mask(fused.str.len() == 0, email_txt)
    fused = fused.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    
    df["text"] = fused
    df = df[df["text"].str.len() > 0].copy()
    
    # 2) استخراج دومين قوي
    dom = df.get("domain")
    if dom is None:
        dom = pd.Series([None] * len(df), index=df.index)
    dom = dom.astype("string")
    
    # استرجاع الدومينات المفقودة من URL
    need = dom.isna() | (dom.str.len() == 0)
    if need.any():
        recov = df.loc[need, ["url_canon", "url"]].astype(str).apply(
            lambda r: _extract_domain(r["url_canon"] or r["url"]), axis=1
        )
        dom.loc[need] = pd.Series(recov, index=dom.loc[need].index).astype("string")
    
    dom = dom.fillna("no-domain").replace({"": "no-domain"})
    modality = df.get("modality", "url").astype(str).str.lower().fillna("url")
    
    # 3) مفاتيح CV: group_key للتجميع و strata للطبقات
    df["_group_key"] = modality + ":" + dom
    df["_strata"] = modality
    
    return df


def make_splits(full_df: pd.DataFrame, label_col: str, folds: int, seed: int = 42):
    """
    إنشاء تقسيمات CV مع stratification و grouping.
    
    يحاول استخدام StratifiedGroupKFold (sklearn >= 1.1)،
    مع fallback إلى GroupKFold العادي.
    
    Args:
        full_df: DataFrame كامل
        label_col: اسم عمود الـlabel
        folds: عدد الفولدات
        seed: random seed
        
    Returns:
        list of (train_idx, test_idx) tuples
    """
    y = full_df[label_col].astype(str)
    
    # دمج الطبقات الإضافية (modality) مع الـlabel
    if "_strata" in full_df.columns:
        y = y.astype(str) + "|" + full_df["_strata"].astype(str)
    
    groups = full_df["_group_key"].values
    
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = sgkf.split(full_df, y=y, groups=groups)
        print(f"✅ Using StratifiedGroupKFold (n_splits={folds})")
    except (ImportError, Exception) as e:
        print(f"⚠️  StratifiedGroupKFold not available, falling back to GroupKFold")
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=folds)
        split_iter = gkf.split(full_df, groups=groups)
    
    return list(split_iter)


# ============================================================================
# Utilities: Label Conversion & Metrics
# ============================================================================

def _to_binary(y_series, pos_set=("1", "true", "phish", "spam", "malicious", "bad")):
    """
    تحويل عمود الـlabel إلى 0/1 binary.
    
    Args:
        y_series: pandas Series يحتوي على labels
        pos_set: قيم تُعتبر إيجابية (phishing/malicious)
        
    Returns:
        numpy array من 0/1
        
    Raises:
        ValueError: إذا فشل التحويل
    """
    s = y_series.astype(str).str.strip().str.lower()
    
    mapping = {k: 1 for k in pos_set}
    mapping.update({
        "0": 0, "false": 0, "benign": 0, 
        "ham": 0, "good": 0, "safe": 0
    })
    
    y = s.map(mapping)
    
    if y.isna().any():
        try:
            y = s.astype(float).astype(int)
        except Exception:
            bad = sorted(set(s[y.isna()]))
            raise ValueError(
                f"Label column not convertible to 0/1. Unmapped values: {bad}"
            )
    
    return y.values


def _pick_label_col(df: pd.DataFrame) -> str:
    """
    اختيار عمود الـlabel تلقائياً من الأسماء الشائعة.
    
    Args:
        df: DataFrame
        
    Returns:
        اسم عمود الـlabel
        
    Raises:
        ValueError: إذا لم يتم العثور على عمود label
    """
    for c in ("label", "is_phish", "target", "y"):
        if c in df.columns:
            return c
    raise ValueError("No label column found in DataFrame")


def _recall_at_fpr(y_true, scores, fpr_max=0.003):
    """
    حساب Recall (TPR) عند FPR محدد (افتراضياً 0.3%).
    
    Args:
        y_true: true labels (0/1)
        scores: predicted scores/probabilities
        fpr_max: الحد الأقصى للـFPR المسموح (0.003 = 0.3%)
        
    Returns:
        tuple: (recall_at_fpr, threshold_at_fpr)
    """
    y_true = np.asarray(y_true)
    neg = int((y_true == 0).sum())
    
    # إذا لم يكن هناك سلبيات، لا يمكن حساب FPR
    if neg == 0:
        return np.nan, float("inf")
    
    # تأكد أن الـtarget FPR ليس أقل من 1/n_negatives
    target = max(fpr_max, 1.0 / neg)
    
    fpr, tpr, thr = roc_curve(y_true, scores)
    
    # استيفاء للحصول على TPR والعتبة عند FPR المطلوب
    tpr_at = float(np.interp(target, fpr, tpr, left=tpr[0], right=tpr[-1]))
    th_at = float(np.interp(target, fpr, thr, left=thr[0], right=thr[-1]))
    
    return tpr_at, th_at


# ============================================================================
# Main CV Loop
# ============================================================================

def main(args):
    """
    الدالة الرئيسية: تحميل البيانات، بناء الميزات، تشغيل CV، وتصدير التشخيص.
    """
    print("=" * 70)
    print("  Domain-Aware Group Cross-Validation @ Fixed FPR")
    print("=" * 70)
    
    # ────────────────────────────────────────────────────────────────────
    # 1) تحميل ودمج البيانات
    # ────────────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    read_kwargs = dict(low_memory=False)
    train = pd.read_csv(args.train, **read_kwargs)
    val = pd.read_csv(args.val, **read_kwargs)
    df = pd.concat([train, val], ignore_index=True)
    print(f"   ✓ Loaded: Train={len(train):,}, Val={len(val):,}, Total={len(df):,}")
    
    # ────────────────────────────────────────────────────────────────────
    # 2) تحديد عمود الـlabel وبناء النص والمجموعات
    # ────────────────────────────────────────────────────────────────────
    print("\n[2/6] Building text and groups...")
    LABEL = _pick_label_col(df)
    print(f"   ✓ Label column: '{LABEL}'")
    
    df = build_text_and_groups(df)
    print(f"   ✓ Text built, samples after filtering: {len(df):,}")
    print(f"   ✓ Unique domains: {df['_group_key'].nunique():,}")
    
    y = _to_binary(df[LABEL])
    corpus = df["text"].fillna("").astype(str).values
    
    pos_rate = y.mean()
    print(f"   ✓ Class distribution: Positive={pos_rate:.2%}, Negative={1-pos_rate:.2%}")
    
    # ────────────────────────────────────────────────────────────────────
    # 3) بناء ميزات TF-IDF (char + word)
    # ────────────────────────────────────────────────────────────────────
    print("\n[3/6] Building TF-IDF features...")
    
    char_vec = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5),
        min_df=5, max_df=0.95, max_features=50000,
        lowercase=True, strip_accents="unicode", dtype=np.float32
    )
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 3),
        min_df=5, max_df=0.95, max_features=50000,
        token_pattern=r"\b\w+\b",
        lowercase=True, strip_accents="unicode", dtype=np.float32
    )
    
    Xc = char_vec.fit_transform(corpus)
    Xw = word_vec.fit_transform(corpus)
    X = hstack([Xc, Xw]).tocsr()
    
    print(f"   ✓ TF-IDF features: {X.shape[1]:,} (char: {Xc.shape[1]:,}, word: {Xw.shape[1]:,})")
    
    # ────────────────────────────────────────────────────────────────────
    # 4) SVD اختياري لتخفيض الأبعاد
    # ────────────────────────────────────────────────────────────────────
    if args.svd > 0 and X.shape[1] > args.svd:
        print(f"\n[4/6] Applying SVD (n_components={args.svd})...")
        svd = TruncatedSVD(
            n_components=min(args.svd, X.shape[1] - 1),
            random_state=RANDOM_SEED
        )
        Xsvd = svd.fit_transform(X)  # dense array
        explained = svd.explained_variance_ratio_.sum()
        print(f"   ✓ SVD done: {Xsvd.shape[1]} components, {explained:.2%} variance explained")
    else:
        print("\n[4/6] Skipping SVD (--svd=0 or features <= svd)")
        Xsvd = X  # sparse matrix
    
    # ────────────────────────────────────────────────────────────────────
    # 5) اختيار المصنف
    # ────────────────────────────────────────────────────────────────────
    print("\n[5/6] Selecting classifier...")
    try:
        from lightgbm import LGBMClassifier
        
        def clf_ctor():
            return LGBMClassifier(
                n_estimators=800,
                learning_rate=0.07,
                num_leaves=255,
                max_depth=-1,
                subsample=1.0,
                colsample_bytree=1.0,
                min_child_samples=20,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbosity=-1
            )
        
        use_proba = True
        print("   ✓ Using LightGBM classifier")
    except ImportError:
        from sklearn.linear_model import LogisticRegression
        
        def clf_ctor():
            return LogisticRegression(max_iter=2000, n_jobs=-1, random_state=RANDOM_SEED)
        
        use_proba = False
        print("   ⚠️  LightGBM not available, using Logistic Regression")
    
    # ────────────────────────────────────────────────────────────────────
    # 6) تشغيل Cross-Validation
    # ────────────────────────────────────────────────────────────────────
    print(f"\n[6/6] Running {args.folds}-Fold Cross-Validation...")
    splits = make_splits(df, label_col=LABEL, folds=args.folds, seed=RANDOM_SEED)
    
    print(f"\nSamples: {len(df):,}  |  Features: {Xsvd.shape[1]:,}  |  Folds: {args.folds}")
    print("=" * 70)
    
    recalls = []
    aucs = []
    
    for i, (tr, te) in enumerate(splits, 1):
        y_tr, y_te = y[tr], y[te]
        unique_te = np.unique(y_te)
        pos, neg = int(y_te.sum()), int(len(y_te) - y_te.sum())
        
        # ────────────────────────────────────────────────────────────
        # حراسة: تخطي الفولدات أحادية الكلاس
        # ────────────────────────────────────────────────────────────
        if unique_te.size < 2:
            print(f"\nFold {i}: ⚠️  SKIPPED (single-class fold)")
            print(f"         Positive: {pos:,}, Negative: {neg:,}")
            recalls.append(np.nan)
            aucs.append(np.nan)
            continue
        
        # ────────────────────────────────────────────────────────────
        # تدريب وتنبؤ
        # ────────────────────────────────────────────────────────────
        model = clf_ctor()
        model.fit(Xsvd[tr], y_tr)
        
        if use_proba:
            scores = model.predict_proba(Xsvd[te])[:, 1]
        else:
            scores = model.decision_function(Xsvd[te])
        
        # ────────────────────────────────────────────────────────────
        # حساب Recall@FPR≤0.3%
        # ────────────────────────────────────────────────────────────
        try:
            tpr_at, th_at = _recall_at_fpr(y_te, scores, fpr_max=0.003)
        except Exception as e:
            print(f"         ⚠️  Error computing recall: {e}")
            tpr_at, th_at = np.nan, float("inf")
        
        # ────────────────────────────────────────────────────────────
        # حساب AUC بشكل آمن
        # ────────────────────────────────────────────────────────────
        try:
            auc = roc_auc_score(y_te, scores)
        except Exception:
            auc = np.nan
        
        recalls.append(tpr_at)
        aucs.append(auc)
        
        # طباعة النتائج
        print(f"\nFold {i}:")
        print(f"   Recall@FPR≤0.3%: {tpr_at:.4f}")
        print(f"   AUC-ROC:         {auc:.4f}" if not np.isnan(auc) else "   AUC-ROC:         N/A")
        print(f"   Samples:         Pos={pos:,}, Neg={neg:,}")
        print(f"   Threshold:       {th_at:.6f}" if np.isfinite(th_at) else "   Threshold:       N/A")
        
        # ────────────────────────────────────────────────────────────
        # تصدير ملفات التشخيص للفولد المحدد
        # ────────────────────────────────────────────────────────────
        if args.dump_fold == i:
            print(f"\n   [Diagnostic Dump for Fold {i}]")
            te_df = df.iloc[te].copy()
            te_df["score"] = scores
            
            # تحديد العتبة للتنبؤات
            if np.isfinite(th_at):
                te_df["pred"] = (te_df["score"] >= th_at).astype(int)
            else:
                # استخدام quantile كـfallback
                th_at = float(np.quantile(scores, 0.997))  # تقريباً FPR 0.3%
                te_df["pred"] = (te_df["score"] >= th_at).astype(int)
            
            # False Positives & False Negatives
            FP = te_df[(te_df["pred"] == 1) & (y_te == 0)]
            FN = te_df[(te_df["pred"] == 0) & (y_te == 1)]
            
            # إحصاءات الدومينات
            dom_stats = te_df.groupby("_group_key").agg(
                n=("score", "size"),
                pos_rate=(LABEL, lambda s: (
                    s.astype(str).str.lower().isin(
                        ["1", "true", "phish", "spam", "malicious", "bad"]
                    ).mean()
                ))
            ).sort_values("n", ascending=False)
            
            # حفظ الملفات
            os.makedirs(args.outdir, exist_ok=True)
            dom_path = os.path.join(args.outdir, f"fold{i}_domains.csv")
            fp_path = os.path.join(args.outdir, f"fold{i}_FP_head.csv")
            fn_path = os.path.join(args.outdir, f"fold{i}_FN_head.csv")
            
            dom_stats.to_csv(dom_path, index=True)
            FP.head(200).to_csv(fp_path, index=False)
            FN.head(200).to_csv(fn_path, index=False)
            
            print(f"   ✓ Saved: {dom_path}")
            print(f"   ✓ Saved: {fp_path} ({len(FP):,} FPs)")
            print(f"   ✓ Saved: {fn_path} ({len(FN):,} FNs)")
    
    # ────────────────────────────────────────────────────────────────────
    # الملخص النهائي
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    valid_recalls = [r for r in recalls if not (r is None or np.isnan(r))]
    valid_aucs = [a for a in aucs if not (a is None or np.isnan(a))]
    
    if valid_recalls:
        print(f"\nRecall@FPR≤0.3%:")
        print(f"   Mean: {np.mean(valid_recalls):.4f} ± {np.std(valid_recalls):.4f}")
        print(f"   Per-fold: {[round(r, 4) if not np.isnan(r) else None for r in recalls]}")
    else:
        print("\n⚠️  No valid folds to summarize (all folds were single-class)")
    
    if valid_aucs:
        print(f"\nAUC-ROC:")
        print(f"   Mean: {np.mean(valid_aucs):.4f} ± {np.std(valid_aucs):.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Cross-Validation completed successfully!")
    print("=" * 70)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Domain-aware Group Cross-Validation with fixed-FPR recall metric",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
مثال على الاستخدام:
─────────────────
$train = "$PWD\\PhishingData\\data\\merged_plus_synth_split_train_uniq.csv"
$val   = "$PWD\\PhishingData\\data\\merged_plus_synth_split_val_uniq.csv"
python groupk_cv.py --train $train --val $val --folds 5 --svd 768 --dump-fold 1

الخرج:
──────
- Recall@FPR≤0.3% لكل فولد + المتوسط
- ملفات تشخيصية (إذا تم تحديد --dump-fold):
  * fold{N}_domains.csv: إحصاءات الدومينات
  * fold{N}_FP_head.csv: أمثلة False Positives
  * fold{N}_FN_head.csv: أمثلة False Negatives
        """
    )
    
    ap.add_argument(
        "--train",
        required=True,
        help="مسار ملف التدريب (CSV)"
    )
    ap.add_argument(
        "--val",
        required=True,
        help="مسار ملف التحقق (CSV)"
    )
    ap.add_argument(
        "--folds",
        type=int,
        default=5,
        help="عدد الفولدات للـCV (افتراضي: 5)"
    )
    ap.add_argument(
        "--svd",
        type=int,
        default=256,
        help="عدد المكونات للـSVD (0 لتعطيله، افتراضي: 256)"
    )
    ap.add_argument(
        "--dump-fold",
        type=int,
        default=0,
        help="رقم الفولد لتصدير الملفات التشخيصية (0 = لا تصدير، افتراضي: 0)"
    )
    ap.add_argument(
        "--outdir",
        default="PhishingData/artifacts/groupk_debug",
        help="مجلد حفظ الملفات التشخيصية"
    )
    
    args = ap.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)