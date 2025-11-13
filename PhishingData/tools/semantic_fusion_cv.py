# -*- coding: utf-8 -*-
"""
Semantic Fusion CV: TF-IDF + SVD + Multilingual Embeddings (Sentence-Transformers)
- Domain-aware, leakage-safe CV (StratifiedGroupKFold if available)
- Robust grouping for no-domain samples using text hash
- FIXED: Lazy import to avoid Windows DLL issues
"""

import os, re, time, argparse, warnings, hashlib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.sparse import hstack, csr_matrix
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ✅ FIXED: Don't set CUDA_VISIBLE_DEVICES (causes DLL issues on Windows)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ✅ FIXED: Lazy import of sentence-transformers (avoid early torch loading)
SentenceTransformer = None
_ST_CHECKED = False
_HAS_ST = False
_ST_ERR = ""

def _ensure_sentence_transformers():
    """Lazy load sentence-transformers only when needed"""
    global SentenceTransformer, _ST_CHECKED, _HAS_ST, _ST_ERR
    
    if _ST_CHECKED:
        return
    
    _ST_CHECKED = True
    try:
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
        _HAS_ST = True
    except Exception as e:
        _HAS_ST = False
        _ST_ERR = str(e)

def _load_st_model(name: str, device: str):
    _ensure_sentence_transformers()
    
    if not _HAS_ST:
        print(f"[warn] sentence_transformers unavailable: {_ST_ERR}")
        return None
    try:
        m = SentenceTransformer(name, device=device)
        return m
    except Exception as e:
        print(f"[warn] ST load failed ({name}): {e}")
        return None

def embed_corpus(model, texts, batch_size: int = 256, max_len: int = 256):
    if model is None:
        return None
    # limit token length
    try:
        model.max_seq_length = max_len
    except Exception:
        pass
    return model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True
    ).astype(np.float32, copy=False)

# ---------- helpers ----------
try:
    import tldextract
    _HAS_TLD = True
except Exception:
    tldextract = None
    _HAS_TLD = False

def _extract_domain(u: str) -> str | None:
    if not u: return None
    u = str(u).strip()
    if not u: return None
    try:
        if _HAS_TLD:
            ex = tldextract.extract(u)
            dom = ".".join([p for p in [ex.domain, ex.suffix] if p])
            return dom or None
        if not re.match(r'^\w+://', u):
            u = f"http://{u}"
        netloc = urlparse(u).netloc
        if not netloc:
            return None
        return netloc.split(":")[0] or None
    except Exception:
        return None

def _text_fingerprint(s: str, n: int = 256) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return hashlib.md5(s[:n].encode("utf-8")).hexdigest()  # stable

def build_text_and_groups(df: pd.DataFrame) -> pd.DataFrame:
    # 1) fused text
    url_txt = df.get("url_canon", "").fillna("")
    url_txt = url_txt.mask(lambda s: s.str.len() == 0, df.get("url", "").fillna(""))
    email_txt = (df.get("email_subject", "").fillna("") + " " +
                 df.get("email_body", "").fillna("")).str.strip()
    fused = url_txt.copy()
    fused = fused.mask(fused.str.len() == 0, email_txt)
    fused = fused.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["text"] = fused
    df = df[df["text"].str.len() > 0].copy()

    # 2) domain
    dom = df.get("domain")
    if dom is None:
        dom = pd.Series([None]*len(df), index=df.index)
    dom = dom.astype("string")
    need = dom.isna() | (dom.str.len() == 0)
    if need.any():
        recov = df.loc[need, ["url_canon","url"]].astype(str).apply(
            lambda r: _extract_domain(r["url_canon"] or r["url"]), axis=1
        )
        dom.loc[need] = pd.Series(recov, index=dom.loc[need].index).astype("string")

    dom = dom.fillna("no-domain").replace({"": "no-domain"})
    modality = df.get("modality", "url").astype(str).str.lower().fillna("url")

    # 3) robust grouping: if no-domain → group by text fingerprint, not one mega-bucket
    is_nodom = (dom == "no-domain")
    group_key = modality + ":" + dom
    if is_nodom.any():
        fp = df.loc[is_nodom, "text"].map(_text_fingerprint)
        group_key.loc[is_nodom] = modality.loc[is_nodom] + ":nodom:" + fp

    df["_group_key"] = group_key
    df["_strata"]    = modality
    return df

def make_splits(full_df: pd.DataFrame, label_col: str, folds: int, seed: int = 42):
    y = full_df[label_col].astype(str)
    if "_strata" in full_df.columns:
        y = y.astype(str) + "|" + full_df["_strata"].astype(str)
    groups = full_df["_group_key"].values
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        it = sgkf.split(full_df, y=y, groups=groups)
        print(f"✅ Using StratifiedGroupKFold (n_splits={folds})")
    except Exception:
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=folds)
        it = gkf.split(full_df, groups=groups)
        print(f"⚠️  Fallback: GroupKFold (no stratification)")
    return list(it)

def _to_binary(y_series, pos_set=("1","true","phish","spam","malicious","bad")):
    s = y_series.astype(str).str.strip().str.lower()
    mapping = {k: 1 for k in pos_set}
    mapping.update({"0":0,"false":0,"benign":0,"ham":0,"good":0,"safe":0})
    y = s.map(mapping)
    if y.isna().any():
        try: y = s.astype(float).astype(int)
        except Exception:
            bad = sorted(set(s[y.isna()]))
            raise ValueError(f"Label not convertible to 0/1. Unmapped: {bad}")
    return y.values

def _pick_label_col(df: pd.DataFrame) -> str:
    for c in ("label","is_phish","target","y"):
        if c in df.columns: return c
    raise ValueError("No label column found")

def _recall_at_fpr(y_true, scores, fpr_max=0.003):
    y_true = np.asarray(y_true)
    neg = int((y_true == 0).sum())
    if neg == 0:
        return np.nan, float("inf")
    target = max(fpr_max, 1.0/neg)
    fpr, tpr, thr = roc_curve(y_true, scores)
    # guard for non-monotonicities
    tpr_at = float(np.interp(target, fpr, tpr, left=tpr[0], right=tpr[-1]))
    th_at  = float(np.interp(target, fpr, thr, left=thr[0], right=thr[-1]))
    return tpr_at, th_at

# ---------- features ----------
def build_features(corpus: np.ndarray, use_embeddings: bool, svd_components: int,
                   emb_model: str, device: str, emb_batch: int, emb_maxlen: int):
    print("\n[Feature Building]")
    # TF-IDF (char + word)
    print("   🔄 TF-IDF ...")
    char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5),
                               min_df=5, max_df=0.95, max_features=50000,
                               lowercase=True, strip_accents="unicode", dtype=np.float32)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1,3),
                               min_df=5, max_df=0.95, max_features=50000,
                               token_pattern=r"\b\w+\b",
                               lowercase=True, strip_accents="unicode", dtype=np.float32)
    Xc = char_vec.fit_transform(corpus)
    Xw = word_vec.fit_transform(corpus)
    X_tfidf = hstack([Xc, Xw]).tocsr()
    print(f"      ✅ TF-IDF: {X_tfidf.shape[1]:,} feats (char {Xc.shape[1]:,} + word {Xw.shape[1]:,})")

    # SVD
    X_svd = None; svd_var = 0.0
    if svd_components > 0 and X_tfidf.shape[1] > svd_components:
        print(f"   🔄 SVD(n={svd_components}) ...")
        svd = TruncatedSVD(n_components=min(svd_components, X_tfidf.shape[1]-1),
                           random_state=RANDOM_SEED)
        X_svd = svd.fit_transform(X_tfidf).astype(np.float32, copy=False)
        svd_var = float(svd.explained_variance_ratio_.sum())
        print(f"      ✅ SVD: {X_svd.shape[1]} comps, {svd_var:.2%} variance")

    # Embeddings
    X_emb = None; emb_dim = 0
    if use_embeddings:
        print("   🔄 Embeddings ...")
        st = _load_st_model(emb_model, device=device)
        X_emb = embed_corpus(st, corpus, batch_size=emb_batch, max_len=emb_maxlen)
        if X_emb is not None:
            emb_dim = X_emb.shape[1]
            print(f"      ✅ Embeddings: dim={emb_dim}")

    # fuse
    print("   🔄 Fusing ...")
    if X_svd is None:
        base = X_tfidf
    else:
        base = csr_matrix(X_svd)  # treat dense as sparse container for hstack

    if (not use_embeddings) or (X_emb is None):
        X_final = base
        kinds = [f"SVD({X_svd.shape[1]})" if X_svd is not None else f"TF-IDF({X_tfidf.shape[1]})"]
    else:
        X_final = hstack([base, csr_matrix(X_emb)], format="csr")
        kinds = [f"SVD({X_svd.shape[1]})" if X_svd is not None else f"TF-IDF({X_tfidf.shape[1]})",
                 f"Emb({emb_dim})"]
    info = dict(tfidf_shape=X_tfidf.shape, svd_components=(X_svd.shape[1] if X_svd is not None else 0),
                svd_variance=svd_var, embeddings_enabled=(X_emb is not None),
                embeddings_dim=emb_dim, total_features=X_final.shape[1], feature_types=kinds)
    print(f"      ✅ Final features: {X_final.shape[1]:,} ({' + '.join(kinds)})")
    return X_final, info

# ---------- main ----------
def main(args):
    print("="*70); print("  🌐 Semantic Fusion Cross-Validation @ Fixed FPR"); print("="*70)

    # ✅ FIXED: Check sentence-transformers only when needed
    if args.emb:
        _ensure_sentence_transformers()
        if not _HAS_ST:
            print("❌ sentence-transformers not installed. Run: pip install sentence-transformers")
            print(f"   Error: {_ST_ERR}")
            return
        try:
            from sentence_transformers import __version__ as st_version
            print(f"✅ sentence-transformers {st_version}")
        except Exception:
            print("✅ sentence-transformers installed (version unknown)")

    # 1) data
    print("\n[1/7] Loading data ...")
    train = pd.read_csv(args.train, low_memory=False)
    val   = pd.read_csv(args.val,   low_memory=False)
    df = pd.concat([train, val], ignore_index=True)
    print(f"   ✓ Train={len(train):,}  Val={len(val):,}  Total={len(df):,}")

    # 2) text & groups
    print("\n[2/7] Building text & groups ...")
    LABEL = _pick_label_col(df)
    print(f"   ✓ Label: {LABEL}")
    df = build_text_and_groups(df)
    print(f"   ✓ Text built, samples: {len(df):,}")
    print(f"   ✓ Unique groups: {df['_group_key'].nunique():,}")

    y = _to_binary(df[LABEL])
    corpus = df["text"].fillna("").astype(str).values
    pos_rate = float(y.mean())
    print(f"   ✓ Class distribution: Pos={pos_rate:.2%}  Neg={(1-pos_rate):.2%}")

    # 3) features
    print("\n[3/7] Features ...")
    X, info = build_features(
        corpus=corpus, use_embeddings=args.emb, svd_components=args.svd,
        emb_model=args.emb_model, device=args.device,
        emb_batch=args.emb_batch, emb_maxlen=args.emb_maxlen
    )
    print(f"\n   📊 Summary: TF-IDF={info['tfidf_shape'][1]:,}  "
          f"SVD={info['svd_components']}({info['svd_variance']:.2%})  "
          f"{'Emb='+str(info['embeddings_dim']) if info['embeddings_enabled'] else ''}  "
          f"Total={info['total_features']:,}")

    # 4) classifier
    print("\n[4/7] Classifier ...")
    try:
        from lightgbm import LGBMClassifier
        def clf_ctor():
            return LGBMClassifier(n_estimators=800, learning_rate=0.07,
                                  num_leaves=255, max_depth=-1,
                                  subsample=1.0, colsample_bytree=1.0,
                                  min_child_samples=20, random_state=RANDOM_SEED,
                                  n_jobs=-1, verbosity=-1)
        use_proba = True
        print("   ✓ LightGBM")
    except Exception:
        from sklearn.linear_model import LogisticRegression
        def clf_ctor(): return LogisticRegression(max_iter=2000, n_jobs=-1, random_state=RANDOM_SEED)
        use_proba = False
        print("   ⚠️  Fallback: LogisticRegression")

    # 5) CV
    print(f"\n[5/7] {args.folds}-Fold CV ...")
    splits = make_splits(df, LABEL, args.folds, RANDOM_SEED)
    print(f"\nSamples: {len(df):,} | Features: {info['total_features']:,} | Folds: {args.folds}")
    print("="*70)

    recalls, aucs, times = [], [], []
    for i,(tr,te) in enumerate(splits,1):
        t0 = time.time()
        y_tr, y_te = y[tr], y[te]
        pos, neg = int(y_te.sum()), int(len(y_te)-y_te.sum())
        if len(np.unique(y_te)) < 2:
            print(f"\nFold {i}: ⚠️  SKIPPED (single-class). Pos={pos:,} Neg={neg:,}")
            recalls.append(np.nan); aucs.append(np.nan); continue

        model = clf_ctor(); model.fit(X[tr], y_tr)
        scores = model.predict_proba(X[te])[:,1] if use_proba else model.decision_function(X[te])

        try:
            tpr_at, th_at = _recall_at_fpr(y_te, scores, fpr_max=0.003)
        except Exception as e:
            print(f"   ⚠️ recall@fpr error: {e}"); tpr_at, th_at = np.nan, float("inf")
        try:
            auc = roc_auc_score(y_te, scores)
        except Exception:
            auc = np.nan

        recalls.append(tpr_at); aucs.append(auc); times.append(time.time()-t0)
        print(f"\nFold {i}:")
        print(f"   Recall@FPR≤0.3%: {tpr_at:.4f}")
        print(f"   AUC-ROC:         {auc:.4f}" if not np.isnan(auc) else "   AUC-ROC:         N/A")
        print(f"   Samples:         Pos={pos:,}, Neg={neg:,}")
        print(f"   Threshold:       {th_at:.6f}" if np.isfinite(th_at) else "   Threshold:       N/A")
        print(f"   Time:            {times[-1]:.1f}s")

        if args.dump_fold == i:
            print(f"\n   [Diagnostic Dump for Fold {i}]")
            te_df = df.iloc[te].copy()
            te_df["score"] = scores
            if np.isfinite(th_at):
                te_df["pred"] = (te_df["score"] >= th_at).astype(int)
            else:
                q = min(max(int(0.997*len(te_df)), 1), len(te_df)-1)
                th_at = float(np.partition(scores, q)[q])
                te_df["pred"] = (te_df["score"] >= th_at).astype(int)
            FP = te_df[(te_df["pred"]==1) & (y_te==0)]
            FN = te_df[(te_df["pred"]==0) & (y_te==1)]
            dom_stats = te_df.groupby("_group_key").agg(
                n=("score","size"),
                pos_rate=(LABEL, lambda s: (s.astype(str).str.lower()
                    .isin(["1","true","phish","spam","malicious","bad"])).mean())
            ).sort_values("n", ascending=False)
            os.makedirs(args.outdir, exist_ok=True)
            dom_stats.to_csv(os.path.join(args.outdir, f"fold{i}_domains.csv"), index=True)
            FP.head(200).to_csv(os.path.join(args.outdir, f"fold{i}_FP_head.csv"), index=False)
            FN.head(200).to_csv(os.path.join(args.outdir, f"fold{i}_FN_head.csv"), index=False)
            print(f"   ✓ Saved CSVs to {args.outdir}")

    # 6) summary
    print("\n"+"="*70); print("  SUMMARY"); print("="*70)
    vrec = [r for r in recalls if not (r is None or np.isnan(r))]
    vaucs = [a for a in aucs if not (a is None or np.isnan(a))]
    if vrec:
        print(f"\nRecall@FPR≤0.3%:  Mean={np.mean(vrec):.4f}  ±{np.std(vrec):.4f}")
        print(f"Per-fold: {[round(r,4) if not np.isnan(r) else None for r in recalls]}")
    else:
        print("\n⚠️  No valid folds.")
    if vaucs:
        print(f"AUC-ROC:           Mean={np.mean(vaucs):.4f}  ±{np.std(vaucs):.4f}")
    if times:
        print(f"Timing:            Avg={np.mean(times):.1f}s  Total={sum(times):.1f}s")
    print("\n✅ Semantic Fusion CV completed successfully!")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Semantic Fusion CV: TF-IDF + SVD + Multilingual Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Baseline (TF-IDF + SVD)
  python semantic_fusion_cv.py --train train.csv --val val.csv --folds 5 --svd 768

  # Add multilingual embeddings
  python semantic_fusion_cv.py --train train.csv --val val.csv --folds 5 --svd 768 --emb --dump-fold 3
"""
    )
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--svd", type=int, default=256)

    # embeddings options
    ap.add_argument("--emb", action="store_true", help="enable sentence-transformers embeddings")
    ap.add_argument("--emb-model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                    help="HF model id")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--emb-batch", type=int, default=32)
    ap.add_argument("--emb-maxlen", type=int, default=256)

    # diagnostics
    ap.add_argument("--dump-fold", type=int, default=0)
    ap.add_argument("--outdir", default="PhishingData/artifacts/semantic_debug")
    args = ap.parse_args()
    main(args)