"""
================================================================================================
🛡️ Production-Ready Phishing Detection System - النسخة الاحترافية الكاملة
================================================================================================
نظام كشف التصيد الإلكتروني جاهز للإنتاج
يتضمن جميع التحسينات المطلوبة:
- تقسيم زمني وتجميع بالمصدر
- معايرة الاحتمالات
- اختيار العتبة الذكي
- اختبارات الصلابة الواقعية
- مراقبة الانجراف
- SHAP للتفسير
================================================================================================
"""

# =========================== Standard library ===========================
import os
import re
import json
import time
import hashlib
import warnings
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Any, Optional
import sys, csv
import random
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
import scipy.sparse as sp  # noqa
import requests
# =========================== Third-party core ===========================
import numpy as np
import pandas as pd
import shap
import joblib

# اختياري
# joblib.dump(self.data_cleaner, "data_cleaner.pkl")
# joblib.dump(self.drift_monitor, "drift_monitor.pkl")

# =========================== Scikit-learn ==============================
from sklearn.model_selection import (
    TimeSeriesSplit,
    GroupKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    roc_curve,
    brier_score_loss,
    log_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp  # noqa

# =========================== Boosting libs =============================
try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import catboost as cb
except Exception:
    cb = None

try:
    from xgboost.callback import EarlyStopping
except Exception:
    EarlyStopping = None


# =========================== Utils ====================================

# =========================== Enhanced TF-IDF Trainer ===================
# ملاحظة: اتركه كما هو إذا الملف enhanced_trainer.py موجود في نفس المجلد
# --- enhanced TF-IDF trainer (optional) ---
HAS_ENHANCED = False
try:
    from enhanced_trainer import EnhancedPhishingTrainer
    HAS_ENHANCED = True
except Exception as e:
    print(f"⚠️ USE_TFIDF=1 but failed to import enhanced_trainer: {e}")
    HAS_ENHANCED = False


# =========================== Misc =====================================
warnings.filterwarnings("ignore")

# تكبير حد حجم الحقل للصفوف الطويلة في CSV (حماية من أخطاء السطور الطويلة)
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

# (اختياري) تثبيت العشوائية لتكرارية النتائج
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# =========================== Dataset Ingestion (archives + folders + multi-files) ===========================
from pathlib import Path
import os, re, json, sys, csv
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd


# ————— helpers used by ingestion —————
def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.is_unique:
        df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _is_archive(p: Path) -> bool:
    s = p.suffix.lower()
    return s in {".zip", ".rar", ".7z", ".tar", ".gz", ".tgz", ".tar.gz"}


def _extract_archive(arc: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    suf = arc.suffix.lower()
    try:
        if suf == ".zip":
            import zipfile

            with zipfile.ZipFile(str(arc), "r") as zf:
                zf.extractall(dest)
        elif suf in {".tar", ".gz", ".tgz", ".tar.gz"}:
            import tarfile

            with tarfile.open(str(arc), "r:*") as tf:
                tf.extractall(dest)
        elif suf == ".rar":
            try:
                import rarfile

                with rarfile.RarFile(str(arc)) as rf:
                    rf.extractall(dest)
            except Exception as e:
                print(f"⚠️ RAR extraction failed ({e}). Install 'rarfile' and 'unrar'.")
        elif suf == ".7z":
            try:
                import py7zr

                with py7zr.SevenZipFile(str(arc), mode="r") as z:
                    z.extractall(path=dest)
            except Exception as e:
                print(f"⚠️ 7z extraction failed ({e}). Install 'py7zr'.")
    except Exception as e:
        print(f"⚠️ Failed to extract {arc.name}: {e}")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # أسقط أي أعمدة Unnamed
    for c in list(df.columns):
        if str(c).lower().startswith("unnamed"):
            df = df.drop(columns=[c])
    cols = {c.lower().strip(): c for c in df.columns}

    # SMS الشائع (v1=label, v2=text)
    if "v1" in cols and "v2" in cols:
        df = df.rename(columns={cols["v1"]: "label", cols["v2"]: "text"})

    # label → "label"
    for k in [
        "label",
        "labels",
        "class",
        "result",
        "target",
        "email type",
        "email_type",
        "type",
        "category",
        "y",
        "is_spam",
    ]:
        if k in cols and cols[k] != "label":
            df = df.rename(columns={cols[k]: "label"})
            break

    # text → "text"
    for k in [
        "text",
        "body",
        "message",
        "content",
        "email text",
        "email_text",
        "msg",
        "sms",
    ]:
        if k in cols and cols[k] != "text":
            df = df.rename(columns={cols[k]: "text"})
            break

    # url → "url"
    for k in ["url", "link", "href", "domain"]:
        if k in cols and cols[k] != "url":
            df = df.rename(columns={cols[k]: "url"})
            break

    return df


def _read_any_file(p: Path) -> Optional[pd.DataFrame]:
    try:
        name = p.name.lower()
        if name == "smsspamcollection":
            df = pd.read_csv(p, sep="\t", header=None, names=["label", "text"])
        elif p.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(
                    p, encoding="utf-8", engine="python", on_bad_lines="skip"
                )
            except UnicodeDecodeError:
                df = pd.read_csv(
                    p, encoding="latin-1", engine="python", on_bad_lines="skip"
                )
        elif p.suffix.lower() == ".tsv":
            df = pd.read_csv(p, sep="\t", engine="python", on_bad_lines="skip")
        elif p.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(p)
        elif p.suffix.lower() == ".jsonl":
            rows = []
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
            df = pd.DataFrame(rows)
        else:
            return None
    except Exception as e:
        print(f"⚠️ Failed to read {p.name}: {e}")
        return None
    return _dedupe_columns(_standardize_columns(df))


def _save_ready_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, base: Path
):
    (base / "train_data.csv").write_text(
        train_df.to_csv(index=False, encoding="utf-8", lineterminator="\n")
    )
    (base / "val_data.csv").write_text(
        val_df.to_csv(index=False, encoding="utf-8", lineterminator="\n")
    )
    (base / "test_data.csv").write_text(
        test_df.to_csv(index=False, encoding="utf-8", lineterminator="\n")
    )


def _prefer_splits_if_exist(
    base_dir: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    p = Path(base_dir)
    tr = p / "blue_multi_big_split_train.csv"
    va = p / "blue_multi_big_split_val.csv"
    te = p / "blue_multi_big_split_test.csv"
    if tr.exists() and va.exists() and te.exists():
        print(
            "✅ Found split CSVs (train/val/test). Using them and skipping ingestion."
        )
        df_tr = _dedupe_columns(
            pd.read_csv(tr, low_memory=False, encoding="utf-8", engine="python")
        )
        df_va = _dedupe_columns(
            pd.read_csv(va, low_memory=False, encoding="utf-8", engine="python")
        )
        df_te = _dedupe_columns(
            pd.read_csv(te, low_memory=False, encoding="utf-8", engine="python")
        )
        return df_tr, df_va, df_te
    return None, None, None


def _prepare_datasets_from_any_source(base_dir: str) -> Dict[str, Path]:
    # كبّر حد الحقل
    try:
        csv.field_size_limit(sys.maxsize)
    except Exception:
        try:
            csv.field_size_limit(2**31 - 1)
        except Exception:
            pass

    base = Path(base_dir)
    train_csv = base / "train_data.csv"
    val_csv = base / "val_data.csv"
    test_csv = base / "test_data.csv"

    # 0) لو موجودة جاهزة، اخرج
    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        print("✅ Found existing train/val/test CSVs. Skipping dataset preparation.")
        return {
            "base": base,
            "script_dir": Path(__file__).resolve().parent,
            "extracted_root": base / "_extracted",
            "train_csv": train_csv,
            "val_csv": val_csv,
            "test_csv": test_csv,
        }

    # 1) لو عندك blue_multi_big_split_* استخدمها وخلّص
    tr_df, va_df, te_df = _prefer_splits_if_exist(base_dir)
    if tr_df is not None:
        base.mkdir(parents=True, exist_ok=True)
        _save_ready_splits(tr_df, va_df, te_df, base)
        return {
            "base": base,
            "script_dir": Path(__file__).resolve().parent,
            "extracted_root": base / "_extracted",
            "train_csv": train_csv,
            "val_csv": val_csv,
            "test_csv": test_csv,
        }

    # 2) خلاف ذلك: استخرج واقرأ كل شيء
    script_dir = Path(__file__).resolve().parent
    base.mkdir(parents=True, exist_ok=True)
    extracted_root = base / "_extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)

    # جد كل المرشحين (ملفات/أرشيفات/مجلدات)
    def _iter_candidate_paths():
        dataset_env = os.getenv("DATASET_PATH")
        if dataset_env:
            yield Path(dataset_env)

        candidates = [
            base / "dataset.rar",
            base / "dataset.zip",
            base / "dataset.7z",
            base / "PhishingData.csv",
            base / "Phishing_Email.csv",
            base / "SMSSpamCollection",
            script_dir / "PhishingData.csv",
            script_dir / "Phishing_Email.csv",
            script_dir / "SMSSpamCollection",
        ]
        for c in candidates:
            if c.exists():
                yield c

        for p in list(base.glob("*")) + list(script_dir.glob("*")):
            if p.is_file() and (
                _is_archive(p)
                or p.suffix.lower() in {".csv", ".tsv", ".xlsx", ".xls", ".jsonl"}
                or p.name == "SMSSpamCollection"
            ):
                yield p

        for d in [base, script_dir]:
            for p in d.glob("*"):
                if p.is_dir() and p.name not in {
                    "artifacts",
                    "_extracted",
                    "__pycache__",
                }:
                    yield p

    # استخرج الأرشيفات
    extracted_dirs: List[Path] = []
    for src in _iter_candidate_paths():
        if src.is_file() and _is_archive(src):
            dest = extracted_root / src.stem
            print(f"📦 Extracting: {src}")
            _extract_archive(src, dest)
            if dest.exists():
                extracted_dirs.append(dest)

    # جهّز قائمة القراءة
    all_targets: List[Path] = []
    for src in _iter_candidate_paths():
        if src.is_dir():
            for p in src.rglob("*"):
                if p.is_file() and (
                    p.suffix.lower() in {".csv", ".tsv", ".xlsx", ".xls", ".jsonl"}
                    or p.name.lower() == "smsspamcollection"
                ):
                    all_targets.append(p)
        elif src.is_file() and (
            src.suffix.lower() in {".csv", ".tsv", ".xlsx", ".xls", ".jsonl"}
            or src.name.lower() == "smsspamcollection"
        ):
            all_targets.append(src)
    for d in extracted_dirs:
        for p in d.rglob("*"):
            if p.is_file() and (
                p.suffix.lower() in {".csv", ".tsv", ".xlsx", ".xls", ".jsonl"}
                or p.name.lower() == "smsspamcollection"
            ):
                all_targets.append(p)

    # تفرّد
    uniq_targets, seen = [], set()
    for p in all_targets:
        r = str(p.resolve())
        if r not in seen:
            uniq_targets.append(p)
            seen.add(r)

    if not uniq_targets:
        print(
            "⚠️ No readable dataset files were found. Put your data in BASE_DIR or zip/rar it there."
        )
        return {
            "base": base,
            "script_dir": script_dir,
            "extracted_root": extracted_root,
            "train_csv": train_csv,
            "val_csv": val_csv,
            "test_csv": test_csv,
        }

    print(f"🔎 Found {len(uniq_targets)} candidate files to ingest.")

    # اقرأ الكل + نزّل أعمدة مكررة داخل كل DF
    frames: List[pd.DataFrame] = []
    for p in uniq_targets:
        df = _read_any_file(p)
        if df is None or df.empty:
            continue
        frames.append(df)

    if not frames:
        print("⚠️ Nothing could be read after standardization.")
        return {
            "base": base,
            "script_dir": script_dir,
            "extracted_root": extracted_root,
            "train_csv": train_csv,
            "val_csv": val_csv,
            "test_csv": test_csv,
        }

    # ✨ هنا أصل المشكلة: لازم نتأكد إن “أسماء الأعمدة فريدة” قبل concat
    frames = [_dedupe_columns(f) for f in frames]
    df_all = pd.concat(frames, ignore_index=True, sort=False)

    # تنظيف خفيف
    if "text" in df_all.columns:
        df_all["text"] = (
            df_all["text"]
            .astype(str)
            .fillna("")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    if "label" not in df_all.columns:
        raise ValueError("No 'label' column found in any source.")
    lab = df_all["label"].astype(str).str.strip().str.lower()
    direct_map = {
        "spam": 1,
        "phishing": 1,
        "malicious": 1,
        "fraud": 1,
        "attack": 1,
        "bad": 1,
        "phishing email": 1,
        "spam email": 1,
        "1": 1,
        "true": 1,
        "yes": 1,
        "ham": 0,
        "legitimate": 0,
        "benign": 0,
        "good": 0,
        "normal": 0,
        "legitimate email": 0,
        "not spam": 0,
        "0": 0,
        "false": 0,
        "no": 0,
    }
    lab_mapped = lab.map(direct_map)
    is_phish = lab_mapped.isna() & lab.str.contains(
        r"(phish|spam|malicious|fraud|attack|scam|fake|decept)", regex=True
    )
    is_legit = lab_mapped.isna() & lab.str.contains(
        r"(legit|ham|benign|safe|normal|clean|valid)", regex=True
    )
    lab_mapped = np.where(is_phish, 1, np.where(is_legit, 0, lab_mapped)).astype(
        "float"
    )
    df_all["label"] = pd.to_numeric(lab_mapped, errors="coerce")

    before = len(df_all)
    df_all = df_all.dropna(subset=["label"]).copy()
    df_all["label"] = df_all["label"].astype(int)
    dropped = before - len(df_all)
    if dropped > 0:
        print(f"🧹 Dropped {dropped} rows with unknown labels.")

    if "text" not in df_all.columns and "url" not in df_all.columns:
        raise ValueError("Dataset must contain at least 'text' or 'url' column.")
    if "text" in df_all.columns:
        df_all = df_all[df_all["text"].astype(str).str.len() > 0].copy()
    dup_subset = [c for c in ["text", "url"] if c in df_all.columns]
    if dup_subset:
        df_all = df_all.drop_duplicates(subset=dup_subset).reset_index(drop=True)

    if "timestamp" not in df_all.columns:
        df_all["timestamp"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(
            np.arange(len(df_all)), unit="h"
        )

    # تقسيم 70/15/15
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df_all, test_size=0.30, stratify=df_all["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    _save_ready_splits(train_df, val_df, test_df, base)
    print(f"✅ Saved train/val/test CSVs into: {base_dir}")

    return {
        "base": base,
        "script_dir": script_dir,
        "extracted_root": extracted_root,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "test_csv": test_csv,
    }


from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ======================================================
# Config
# ======================================================


@dataclass
class ModelConfig:
    """Configuration for model training and evaluation."""

    # Operating point / cost
    target_fpr: float = 0.003  # 0.3% FPR
    cost_fp: float = 1.0
    cost_fn: float = 10.0

    # CV / evaluation
    early_stopping_rounds: int = 50
    cv_folds: int = 5
    time_series_cv: bool = True
    group_by_column: str = "sender_domain"
    calibration_method: str = "isotonic"  # 'isotonic' or 'sigmoid'
    min_recall_threshold: float = 0.80
    drift_threshold: float = 0.10  # PSI threshold


# ======================================================
# Part 1: Advanced Data Cleaning & Preprocessing
# ======================================================


class DataPreprocessor:
    """
    Advanced data preprocessing with:
      - duplicate removal
      - safe feature engineering (URL/Text)
      - low-variance pruning (recorded on train)
      - robust outlier clipping (IQR, recorded on train)
      - log-transform for skewed features
      - missing-value imputation (recorded on train)
      - scaling + strict column alignment across splits
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.scaler = StandardScaler()

        # كل ما يلزم لإعادة تطبيق نفس التحويلات على val/test والإنتاج
        self.feature_stats: Dict[str, Any] = {
            "removed_low_var": [],  # List[str]
            "feature_order": None,  # List[str]
            "impute_values": {},  # Dict[str, float]
            # إضافةً إلى حدود الـ IQR لكل عمود رقمي: f"{col}_bounds": (lower, upper)
        }

    # ---------------------- public API ----------------------

    def clean_data(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        print("=" * 100)
        print("🧹 Advanced Data Cleaning")
        print("=" * 100)

        print(f"📊 Initial shape: {df.shape}")

        # 1) إزالة التطابقات التامة
        before = len(df)
        df = df.drop_duplicates()
        print(f"✅ Removed {before - len(df)} exact duplicates")

        # 2) ميزات زمنية
        if "timestamp" in df.columns:
            df = self._add_temporal_features(df)

        # 3) مجموعات المُرسل (تجميع الحملات)
        if ("sender" in df.columns) or ("from_address" in df.columns):
            df = self._create_sender_groups(df)

        # 4) هندسة ميزات آمنة
        df = self._engineer_features(df)

        # 5) إزالة الميزات ضعيفة التباين
        if is_train:
            df, removed = self._remove_low_variance(df, threshold=0.01)
            self.feature_stats["removed_low_var"] = removed
        else:
            removed = self.feature_stats.get("removed_low_var", [])
            if removed:
                df = df.drop(columns=removed, errors="ignore")

        # 6) التعامل مع القيم الشاذة (IQR)
        df = self._handle_outliers(df, is_train=is_train)

        # 7) التحويل اللوغاريتمي للمنحرفة
        df = self._log_transform_skewed(df, skew_threshold=1.0)

        # 8) تعويض القيم المفقودة
        df = self._impute_missing(df, is_train=is_train)

        # 9) التحجيم ومحاذاة الأعمدة الرقمية
        if is_train:
            num_cols = self._numeric_cols(df)
            self.feature_stats["feature_order"] = list(num_cols)
            if len(num_cols) > 0:
                df[num_cols] = self.scaler.fit_transform(df[num_cols])
        else:
            train_order: Optional[List[str]] = self.feature_stats.get("feature_order")
            if train_order is None:
                train_order = list(self._numeric_cols(df))  # fallback الآمن

            # أضف أي عمود رقمي ناقص بقيمة 0.0
            for c in train_order:
                if c not in df.columns:
                    df[c] = 0.0

            # احذف الأعمدة الرقمية الزائدة غير الموجودة في تدريب
            extra = [c for c in self._numeric_cols(df) if c not in train_order]
            if extra:
                df = df.drop(columns=extra, errors="ignore")

            if len(train_order) > 0:
                df[train_order] = self.scaler.transform(df[train_order])

        print(f"✅ Final shape: {df.shape}\n")
        return df

    # ---------------------- helpers ----------------------

    @staticmethod
    def _numeric_cols(df: pd.DataFrame) -> pd.Index:
        # نحافظ على label خارج التحجيم دائمًا
        return df.select_dtypes(include=[np.number, bool]).columns.drop(
            "label", errors="ignore"
        )

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.copy()
        df["timestamp"] = ts
        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)
        return df

    def _create_sender_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        sender_col = "sender" if "sender" in df.columns else "from_address"
        vals = df[sender_col].fillna("unknown").astype(str)

        # استخراج الدومين بشكل آمن
        def _domain(x: str) -> str:
            x = x.strip()
            if "@" in x:
                part = x.split("@", 1)[1]
                return part.lower() or "unknown"
            return "unknown"

        df["sender_domain"] = vals.apply(_domain)
        # factorize يُنتج mapping ثابتًا بالترتيب
        df["campaign_id"] = pd.factorize(df["sender_domain"])[0]
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # URL features
        if "url" in df.columns:
            url = df["url"].fillna("").astype(str)
            df["has_ip_in_url"] = url.str.contains(
                r"\d+\.\d+\.\d+\.\d+", regex=True
            ).astype(int)
            df["url_entropy"] = url.apply(self._calculate_entropy)
            df["has_homograph"] = url.apply(self._detect_homograph)
            df["has_url_shortener"] = url.str.contains(
                r"(bit\.ly|goo\.gl|tinyurl|short\.link|ow\.ly)", regex=True, case=False
            ).astype(int)
            df["url_length"] = url.str.len()
            df["num_dots"] = url.str.count(r"\.")
            df["num_digits_in_url"] = url.str.count(r"\d")

        # Text features (body > text)
        txt_col = (
            "body"
            if "body" in df.columns
            else ("text" if "text" in df.columns else None)
        )
        if txt_col:
            t = df[txt_col].fillna("").astype(str)
            df["urgency_score"] = t.apply(self._calculate_urgency_score)
            df["grammar_errors"] = t.apply(self._estimate_grammar_errors)
            df["personalization_score"] = t.apply(self._calculate_personalization)
            df["char_count"] = t.str.len()

        return df

    @staticmethod
    def _calculate_entropy(text: str) -> float:
        if not text:
            return 0.0
        s = str(text)
        n = max(len(s), 1)
        # احتمال كل رمز
        probs = [s.count(c) / n for c in set(s)]
        return float(-sum(p * np.log2(p) for p in probs if p > 0))

    @staticmethod
    def _detect_homograph(url: str) -> int:
        # كشف مبسّط لوجود رموز/أشكال بديلة قد تستغل للتضليل
        homographs = {
            "0": "o",
            "1": "l",
            "@": "a",
            "3": "e",
            "5": "s",
            "!": "i",
            "4": "a",
        }
        u = str(url).lower()
        for ch in homographs:
            if ch in u:
                return 1
        return 0

    @staticmethod
    def _calculate_urgency_score(text: str) -> float:
        terms = [
            # EN
            "urgent",
            "immediate",
            "expire",
            "suspend",
            "deadline",
            "limited time",
            "act now",
            "hurry",
            "quick",
            "fast",
            # AR
            "عاجل",
            "هام",
            "مهم",
            "فوري",
            "تنبيه",
            "انتهت",
            "انتهاء",
            "مطلوب فوراً",
            "تحقق الآن",
            "تحديث",
            "تأكيد",
            "استعادة",
            "حسابك",
            "كلمة المرور",
        ]
        t = str(text).lower()
        score = sum(1 for w in terms if w in t)
        return float(min(score / max(len(terms), 1), 1.0))

    @staticmethod
    def _estimate_grammar_errors(text: str) -> int:
        t = str(text)
        errs = 0
        errs += len(re.findall(r"\s{2,}", t))  # مسافات متتالية
        errs += len(re.findall(r"[.!?]{2,}", t))  # علامات ترقيم متتالية
        errs += len(re.findall(r"\b[A-Z]{3,}\b", t))  # كلمات CAPITAL طويلة
        return int(min(errs, 10))

    @staticmethod
    def _calculate_personalization(text: str) -> float:
        pats = [
            r"dear\s+\w+",
            r"hello\s+\w+",
            r"hi\s+\w+",
            r"your account",
            r"your password",
            r"your payment",
        ]
        t = str(text).lower()
        score = sum(1 for p in pats if re.search(p, t))
        return float(min(score / max(len(pats), 1), 1.0))

    def _remove_low_variance(
        self, df: pd.DataFrame, threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, List[str]]:
        num_cols = self._numeric_cols(df)
        removed: List[str] = []
        for c in num_cols:
            v = float(pd.to_numeric(df[c], errors="coerce").var())  # ddof=1
            if not np.isfinite(v) or v < threshold:
                removed.append(c)
        if removed:
            df = df.drop(columns=removed, errors="ignore")
            print(f"🧹 Removed {len(removed)} low-variance features")
        return df, removed

    def _handle_outliers(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        df = df.copy()
        for c in self._numeric_cols(df):
            key = f"{c}_bounds"
            s = pd.to_numeric(df[c], errors="coerce")
            if is_train:
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                if not np.isfinite(q1) or not np.isfinite(q3) or q1 == q3:
                    # عمود ثابت أو كله NaN → لا نحفظ حدود
                    continue
                iqr = q3 - q1
                lower, upper = q1 - 3.0 * iqr, q3 + 3.0 * iqr
                self.feature_stats[key] = (float(lower), float(upper))
                df[c] = s.clip(lower, upper)
            else:
                if key in self.feature_stats:
                    lower, upper = self.feature_stats[key]
                    df[c] = s.clip(lower, upper)
        return df

    def _log_transform_skewed(
        self, df: pd.DataFrame, skew_threshold: float = 1.0
    ) -> pd.DataFrame:
        df = df.copy()
        for c in self._numeric_cols(df):
            s = pd.to_numeric(df[c], errors="coerce")
            try:
                sk = float(s.skew())
            except Exception:
                continue
            if np.isfinite(sk) and abs(sk) > skew_threshold:
                # shift-to-positive ثم log1p
                mn = np.nanmin(s.values) if np.size(s.values) else 0.0
                s = np.log1p(s - (mn if np.isfinite(mn) else 0.0) + 1.0)
                df[c] = s
        return df

    def _impute_missing(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        df = df.copy()
        num_cols = self._numeric_cols(df)
        if is_train:
            impute_values: Dict[str, float] = {}
            for c in num_cols:
                s = pd.to_numeric(df[c], errors="coerce")
                med = float(np.nanmedian(s.values)) if s.notna().any() else 0.0
                if not np.isfinite(med):
                    med = 0.0
                impute_values[c] = med
                if s.isnull().any():
                    df[c] = s.fillna(med)
            self.feature_stats["impute_values"] = impute_values
        else:
            impute_values = self.feature_stats.get("impute_values", {})
            for c in num_cols:
                fill_value = impute_values.get(c, 0.0)
                s = pd.to_numeric(df.get(c, 0.0), errors="coerce")
                df[c] = s.fillna(0.0 if not np.isfinite(fill_value) else fill_value)
        return df


# ================================================================================================
# Part 2: Advanced Leakage Detection
# ================================================================================================


class LeakageDetector:
    """
    Advanced data leakage detection using multiple signals:
      1) Exact matches across splits (ignoring 'label')
      2) Near-duplicates via char TF-IDF + cosine (aggressive but safe)
      3) Campaign/sender separation (by sender_domain)
      4) Temporal ordering (train < val < test), with safety floor

    Aggressiveness and toggles via env vars:
      - DISABLE_NEAR_DUP=1       → skip near-duplicate removal
      - DISABLE_TEMPORAL=1       → skip temporal order enforcement
      - NEAR_DUP_SIM=0.975       → cosine similarity threshold
      - NEAR_DUP_MIN_DF=2        → TF-IDF min_df (fallback to 1 if empty vocab)
      - NEAR_DUP_BUCKET=80       → candidate cap per query row
      - NEAR_DUP_COLS="text,body,subject,url" → columns to join (optional)
    """

    def __init__(self, similarity_threshold: float = 0.975, rp_bits: int = 16):
        self.similarity_threshold = float(
            os.getenv("NEAR_DUP_SIM", similarity_threshold)
        )
        self.rp_bits = int(rp_bits)  # reserved for future RP hashing if needed

    # ------------------------- public API -------------------------

    def detect_and_remove_leakage(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("=" * 100)
        print("🔒 Advanced Leakage Detection")
        print("=" * 100)

        # 1) Exact matches (remove from val/test only)
        train_df, val_df, test_df = self._remove_exact_matches(
            train_df, val_df, test_df
        )

        # 2) Near-duplicates (char TF-IDF, cosine) — optional
        disable_near = str(os.getenv("DISABLE_NEAR_DUP", "0")).lower() in (
            "1",
            "true",
            "yes",
        )
        if disable_near:
            print("ℹ️ Skipping near-duplicate removal (DISABLE_NEAR_DUP=1).")
        else:
            if min(len(val_df), len(test_df)) >= 50 and len(train_df) >= 100:
                train_df, val_df, test_df = self._remove_near_duplicates(
                    train_df, val_df, test_df
                )
            else:
                print("ℹ️ Skipping near-duplicates (splits too small).")

        # 3) Campaign separation (sender_domain) — optional if column exists
        if "sender_domain" in train_df.columns:
            train_df, val_df, test_df = self._ensure_campaign_separation(
                train_df, val_df, test_df
            )

        # 4) Temporal order — optional
        if "timestamp" in train_df.columns:
            train_df, val_df, test_df = self._ensure_temporal_order(
                train_df, val_df, test_df
            )

        print(f"\n✅ Final shapes after leakage removal:")
        print(f"   Train: {train_df.shape}")
        print(f"   Val:   {val_df.shape}")
        print(f"   Test:  {test_df.shape}\n")

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    # ------------------------- helpers -------------------------

    @staticmethod
    def _row_md5(x: pd.Series) -> str:
        # Hash “content” not identity; ignore label
        return hashlib.md5(str(tuple(x.values)).encode("utf-8")).hexdigest()

    def _remove_exact_matches(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Remove exact duplicate rows across splits (ignoring 'label')."""
        # Use a safe view without label
        Xtr = train_df.drop(columns=["label"], errors="ignore")
        Xva = val_df.drop(columns=["label"], errors="ignore")
        Xte = test_df.drop(columns=["label"], errors="ignore")

        tr_hash = Xtr.apply(self._row_md5, axis=1)
        va_hash = Xva.apply(self._row_md5, axis=1)
        te_hash = Xte.apply(self._row_md5, axis=1)

        val_overlap = va_hash.isin(tr_hash)
        test_overlap = te_hash.isin(tr_hash) | te_hash.isin(va_hash)

        rm_va = int(val_overlap.sum())
        rm_te = int(test_overlap.sum())
        if rm_va:
            print(f"🔍 Removed {rm_va} exact matches from validation")
            val_df = val_df.loc[~val_overlap]
        if rm_te:
            print(f"🔍 Removed {rm_te} exact matches from test")
            test_df = test_df.loc[~test_overlap]

        return train_df, val_df, test_df

    def _remove_near_duplicates(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Removes VAL/TEST rows that are near-duplicates of TRAIN using char TF-IDF + cosine.
        - Numerically stable (NaN/inf handled)
        - Efficient candidate pruning via norm buckets
        - Returns filtered val/test only (train left intact)
        """
        import numpy as _np
        import scipy.sparse as _sp
        from sklearn.feature_extraction.text import TfidfVectorizer as _TF
        from sklearn.metrics.pairwise import cosine_similarity as _cos

        # Which text columns to join
        cols_env = os.getenv("NEAR_DUP_COLS", "").strip()
        if cols_env:
            prefer_cols = [c.strip() for c in cols_env.split(",") if c.strip()]
        else:
            prefer_cols = ["text", "body", "subject", "url", "message", "content"]

        def _join_text(df: pd.DataFrame) -> pd.Series:
            parts = []
            cols = set(df.columns)
            for c in prefer_cols:
                if c in cols:
                    parts.append(df[c].astype(str).fillna(""))
            if not parts:
                return pd.Series([""] * len(df), index=df.index)
            s = parts[0]
            for p in parts[1:]:
                s = s.str.cat(p, sep=" ")
            return s.fillna("").str.strip()

        tr_txt = _join_text(train_df)
        va_txt = _join_text(val_df)
        te_txt = _join_text(test_df)

        # If all empty, nothing to do
        if (tr_txt.str.len().sum() == 0) or (
            (va_txt.str.len().sum() == 0) and (te_txt.str.len().sum() == 0)
        ):
            print("ℹ️ Near-duplicate step skipped (no textual content).")
            return train_df, val_df, test_df

        min_df = int(os.getenv("NEAR_DUP_MIN_DF", "2"))
        vec = _TF(analyzer="char", ngram_range=(3, 6), min_df=min_df, dtype=_np.float32)
        Xtr = vec.fit_transform(tr_txt)
        Xva = vec.transform(va_txt)
        Xte = vec.transform(te_txt)

        # Fallback in case vocabulary was empty
        if Xtr.shape[1] == 0:
            vec = _TF(analyzer="char", ngram_range=(3, 6), min_df=1, dtype=_np.float32)
            Xtr = vec.fit_transform(tr_txt)
            Xva = vec.transform(va_txt)
            Xte = vec.transform(te_txt)

        def _clean(M):
            if _sp.issparse(M):
                M.data = _np.nan_to_num(M.data, nan=0.0, posinf=0.0, neginf=0.0)
                return M
            A = _np.asarray(M, dtype=_np.float32)
            return _np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

        Xtr = _clean(Xtr)
        Xva = _clean(Xva)
        Xte = _clean(Xte)

        # Candidate pruning by L2-norm “buckets”
        def _bucket_keys(X):
            norms = _np.sqrt((X.multiply(X)).sum(axis=1)).A1
            norms = _np.nan_to_num(norms, nan=0.0, posinf=0.0, neginf=0.0)
            return (norms * 100).astype(int)

        tr_keys = _bucket_keys(Xtr)
        buckets: Dict[int, List[int]] = {}
        for idx, key in enumerate(tr_keys):
            buckets.setdefault(int(key), []).append(idx)

        bucket_cap = int(os.getenv("NEAR_DUP_BUCKET", "80"))
        sim_thr = float(self.similarity_threshold)

        def _mask_keep(Xc):
            keep = _np.ones(Xc.shape[0], dtype=bool)
            c_keys = _bucket_keys(Xc)
            for i in range(Xc.shape[0]):
                k = int(c_keys[i])
                cand = []
                for kk in (k - 1, k, k + 1):  # neighbor buckets
                    cand.extend(buckets.get(kk, []))
                if not cand:
                    continue
                if len(cand) > bucket_cap:
                    cand = cand[:bucket_cap]
                sims = _cos(Xc[i : i + 1], Xtr[cand]).ravel()
                if _np.any(sims >= sim_thr):
                    keep[i] = False
            return keep

        va_keep = _mask_keep(Xva) if Xva.shape[0] else _np.array([], dtype=bool)
        te_keep = _mask_keep(Xte) if Xte.shape[0] else _np.array([], dtype=bool)

        rm_va = int((~va_keep).sum()) if va_keep.size else 0
        rm_te = int((~te_keep).sum()) if te_keep.size else 0
        if rm_va:
            print(f"🔍 Removed {rm_va} near-duplicates from validation")
        if rm_te:
            print(f"🔍 Removed {rm_te} near-duplicates from test")

        val_df_f = val_df.loc[va_keep] if va_keep.size else val_df
        test_df_f = test_df.loc[te_keep] if te_keep.size else test_df
        return train_df, val_df_f, test_df_f

    def _ensure_campaign_separation(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Ensure campaigns/senders don't overlap across splits."""
        tr = set(train_df["sender_domain"].astype(str).unique())
        va = set(val_df["sender_domain"].astype(str).unique())
        te = set(test_df["sender_domain"].astype(str).unique())

        val_overlap = val_df["sender_domain"].astype(str).isin(tr)
        test_overlap = test_df["sender_domain"].astype(str).isin(tr | va)

        rm_va = int(val_overlap.sum())
        rm_te = int(test_overlap.sum())
        if rm_va:
            print(f"🔍 Removed {rm_va} validation samples from overlapping campaigns")
            val_df = val_df.loc[~val_overlap]
        if rm_te:
            print(f"🔍 Removed {rm_te} test samples from overlapping campaigns")
            test_df = test_df.loc[~test_overlap]

        return train_df, val_df, test_df

    def _ensure_temporal_order(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Enforce train < val < test (by timestamp) with safety floors.
        - Skip if DISABLE_TEMPORAL=1
        - If filtering would shrink val/test below minimum, skip and print ranges
        """
        if str(os.getenv("DISABLE_TEMPORAL", "0")).lower() in ("1", "true", "yes"):
            print("ℹ️ Skipping temporal order enforcement (DISABLE_TEMPORAL=1).")
            return train_df, val_df, test_df

        for df in (train_df, val_df, test_df):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        train_max = train_df["timestamp"].max()
        val_min = val_df["timestamp"].min() if len(val_df) else None
        val_max = val_df["timestamp"].max() if len(val_df) else None
        test_min = test_df["timestamp"].min() if len(test_df) else None

        # Filter
        val_f = val_df[val_df["timestamp"] > train_max] if len(val_df) else val_df
        test_f = test_df
        if len(val_f):
            new_val_max = val_f["timestamp"].max()
            test_f = test_df[test_df["timestamp"] > new_val_max]

        # Safety: don't empty splits
        min_val, min_test = 100, 100
        if (len(val_f) < min_val) or (len(test_f) < min_test):
            print(
                f"📅 Temporal order would shrink val/test too much (val:{len(val_f)} test:{len(test_f)}). Skipping temporal enforcement."
            )
            print(f"   Train: up to {train_max}")
            if len(val_df):
                print(f"   Val:   {val_min} → {val_max}  (no filtering)")
            if len(test_df):
                print(f"   Test:  from {test_min}       (no filtering)")
            return train_df, val_df, test_df

        print("📅 Temporal order enforced:")
        print(f"   Train: up to {train_max}")
        if len(val_f):
            print(f"   Val:   {val_f['timestamp'].min()} → {val_f['timestamp'].max()}")
        if len(test_f):
            print(f"   Test:  from {test_f['timestamp'].min()}")

        return train_df, val_f, test_f


# ================================================================================================
# Part 3: Advanced Model Training with All Improvements (PROD-GRADE v2)
# ================================================================================================
import os
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold,
    GroupKFold,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


# استيرادات اختيارية (لن تتعطل لو المكتبات مش مثبتة)
try:
    from xgboost import XGBClassifier as _XGBClassifier  # type: ignore
except Exception:
    _XGBClassifier = None

try:
    import lightgbm as lgb  # type: ignore
    _LGBMClassifier = lgb.LGBMClassifier
except Exception:
    lgb = None
    _LGBMClassifier = None

try:
    from catboost import CatBoostClassifier as _CatBoostClassifier  # type: ignore
except Exception:
    _CatBoostClassifier = None


class AdvancedModelTrainer:
    """Advanced model training with production features:
       - اسماء ميزات معقّمة وثابتة
       - CV زمني/حسب المجموعات/طبقي حسب الحاجة
       - معايرة احتمالات على مجموعة التحقق
       - اختيار عتبة يحترم قيود FPR + كلفة FP/FN
       - دعم XGBoost 2.x (EarlyStopping عبر callbacks) وGPU
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._feature_name_map: Dict[str, str] = {}
        self.models = self._initialize_models()
        self.calibrated_models: Dict[str, Any] = {}
        self.best_thresholds: Dict[str, float] = {}
        self.feature_importances: Dict[str, Any] = {}

        tree_types = [RandomForestClassifier]
        if _XGBClassifier is not None:
            tree_types.append(_XGBClassifier)
        if _LGBMClassifier is not None:
            tree_types.append(_LGBMClassifier)
        if _CatBoostClassifier is not None:
            tree_types.append(_CatBoostClassifier)
        self._tree_types = tuple(tree_types)

    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize basic models - called only when USE_TFIDF=0"""
        models = {}
        
        # Logistic Regression
        models["LogisticRegression"] = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        
        # Random Forest
        models["RandomForest"] = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        # XGBoost (if available)
        if _XGBClassifier is not None:
            models["XGBoost"] = _XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        
        # LightGBM (if available)
        if _LGBMClassifier is not None:
            models["LightGBM"] = _LGBMClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        
        return models

    # ------------------------- feature name hygiene -------------------------
    def _sanitize_feature_names_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        """تعقيم أسماء الميزات أثناء التدريب وتخزين الخريطة."""
        mapping: Dict[str, str] = {}
        seen = set()
        new_cols: List[str] = []

        for c in X.columns:
            s = re.sub(r"[^\w]+", "_", str(c))
            s = re.sub(r"_+", "_", s).strip("_")
            if not s or s[0].isdigit():
                s = f"f_{s}"
            base = s
            i = 2
            while s in seen:
                s = f"{base}_{i}"
                i += 1
            seen.add(s)
            mapping[c] = s
            new_cols.append(s)

        X2 = X.copy()
        X2.columns = new_cols
        self._feature_name_map = mapping
        return X2

    def _sanitize_feature_names_apply(self, X: pd.DataFrame) -> pd.DataFrame:
        """تطبيق نفس خريطة الأسماء على val/test مع ملء الأعمدة المفقودة وترتيبها."""
        if not self._feature_name_map:
            return X.copy()
        X2 = X.rename(columns=self._feature_name_map).copy()
        wanted = list(self._feature_name_map.values())
        for col in wanted:
            if col not in X2.columns:
                X2[col] = 0.0
        return X2.reindex(columns=wanted, fill_value=0.0)

    # ---------------------------- utils ----------------------------
    def _num_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """إرجاع الأعمدة الرقمية/المنطقية فقط (بدون label)."""
        if df is None or len(df) == 0:
            return df
        X = df.select_dtypes(include=[np.number, bool]).copy()
        if "label" in X.columns:
            X = X.drop(columns=["label"])
        return X.astype(float)

    def _fit_with_optional_eval(self, model, X_tr, y_tr, X_va=None, y_va=None):
        """تدريب آمن مع دعم EarlyStopping لـ XGBoost 2.x عبر callbacks - FIXED."""
        fit_kwargs = {}
        
        # XGBoost 2.x: لا نمرر early_stopping_rounds مباشرة → callbacks
        es_rounds = int(os.getenv("XGB_ES", str(self.config.early_stopping_rounds)))
        try:
            from xgboost import XGBClassifier as _XGB
            try:
                from xgboost.callback import EarlyStopping  # type: ignore
            except Exception:
                EarlyStopping = None  # noqa
        except Exception:
            _XGB = None
            EarlyStopping = None  # noqa

        # ✅ فحص: هل النموذج يقبل eval_set؟
        model_accepts_eval_set = False
        model_type_name = type(model).__name__
        
        # Check XGBoost
        if _XGB is not None and isinstance(model, _XGB):
            model_accepts_eval_set = True
        
        # Check LightGBM
        if 'LGB' in model_type_name or 'LightGBM' in model_type_name:
            try:
                import lightgbm as lgb
                if isinstance(model, lgb.LGBMClassifier):
                    model_accepts_eval_set = True
            except:
                pass
        
        # Check CatBoost
        if 'CatBoost' in model_type_name:
            try:
                from catboost import CatBoostClassifier
                if isinstance(model, CatBoostClassifier):
                    model_accepts_eval_set = True
            except:
                pass
        
        # ✅ أضف eval_set فقط إذا النموذج يقبلها
        if X_va is not None and y_va is not None and model_accepts_eval_set:
            fit_kwargs["eval_set"] = [(X_va, y_va)]

        if _XGB is not None and isinstance(model, _XGB):
            cbs = []
            if es_rounds > 0 and EarlyStopping is not None:
                cbs.append(EarlyStopping(rounds=es_rounds, maximize=True, save_best=True))
            fit_kwargs["verbose"] = False
            fit_kwargs["callbacks"] = cbs
        else:
            # لبعض النماذج (مثل LightGBM) التي تدعم early_stopping_rounds اسماً
            if hasattr(model, "fit") and "early_stopping_rounds" in model.fit.__code__.co_varnames:
                fit_kwargs["early_stopping_rounds"] = es_rounds

        return model.fit(X_tr, y_tr, **fit_kwargs)

    # ---------------------------- main API ----------------------------
    def train_with_temporal_cv(self, X_train, y_train, X_val, y_val) -> pd.DataFrame:
        """تدريب كل النماذج + CV آمن + معايرة + اختيار عتبة."""
        print("=" * 100)
        print("🚀 Advanced Model Training")
        print("=" * 100)

        # 1) ميزات رقمية فقط + تنظيف
        X_train_num = self._num_only(X_train).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_val_num   = self._num_only(X_val).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 2) تعقيم أسماء الميزات
        X_train_num = self._sanitize_feature_names_fit(X_train_num)
        X_val_num   = self._sanitize_feature_names_apply(X_val_num)

        results: List[Dict[str, Any]] = []

        # 3) اختيار نوع الـCV
        use_temporal = (
            getattr(self.config, "time_series_cv", False)
            and ("timestamp" in X_train.columns)
            and os.getenv("DISABLE_TEMPORAL", "0") != "1"
        )
        if use_temporal:
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            groups = None
            print("Using TimeSeriesSplit CV")
        elif "sender_domain" in X_train.columns:
            cv = GroupKFold(n_splits=self.config.cv_folds)
            groups = X_train["sender_domain"]
            print("Using GroupKFold CV")
        else:
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
            groups = None
            print("Using StratifiedKFold CV")

        # 4) حلقة التدريب/المعايرة/التقييم
        for name, model in self.models.items():
            print(f"\n{'='*50}\n📊 Training: {name}\n{'='*50}")

            # تدريب بمجموع التدريب + Val للإيقاف المبكر حيث ينطبق
            model = self._fit_with_optional_eval(model, X_train_num, y_train, X_val_num, y_val)

            # نسخة للـCV: لو XGB عطّل ES داخل الـCV
            model_for_cv = model
            try:
                if _XGBClassifier is not None and isinstance(model, _XGBClassifier):
                    model_for_cv = clone(model)
                    if hasattr(model_for_cv, "set_params"):
                        model_for_cv.set_params(early_stopping_rounds=None)
            except Exception:
                model_for_cv = model

            # تنفيذ CV آمن
            try:
                cv_kwargs = dict(cv=cv, scoring="roc_auc", n_jobs=1)
                if groups is not None:
                    cv_scores = cross_val_score(model_for_cv, X_train_num, y_train, groups=groups, **cv_kwargs)
                else:
                    cv_scores = cross_val_score(model_for_cv, X_train_num, y_train, **cv_kwargs)
            except Exception as e:
                print(f"⚠️ CV failed for {name}: {e}\n   Falling back to validation only.")
                try:
                    tmp_cal = self._calibrate_model(model, X_train_num, y_train)
                    val_auc = roc_auc_score(y_val, tmp_cal.predict_proba(X_val_num)[:, 1])
                    cv_scores = np.array([val_auc])
                except Exception:
                    cv_scores = np.array([0.0])

            # معايرة على الـ VAL (cv='prefit')
            calibrated_model = self._calibrate_model_on_val(model, X_train_num, y_train, X_val_num, y_val)
            self.calibrated_models[name] = calibrated_model

            # احتمالات + اختيار العتبة
            y_proba = np.clip(calibrated_model.predict_proba(X_val_num)[:, 1], 1e-9, 1 - 1e-9)
            optimal_threshold = self._find_optimal_threshold(y_val, y_proba)
            self.best_thresholds[name] = float(optimal_threshold)

            # متريكس
            y_pred = (y_proba >= optimal_threshold).astype(int)
            metrics = self._calculate_comprehensive_metrics(y_val, y_pred, y_proba)
            metrics["Model"] = name
            metrics["CV_Mean"] = float(np.mean(cv_scores))
            metrics["CV_Std"] = float(np.std(cv_scores))
            metrics["Optimal_Threshold"] = float(optimal_threshold)
            results.append(metrics)

            # أهميات الميزات (إن وجدت)
            self._maybe_collect_feature_importances(name, model, X_train_num.columns)

            print(
                f"ROC-AUC: {metrics['ROC-AUC']:.4f} | PR-AUC: {metrics['PR-AUC']:.4f} "
                f"| Recall@FPR≤{self.config.target_fpr}: {metrics['Recall_at_FPR']:.4f}"
            )
            print(
                f"Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f} "
                f"| F1: {metrics['F1']:.4f} | Brier: {metrics['Brier_Score']:.4f}"
            )
            print(
                f"Optimal Threshold: {metrics['Optimal_Threshold']:.3f} "
                f"| CV: {metrics['CV_Mean']:.4f}±{metrics['CV_Std']:.4f}"
            )

        return pd.DataFrame(results)

    # ----------------------- calibration helpers -----------------------
    def _calibrate_model(self, estimator, X_train, y_train):
        """تقسيم 10% من train للمعايرة؛ متوافق مع الإصدارات."""
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        tr_idx, cal_idx = next(sss.split(X_train, y_train))
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_cal, y_cal = X_train.iloc[cal_idx], y_train.iloc[cal_idx]
        estimator.fit(X_tr, y_tr)
        return self._calibrate_model_on_val(estimator, X_tr, y_tr, X_cal, y_cal)

    def _calibrate_model_on_val(self, estimator, X_train, y_train, X_val, y_val):
        """CalibratedClassifierCV مع cv='prefit' (يدعم estimator/base_estimator)."""
        if not hasattr(estimator, "classes_"):
            estimator.fit(X_train, y_train)
        method = self._calibration_method_for(estimator)
        try:
            calibrator = CalibratedClassifierCV(estimator=estimator, method=method, cv="prefit")
        except TypeError:
            calibrator = CalibratedClassifierCV(base_estimator=estimator, method=method, cv="prefit")
        calibrator.fit(X_val, y_val)
        return calibrator

    # ---------------------- threshold & metrics -----------------------
    def _find_optimal_threshold(self, y_true, y_proba):
        """اختيار عتبة تحترم FPR المستهدف؛ والعودة لعتبة الكلفة لو الـRecall منخفض."""
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_proba = np.asarray(y_proba, dtype=float).ravel()

        # قيد FPR
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        valid_idx = np.where(fpr <= self.config.target_fpr)[0]
        if len(valid_idx):
            thr_fpr = float(thresholds[valid_idx[np.argmax(tpr[valid_idx])]])
        else:
            thr_fpr = 0.5  # fallback

        # بحث حسّاس للكلفة
        grid = np.linspace(0.01, 0.99, 99)
        costs = []
        for thr in grid:
            y_pred = (y_proba >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            costs.append(fp * self.config.cost_fp + fn * self.config.cost_fn)
        thr_cost = float(grid[int(np.argmin(costs))])

        # حارس Recall عند عتبة FPR
        recall_fpr = recall_score(y_true, (y_proba >= thr_fpr).astype(int), zero_division=0)
        return float(thr_fpr) if recall_fpr >= self.config.min_recall_threshold else float(thr_cost)

    def _calculate_comprehensive_metrics(self, y_true, y_pred, y_proba):
        """حزمة متريكس شاملة + Recall@FPR + Brier + LogLoss."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr_curve, tpr_curve, thr = roc_curve(y_true, y_proba)
        idx = np.where(fpr_curve <= self.config.target_fpr)[0]
        recall_at_fpr = float(tpr_curve[idx[-1]]) if len(idx) > 0 else 0.0

        y_proba_safe = np.clip(y_proba, 1e-9, 1 - 1e-9)
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "ROC-AUC": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
            "PR-AUC": average_precision_score(y_true, y_proba),
            "Recall_at_FPR": recall_at_fpr,
            "Brier_Score": brier_score_loss(y_true, y_proba_safe),
            "Log_Loss": log_loss(y_true, y_proba_safe),
            "TN": float(tn),
            "FP": float(fp),
            "FN": float(fn),
            "TP": float(tp),
            "FPR": float(fp) / (fp + tn) if (fp + tn) > 0 else 0.0,
            "TPR": float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0,
        }

    # ---------------------- feature importances ----------------------
    def _maybe_collect_feature_importances(self, name: str, model, sanitized_cols: List[str]):
        """جمع أهميات الميزات وإرجاعها للأسماء الأصلية."""
        base = getattr(model, "best_estimator_", model)
        importances = None
        if hasattr(base, "feature_importances_"):
            importances = np.array(base.feature_importances_, dtype=float)
        elif hasattr(base, "coef_"):
            coef = getattr(base, "coef_", None)
            if coef is not None:
                importances = np.abs(coef.ravel())

        if importances is None or len(importances) != len(sanitized_cols):
            return

        inv_map = {v: k for k, v in self._feature_name_map.items()}
        original_names = [inv_map.get(c, c) for c in sanitized_cols]
        df_imp = (
            pd.DataFrame({"feature": original_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        self.feature_importances[name] = df_imp


# ================================================================================================
# Part 4: Robustness Testing (Production-grade)
# ================================================================================================
import re
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, List

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
)


class RobustnessTestSuite:
    """Comprehensive robustness and adversarial testing with metric tracking."""

    def __init__(self, target_fpr: float = 0.003, random_state: int = 42):
        self.target_fpr = float(target_fpr)
        self.random_state = int(random_state)
        self.test_results: Dict[str, Any] = {}
        rng = np.random.RandomState(self.random_state)
        self._rng = rng

    # ------------------------------ helpers ------------------------------
    def _num_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep numeric/bool columns only (drop label if exists)."""
        if df is None or len(df) == 0:
            return df
        X = df.select_dtypes(include=[np.number, bool]).copy()
        if "label" in X.columns:
            X = X.drop(columns=["label"])
        return X.astype(float)

    def _sanitize_feature_names_apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        تطهير أسماء الأعمدة لتكون متوافقة مع نماذج الشجر/التصدير:
        أحرف/أرقام/Underscore فقط، لا يبدأ برقم، وضمان التفرد.
        (نسخة خفيفة تعمل دون الاعتماد على Part 3 map)
        """
        if not isinstance(df, pd.DataFrame):
            return df

        new_cols: List[str] = []
        seen = set()
        for col in df.columns:
            name = re.sub(r"[^0-9a-zA-Z_]+", "_", str(col))
            if re.match(r"^\d", name):  # لا يبدأ برقم
                name = f"F_{name}"
            name = name.strip("_") or "F"
            base = name
            k = 2
            while name in seen:
                name = f"{base}_{k}"
                k += 1
            seen.add(name)
            new_cols.append(name)

        df2 = df.copy()
        df2.columns = new_cols
        return df2

    # helper
    def _select_threshold(self, y_true, y_proba, thresholds):
        y_true = np.asarray(y_true).astype(int)
        costs = []
        for thr in thresholds:
            y_pred = (y_proba >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            costs.append(fp * self.config.cost_fp + fn * self.config.cost_fn)
        return thresholds[int(np.argmin(costs))]

    def _predict_labels(
        self, model, X: pd.DataFrame, threshold: Optional[float]
    ) -> np.ndarray:
        """
        يتعامل تلقائيًا مع:
          - calibrated models (predict_proba متاح)
          - نماذج بلا predict_proba (يسقط على predict)
        """
        if threshold is None:
            # لا يوجد عتبة مخصصة → استعمل predict مباشرةً
            try:
                return model.predict(X)
            except Exception:
                # fallback: احسب proba ثم 0.5
                p = self._predict_proba_safe(model, X)
                return (p >= 0.5).astype(int)

        # يوجد عتبة مخصصة
        p = self._predict_proba_safe(model, X)
        return (p >= float(threshold)).astype(int)

    def _predict_proba_safe(self, model, X: pd.DataFrame) -> np.ndarray:
        """ارجع احتمالات الفئة الإيجابية بأمان (يقصّ على [1e-9,1-1e-9])."""
        try:
            proba = model.predict_proba(X)[:, 1]
        except Exception:
            # بعض النماذج تعطي decision_function
            df = model.decision_function(X)
            # Platt-like squashing
            proba = 1.0 / (1.0 + np.exp(-np.clip(df, -20, 20)))
        return np.clip(np.asarray(proba, dtype=float), 1e-9, 1.0 - 1e-9)

    def _metrics(self, y_true, y_pred, y_proba) -> Dict[str, float]:
        """حساب باقة المقاييس + Recall@FPR≤target."""
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        y_proba = np.asarray(y_proba, dtype=float)

        # Recall@FPR≤target
        fpr_curve, tpr_curve, thr = roc_curve(y_true, y_proba)
        idx = np.where(fpr_curve <= self.target_fpr)[0]
        recall_at_fpr = float(tpr_curve[idx[-1]]) if len(idx) else 0.0

        # ROC-AUC قد يفشل لو صنف واحد فقط يوجد في y_true
        try:
            roc = float(roc_auc_score(y_true, y_proba))
        except Exception:
            roc = float("nan")

        return {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred)),
            "F1": float(f1_score(y_true, y_pred)),
            "ROC-AUC": roc,
            "PR-AUC": float(average_precision_score(y_true, y_proba)),
            "Recall_at_FPR": recall_at_fpr,
            "TN": float(confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()[0]),
            "FP": float(confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()[1]),
            "FN": float(confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()[2]),
            "TP": float(confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()[3]),
        }

    def _record(self, key: str, value: Any):
        self.test_results[key] = value

    # ------------------------------ main API ------------------------------
    def run_all_tests(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: Optional[float] = None,
        group_col: str = "sender_domain",
    ) -> Dict[str, Any]:
        """
        شغّل باقة اختبارات الصلابة بالكامل.
        - model: أي مصنّف (يفضّل مُعايَر CalibratedClassifierCV)
        - threshold: عتبة التنبؤ (إن None تُستخدم 0.5 أو model.predict)
        - group_col: اسم عمود الدومين/الحملة إن وُجد (لتقييم domain shift الحقيقي)
        """
        print("=" * 100)
        print("🔬 Robustness Testing Suite")
        print("=" * 100)

        # تجهيز الميزات الرقمية والسِمتة
        Xn = self._sanitize_feature_names_apply(self._num_only(X_test))
        y = np.asarray(y_test).astype(int)

        # --- Baseline
        base_proba = self._predict_proba_safe(model, Xn)
        base_pred = self._predict_labels(model, Xn, threshold)
        baseline = self._metrics(y, base_pred, base_proba)
        self._record("baseline", baseline)
        print(
            f"Baseline → AUC: {baseline['ROC-AUC']:.4f} | F1: {baseline['F1']:.4f} | R@FPR≤{self.target_fpr}: {baseline['Recall_at_FPR']:.4f}"
        )

        # 1) Obfuscation
        self._test_obfuscation_resistance(model, Xn, y, threshold)

        # 2) Domain Shift (حقيقي إن وُجد group_col، وإلا محاكاة)
        self._test_domain_shift(model, X_test, Xn, y, threshold, group_col=group_col)

        # 3) Temporal Drift (حقيقي إن وُجد timestamp، وإلا محاكاة)
        self._test_temporal_drift(model, X_test, Xn, y, threshold)

        # 4) Adversarial numeric (FGSM-like بدون مشتقات)
        self._test_adversarial_examples(model, Xn, y, threshold)

        # 5) Multilingual / Locale (محاكاة عبر text_entropy وخصائص النص)
        self._test_multilingual_robustness(model, Xn, y, threshold)

        # 6) URL Shorteners
        self._test_url_shorteners(model, Xn, y, threshold)

        # 7) Threshold sensitivity
        self._test_threshold_sensitivity(model, Xn, y, threshold)

        # 8) Calibration drift (temperature scaling on probabilities)
        self._test_calibration_drift(model, Xn, y, threshold)

        return self.test_results

    # ------------------------ individual tests ------------------------
    def _test_obfuscation_resistance(self, model, Xn, y, threshold):
        print("\n🔍 Testing Obfuscation Resistance...")
        transforms = {
            "homoglyphs": self._t_homoglyphs,
            "url_encoding": self._t_url_encoding,
            "zero_width": self._t_zero_width,
            "case_variation": self._t_case_variation,
            "leetspeak": self._t_leetspeak,
        }
        results = {}
        base_proba = self._predict_proba_safe(model, Xn)
        base_pred = self._predict_labels(model, Xn, threshold)
        base = self._metrics(y, base_pred, base_proba)

        for name, fn in transforms.items():
            Xp = fn(Xn.copy())
            proba = self._predict_proba_safe(model, Xp)
            pred = self._predict_labels(model, Xp, threshold)
            m = self._metrics(y, pred, proba)
            degradation = float(base["Accuracy"] - m["Accuracy"])
            passed = degradation < 0.10  # ≤ 10% drop
            print(
                f"   {name}: ACC {m['Accuracy']:.4f} (Δ {degradation:+.2%})  → {'✅ Passed' if passed else '❌ Failed'}"
            )
            results[name] = {
                "metrics": m,
                "degradation_Accuracy": degradation,
                "passed": passed,
            }

        self._record("obfuscation", results)

    # --- numeric approximations for obfuscations
    def _t_homoglyphs(self, X: pd.DataFrame) -> pd.DataFrame:
        if "url_entropy" in X.columns:
            X["url_entropy"] *= 1.20
        if "has_homograph" in X.columns:
            X["has_homograph"] = 1
        if "num_dots" in X.columns:
            X["num_dots"] += 1
        return X

    def _t_url_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        if "url_length" in X.columns:
            X["url_length"] *= 1.50
        if "special_char_ratio" in X.columns:
            X["special_char_ratio"] *= 2.00
        return X

    def _t_zero_width(self, X: pd.DataFrame) -> pd.DataFrame:
        if "char_count" in X.columns:
            X["char_count"] *= 1.10
        if "text_entropy" in X.columns:
            X["text_entropy"] *= 0.90
        return X

    def _t_case_variation(self, X: pd.DataFrame) -> pd.DataFrame:
        if "all_caps_ratio" in X.columns:
            X["all_caps_ratio"] = self._rng.uniform(0, 1, len(X))
        return X

    def _t_leetspeak(self, X: pd.DataFrame) -> pd.DataFrame:
        if "text_entropy" in X.columns:
            X["text_entropy"] *= 1.15
        if "digits_in_text" in X.columns:
            X["digits_in_text"] = X["digits_in_text"] * 1.20 + 1
        return X

    def _test_domain_shift(
        self, model, X_raw, Xn, y, threshold, group_col="sender_domain"
    ):
        print("\n🌐 Testing Domain Shift Robustness...")
        results: Dict[str, Any] = {}
        if isinstance(X_raw, pd.DataFrame) and group_col in X_raw.columns:
            # تقييم واقعي لكل دومين موجود في الاختبار
            for dom, idx in X_raw.groupby(group_col).groups.items():
                idx = np.asarray(list(idx))
                Xd = Xn.iloc[idx]
                yd = y[idx]
                proba = self._predict_proba_safe(model, Xd)
                pred = self._predict_labels(model, Xd, threshold)
                m = self._metrics(yd, pred, proba)
                results[str(dom)] = m
                print(f"   {dom}: AUC {m['ROC-AUC']:.4f} | F1 {m['F1']:.4f}")
        else:
            # محاكاة لو ما في عمود دومين
            for domain in ["financial", "ecommerce", "social_media", "government"]:
                Xd = self._simulate_domain(Xn.copy(), domain)
                proba = self._predict_proba_safe(model, Xd)
                pred = self._predict_labels(model, Xd, threshold)
                m = self._metrics(y, pred, proba)
                results[domain] = m
                print(f"   {domain}: AUC {m['ROC-AUC']:.4f} | F1 {m['F1']:.4f}")
        self._record("domain_shift", results)

    def _simulate_domain(self, X: pd.DataFrame, domain: str) -> pd.DataFrame:
        if domain == "financial":
            if "urgency_score" in X.columns:
                X["urgency_score"] *= 1.30
        elif domain == "ecommerce":
            for c in ["has_form", "form_present"]:
                if c in X.columns:
                    X[c] = 1
        elif domain == "social_media":
            for c in ["has_url_shortener", "url_shortener"]:
                if c in X.columns:
                    X[c] = self._rng.binomial(1, 0.3, len(X))
        elif domain == "government":
            if "has_ip_in_url" in X.columns:
                X["has_ip_in_url"] = 0
        return X

    def _test_temporal_drift(self, model, X_raw, Xn, y, threshold):
        print("\n📅 Testing Temporal Drift...")
        results: Dict[str, Any] = {}

        # تقييم حقيقي حسب فترات زمنية إن توفر timestamp
        if isinstance(X_raw, pd.DataFrame) and "timestamp" in X_raw.columns:
            df = X_raw[["timestamp"]].copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            # اقسم إلى رباعيات زمنية
            q = pd.qcut(
                df["timestamp"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]
            )
            for label in ["Q1", "Q2", "Q3", "Q4"]:
                idx = np.where(q == label)[0]
                if len(idx) == 0:
                    continue
                Xq = Xn.iloc[idx]
                yq = y[idx]
                proba = self._predict_proba_safe(model, Xq)
                pred = self._predict_labels(model, Xq, threshold)
                m = self._metrics(yq, pred, proba)
                results[label] = m
                print(f"   {label}: AUC {m['ROC-AUC']:.4f} | F1 {m['F1']:.4f}")
        else:
            # محاكاة انجراف تدريجي
            factors = {
                "current": 1.00,
                "1_month": 1.05,
                "3_months": 1.15,
                "6_months": 1.30,
            }
            for period, factor in factors.items():
                Xd = Xn.copy()
                for c in Xd.columns:
                    if np.issubdtype(Xd[c].dtype, np.number):
                        Xd[c] = Xd[c] * factor
                proba = self._predict_proba_safe(model, Xd)
                pred = self._predict_labels(model, Xd, threshold)
                m = self._metrics(y, pred, proba)
                results[period] = m
                print(f"   {period}: AUC {m['ROC-AUC']:.4f} | F1 {m['F1']:.4f}")

        self._record("temporal_drift", results)

    def _test_adversarial_examples(self, model, Xn, y, threshold):
        print("\n⚔️ Testing Adversarial Robustness (numeric FGSM-like)...")
        eps = 0.10
        Xd = Xn.copy()
        # Perturb suspicious features “towards phishing”
        inc_keys = [
            "url_entropy",
            "num_dots",
            "has_homograph",
            "urgency_score",
            "char_count",
            "digits_in_text",
        ]
        for c in Xd.columns:
            if np.issubdtype(Xd[c].dtype, np.number):
                if c in inc_keys:
                    Xd[c] = Xd[c] + eps * np.abs(self._rng.randn(len(Xd)))
                else:
                    Xd[c] = Xd[c] + 0.02 * self._rng.randn(len(Xd))
        proba = self._predict_proba_safe(model, Xd)
        pred = self._predict_labels(model, Xd, threshold)
        m = self._metrics(y, pred, proba)
        print(f"   Adversarial → AUC {m['ROC-AUC']:.4f} | F1 {m['F1']:.4f}")
        self._record("adversarial", m)

    def _test_multilingual_robustness(self, model, Xn, y, threshold):
        print("\n🌍 Testing Multilingual Robustness...")
        results: Dict[str, Any] = {}
        lang_factors = {
            "english": 1.00,
            "spanish": 0.95,
            "arabic": 1.10,
            "chinese": 1.20,
        }
        for lang, factor in lang_factors.items():
            Xd = Xn.copy()
            if "text_entropy" in Xd.columns:
                Xd["text_entropy"] = Xd["text_entropy"] * factor
            proba = self._predict_proba_safe(model, Xd)
            pred = self._predict_labels(model, Xd, threshold)
            m = self._metrics(y, pred, proba)
            results[lang] = m
            print(f"   {lang}: AUC {m['ROC-AUC']:.4f} | F1 {m['F1']:.4f}")
        self._record("multilingual", results)

    def _test_url_shorteners(self, model, Xn, y, threshold):
        print("\n🔗 Testing URL Shortener Handling...")
        Xd = Xn.copy()
        for c in ["has_url_shortener", "url_shortener"]:
            if c in Xd.columns:
                Xd[c] = self._rng.binomial(1, 0.5, len(Xd))
        if "url_length" in Xd.columns:
            Xd["url_length"] = Xd["url_length"] * 0.30
        proba = self._predict_proba_safe(model, Xd)
        pred = self._predict_labels(model, Xd, threshold)
        m = self._metrics(y, pred, proba)
        print(f"   With shorteners: AUC {m['ROC-AUC']:.4f} | F1 {m['F1']:.4f}")
        self._record("url_shorteners", m)

    def _test_threshold_sensitivity(self, model, Xn, y, threshold):
        print("\n🎚️ Testing Threshold Sensitivity...")
        # إن لم تُمرَّر عتبة، اعتبر 0.5 baseline
        base_thr = 0.5 if threshold is None else float(threshold)
        grid = [max(0.01, base_thr * f) for f in [0.8, 0.9, 1.0, 1.1, 1.2]]
        results = {}
        for thr in grid:
            proba = self._predict_proba_safe(model, Xn)
            pred = (proba >= thr).astype(int)
            m = self._metrics(y, pred, proba)
            results[f"{thr:.3f}"] = m
            print(f"   thr={thr:.3f}: AUC {m['ROC-AUC']:.4f} | F1 {m['F1']:.4f}")
        self._record("threshold_sensitivity", results)

    def _test_calibration_drift(self, model, Xn, y, threshold):
        print("\n🌡️ Testing Calibration Drift (temperature scaling simulation)...")

        def _temp_scale(p, T):
            # لوݣِت → Temperature
            p = np.clip(p, 1e-9, 1 - 1e-9)
            logit = np.log(p / (1 - p))
            p2 = 1.0 / (1.0 + np.exp(-logit / float(T)))
            return np.clip(p2, 1e-9, 1 - 1e-9)

        base = self._predict_proba_safe(model, Xn)
        results = {}
        for T in [0.8, 1.0, 1.2, 1.5]:
            pT = _temp_scale(base, T)
            thr = 0.5 if threshold is None else float(threshold)
            pred = (pT >= thr).astype(int)
            m = self._metrics(y, pred, pT)
            results[f"T={T}"] = m
            print(f"   T={T}: AUC {m['ROC-AUC']:.4f} | F1 {m['F1']:.4f}")
        self._record("calibration_drift", results)


# ================================================================================================
# Part 5: Explainability with SHAP (Production-grade)
# ================================================================================================
import numpy as np
import pandas as pd
import shap
from typing import Any, Dict, Optional, List

from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


class ModelExplainer:
    """Model explainability using SHAP + (tree & permutation) importance + error analysis."""

    def __init__(self, shap_samples: int = 200, random_state: int = 42):
        # الحد الأقصى لعدد العينات عند حساب SHAP (لتقليل الزمن)
        self.shap_samples = int(
            os.getenv("SHAP_SAMPLES", shap_samples)
            if "SHAP_SAMPLES" in os.environ
            else shap_samples
        )
        self.random_state = int(random_state)
        self.shap_values = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self._last_shap_importance_df: Optional[pd.DataFrame] = None

    # ---------- helpers ----------
    def _num_only(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return df
        X = df.select_dtypes(include=[np.number, bool]).copy()
        if "label" in X.columns:
            X = X.drop(columns=["label"])
        return X.astype(float)

    def _unwrap_base_estimator(self, model):
        """
        إذا كان الموديل مُعايَر (CalibratedClassifierCV) نرجع المُقدّر الأساسي،
        وнакعُد fallback لنفس الموديل إن لم يتوافر.
        """
        base = getattr(model, "base_estimator", None)
        if base is None:
            base = getattr(model, "estimator", None)
        return base if base is not None else model

    def _is_tree_model(self, est) -> bool:
        names = (
            "RandomForest",
            "XGBClassifier",
            "LGBMClassifier",
            "CatBoostClassifier",
            "ExtraTrees",
            "GradientBoosting",
        )
        cls = est.__class__.__name__
        return any(n in cls for n in names) or hasattr(est, "feature_importances_")

    def _ensure_dense(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP Kernel/Linear يحتاج أحياناً Dense؛ نحول بأمان."""
        try:
            import scipy.sparse as sp  # noqa

            if sp.issparse(X):
                return X.toarray()
        except Exception:
            pass
        return np.asarray(X)

    # ---------- public API ----------
    def explain_model(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        y_train: Optional[pd.Series] = None,
        y_test: Optional[pd.Series] = None,
        threshold: Optional[float] = None,
    ):
        """
        يُرجع:
          - shap_df: DataFrame لأهميّة SHAP (feature, shap_importance)
          - fi_df  : DataFrame مدمجة (SHAP + tree + permutation)
          - errors : DataFrame لأخطاء التنبؤ (إن مرّرنا y_test)
        """
        print("=" * 100)
        print("🔍 Model Explainability Analysis")
        print("=" * 100)

        # ميزات رقمية فقط
        X_train_num = self._num_only(X_train)
        X_test_num = self._num_only(X_test)

        if feature_names is None or len(feature_names) != X_train_num.shape[1]:
            feature_names = list(X_train_num.columns)

        # 1) SHAP
        shap_df = self._generate_shap_explanations(
            model, X_train_num, X_test_num, feature_names
        )
        self._last_shap_importance_df = shap_df.copy() if shap_df is not None else None

        # 2) Feature Importance (tree + permutation) ودمج SHAP
        fi_df = self._calculate_feature_importance(
            model, X_train_num, y_train, feature_names
        )

        # 3) Error Analysis (اختياري)
        errors_df = None
        if y_test is not None:
            errors_df = self._perform_error_analysis(
                model, X_test_num, y_test, threshold=threshold
            )

        return shap_df, fi_df, errors_df

    # ---------- SHAP ----------
    def _generate_shap_explanations(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Generate SHAP explanations (يدعم شجري/خطي/مُعايَر) مع fallbacks آمنة."""
        print("\n📊 Generating SHAP explanations...")
        try:
            base_model = self._unwrap_base_estimator(model)

            # نختار Explainer الأنسب
            explainer = None
            if self._is_tree_model(base_model):
                # أسرع وأكثر استقرارًا للنماذج الشجرية
                try:
                    explainer = shap.TreeExplainer(base_model)
                except Exception:
                    explainer = None

            if explainer is None:
                # Linear → LinearExplainer لو أمكن
                if base_model.__class__.__name__ in (
                    "LogisticRegression",
                    "RidgeClassifier",
                ):
                    try:
                        explainer = shap.LinearExplainer(
                            base_model, self._ensure_dense(X_train)
                        )
                    except Exception:
                        explainer = None

            if explainer is None:
                # Fallback عام (Kernel/Auto)
                try:
                    explainer = shap.Explainer(
                        base_model.predict_proba, self._ensure_dense(X_train)
                    )
                except Exception:
                    explainer = shap.Explainer(base_model, self._ensure_dense(X_train))

            # عيّنة اختبار (لتقليل الوقت)
            n = min(self.shap_samples, len(X_test))
            if n <= 0:
                return pd.DataFrame(columns=["feature", "shap_importance"])
            X_sample = X_test.iloc[:n]
            sv = explainer(X_sample)
            self.shap_values = sv

            values = sv.values
            # multiclass → خُذ صنف الإيجابي
            if hasattr(values, "ndim") and values.ndim == 3:
                values = values[:, :, 1]
            shap_imp = np.abs(values).mean(axis=0)

            importance_df = (
                pd.DataFrame(
                    {
                        "feature": feature_names[: len(shap_imp)],
                        "shap_importance": shap_imp.astype(float),
                    }
                )
                .sort_values("shap_importance", ascending=False)
                .reset_index(drop=True)
            )

            print("\n🎯 Top 10 Features (by SHAP):")
            for _, row in importance_df.head(10).iterrows():
                print(f"   {row['feature']:30} {row['shap_importance']:.4f}")

            return importance_df

        except Exception as e:
            print(f"⚠️ SHAP failed: {e}")
            return pd.DataFrame(columns=["feature", "shap_importance"])

    # ---------- Importance ----------
    def _calculate_feature_importance(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series],
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Tree importance + permutation (roc_auc/accuracy) + دمج SHAP في إطار واحد."""
        print("\n📊 Calculating Feature Importance...")

        base_model = self._unwrap_base_estimator(model)
        importance_dict: Dict[str, np.ndarray] = {}

        # أهمية شجرية إن توفّرت
        if hasattr(base_model, "feature_importances_"):
            try:
                importance_dict["tree_importance"] = np.asarray(
                    getattr(base_model, "feature_importances_"), dtype=float
                )
            except Exception:
                pass

        # Permutation (إن وُجد y_train)
        try:
            if y_train is not None and len(y_train) == len(X_train):
                scoring = (
                    "roc_auc" if hasattr(base_model, "predict_proba") else "accuracy"
                )
                # نأخذ عيّنة صغيرة لو الداتا ضخمة (لتسريع الحساب)
                max_perm = int(os.getenv("PERM_SAMPLES", "400"))
                if len(X_train) > max_perm:
                    Xp = X_train.iloc[:max_perm]
                    yp = y_train.iloc[:max_perm]
                else:
                    Xp, yp = X_train, y_train

                perm = permutation_importance(
                    base_model,
                    Xp,
                    yp,
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1,
                    scoring=scoring,
                )
                importance_dict["permutation_importance"] = (
                    perm.importances_mean.astype(float)
                )
        except Exception as e:
            print(f"⚠️ Permutation importance failed: {e}")

        # إطار مجمّع
        out = pd.DataFrame({"feature": list(feature_names)})

        # دمج SHAP إن وُجد
        if getattr(self, "_last_shap_importance_df", None) is not None:
            try:
                out = out.merge(
                    self._last_shap_importance_df[["feature", "shap_importance"]],
                    on="feature",
                    how="left",
                )
            except Exception:
                pass

        # أضف باقي الأهميات
        for k, v in importance_dict.items():
            v = np.array(v, dtype=float)
            v = v[: len(out)]
            out[k] = v

        sort_cols = [
            c
            for c in ["shap_importance", "tree_importance", "permutation_importance"]
            if c in out.columns
        ]
        if sort_cols:
            out = out.sort_values(by=sort_cols, ascending=False).reset_index(drop=True)

        self.feature_importance = out
        return self.feature_importance

    # ---------- Error Analysis ----------
    def _perform_error_analysis(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """تحليل أخطاء التنبؤ (يدعم threshold مخصّصة)."""
        print("\n🔍 Error Analysis...")

        # احتمالات + عتبة
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            thr = 0.5 if threshold is None else float(threshold)
            y_pred = (proba >= thr).astype(int)
        else:
            y_pred = model.predict(X_test)
            proba = np.zeros_like(y_pred, dtype=float)

        mis_idx = np.where(y_pred != np.asarray(y_test))[0]
        errors = X_test.iloc[mis_idx].copy()
        errors["true_label"] = np.asarray(y_test)[mis_idx]
        errors["predicted_label"] = y_pred[mis_idx]
        errors["confidence"] = (
            proba[mis_idx] if proba.shape[0] == len(X_test) else np.nan
        )

        # ملخص سريع
        tn, fp, fn, tp = confusion_matrix(np.asarray(y_test), y_pred).ravel()
        total = max(len(y_test), 1)
        print(f"   False Positives: {int(fp)} ({(fp/total)*100:.2f}%)")
        print(f"   False Negatives: {int(fn)} ({(fn/total)*100:.2f}%)")
        print(
            f"   Precision: {precision_score(y_test, y_pred, zero_division=0):.4f} | "
            f"Recall: {recall_score(y_test, y_pred):.4f} | F1: {f1_score(y_test, y_pred):.4f}"
        )

        # لو وُجد sender_domain في X_test (قبل التحويل) اعرض أكبر الدومينات المسببة للأخطاء
        if "sender_domain" in getattr(X_test, "columns", []):
            domain_errors = (
                errors.groupby("sender_domain").size().sort_values(ascending=False)
            )
            print("\n   Top domains with errors:")
            for domain, count in domain_errors.head(5).items():
                print(f"      {domain}: {int(count)} errors")

        return errors


# ================================================================================================
# Part 6: Monitoring and Drift Detection (Production-grade, JSON persistence, predictor support)
# ================================================================================================

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class DriftMonitor:
    """Monitor model performance and detect drift (Data / Label / Performance / Feature).

    ملاحظات:
    - الفحوصات التوزيعية تتم على الأعمدة الرقمية فقط (Numeric/Bool).
    - Performance drift يمكن حسابه بطريقتين:
        1) model.predict(X_num)  <- إذا كان الموديل يتوقع مصفوفة ميزات رقمية جاهزة.
        2) predictor(X_raw_df)   <- Callable يلفّ كامل البايبلاين (يوصى به لخطّك الحالي).
    - baseline/history تُخزّن كـ JSON (آمن ومحمول) مع تحويل الـ NumPy إلى lists.
    """

    def __init__(self, config: "ModelConfig"):
        self.config = config
        self.baseline_stats: Dict[str, Any] = {}
        self.monitoring_history: List[Dict[str, Any]] = []

        # عتبات داخلية قابلة للتعديل
        self._psi_feature_threshold = 0.20  # PSI > 0.20 => ملامح Drift
        self._bins = 10
        self._max_history = 2000

    # ---------- helpers ----------
    def _num_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """أبقِ الأعمدة الرقمية/المنطقية فقط (واحذف label لو كان رقمياً)."""
        if df is None or len(df) == 0:
            return df
        X = df.select_dtypes(include=[np.number, bool]).copy()
        if "label" in X.columns:
            X = X.drop(columns=["label"])
        return X.astype(float)

    def _clean_numeric_series(self, s: pd.Series) -> pd.Series:
        """حوّل لقيم رقمية ونظّف NaN/Inf، وأرجع float32."""
        s = pd.to_numeric(s, errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        return s.astype(np.float32)

    def _safe_histogram(
        self, values: np.ndarray, bins: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Histogram آمن (يتعامل مع الأعمدة الثابتة والقيم غير المنتهية)."""
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return np.array([len(values)] + [0] * (bins - 1), dtype=int), np.linspace(
                0.0, 1.0, bins + 1
            )
        if vmin == vmax:
            eps = 1e-6 if vmin == 0 else abs(vmin) * 1e-6
            vmin -= eps
            vmax += eps
        c, e = np.histogram(values, bins=bins, range=(vmin, vmax))
        return c, e

    # ---------- Baseline ----------
    def establish_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: Optional[np.ndarray] = None,
        *,
        model: Optional[Any] = None,
        predictor: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
    ) -> None:
        """
        يبني baseline من الأعمدة الرقمية فقط ويخزّنه بصيغة JSON جاهزة للحفظ.
        - feature_distributions: {col: (counts, bin_edges)}
        - label_distribution: نسب الفئات
        - train_accuracy: دقة الأداء baseline (إن توفّر predictor/model)
        - feature_means / feature_stds: معايير z-score
        """
        Xn = self._num_only(X_train)
        fd: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        kept = 0
        for col in Xn.columns:
            s = self._clean_numeric_series(Xn[col])
            if s.empty:
                continue
            counts, edges = self._safe_histogram(s.to_numpy(), bins=self._bins)
            fd[col] = (counts.astype(int), edges.astype(float))
            kept += 1

        label_dist = None
        if y_train is not None:
            label_dist = pd.Series(y_train).value_counts(normalize=True)

        # Performance baseline (اختياري)
        train_acc = None
        try:
            if predictor is not None and y_train is not None:
                train_acc = float(accuracy_score(y_train, predictor(X_train)))
            elif (
                model is not None and y_train is not None and hasattr(model, "predict")
            ):
                train_acc = float(accuracy_score(y_train, model.predict(Xn)))
        except Exception:
            train_acc = None

        self.baseline_stats = {
            "feature_distributions": {
                k: (v[0].tolist(), v[1].tolist()) for k, v in fd.items()
            },
            "label_distribution": (
                label_dist.to_dict() if label_dist is not None else None
            ),
            "train_accuracy": train_acc,
            "feature_means": Xn.mean().astype(float).to_dict(),
            "feature_stds": Xn.std(ddof=0).replace(0.0, 1.0).astype(float).to_dict(),
            "created_at": datetime.utcnow().isoformat(),
        }
        print(f"✅ Baseline established (numeric kept: {kept})")

    # ---------- Detection ----------
    def detect_drift(
        self,
        X_new: pd.DataFrame,
        y_new: Optional[np.ndarray] = None,
        *,
        model: Optional[Any] = None,
        predictor: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """نفّذ تحليل الانحراف الكامل وأرجع تقريراً مفصلاً."""
        print("\n🔍 Drift Detection Analysis...")

        X_new_num = self._num_only(X_new)
        data_drift = self._detect_data_drift(X_new_num)
        label_drift = self._detect_label_drift(y_new)
        performance_drift = self._detect_performance_drift(
            X_new, X_new_num, y_new, model, predictor
        )
        feature_drift = self._detect_feature_drift(X_new_num)

        psi_mean = float(data_drift.get("psi_mean", 0.0))
        js = label_drift.get("js_divergence", None)
        js_norm = (js / np.log(2)) if js is not None else 0.0  # طبّع إلى [0..1]
        acc_deg = performance_drift.get("accuracy_degradation", 0.0) or 0.0
        acc_deg = max(0.0, float(acc_deg))

        overall = 0.5 * psi_mean + 0.2 * js_norm + 0.3 * acc_deg
        out = {
            "timestamp": datetime.utcnow().isoformat(),
            "data_drift": data_drift,
            "label_drift": label_drift,
            "performance_drift": performance_drift,
            "feature_drift": feature_drift,
            "overall_drift_score": float(overall),
            "alert_triggered": bool(
                overall > getattr(self.config, "drift_threshold", 0.1)
            ),
        }

        if out["alert_triggered"]:
            print(f"⚠️ DRIFT ALERT! overall={overall:.4f}")
        else:
            print("✅ No significant drift detected")

        self.monitoring_history.append(out)
        if len(self.monitoring_history) > self._max_history:
            self.monitoring_history = self.monitoring_history[-self._max_history :]
        return out

    def _detect_data_drift(self, X_new_num: pd.DataFrame) -> Dict[str, Any]:
        """PSI باستعمال نفس Bins للـbaseline (ثابتة)."""
        psi_scores: Dict[str, float] = {}
        base_fd: Dict[str, Tuple[List[int], List[float]]] = self.baseline_stats.get(
            "feature_distributions", {}
        )

        for col in X_new_num.columns:
            if col not in base_fd:
                continue
            base_counts_list, base_bins_list = base_fd[col]
            base_counts = np.asarray(base_counts_list, dtype=float)
            base_bins = np.asarray(base_bins_list, dtype=float)
            new_counts, _ = np.histogram(X_new_num[col].values, bins=base_bins)
            psi_scores[col] = self._calculate_psi(base_counts, new_counts)

        psi_mean = float(np.mean(list(psi_scores.values()))) if psi_scores else 0.0
        drifted = [k for k, v in psi_scores.items() if v > self._psi_feature_threshold]
        return {
            "psi_scores": {k: float(v) for k, v in psi_scores.items()},
            "psi_mean": psi_mean,
            "drifted_features": drifted,
        }

    def _detect_label_drift(self, y_new) -> Dict[str, Any]:
        """Jensen–Shannon بين توزيع اللابل الحالي والـbaseline (إن وُجد)."""
        base = self.baseline_stats.get("label_distribution", None)
        if y_new is None or base is None:
            return {
                "new_distribution": None,
                "baseline_distribution": base,
                "js_divergence": None,
            }

        new = pd.Series(y_new).value_counts(normalize=True).to_dict()
        keys = sorted(set(list(base.keys()) + list(new.keys())))
        p = np.array([float(base.get(k, 0.0)) for k in keys], dtype=float)
        q = np.array([float(new.get(k, 0.0)) for k in keys], dtype=float)
        js = self._calculate_js_divergence(p, q)
        return {
            "new_distribution": new,
            "baseline_distribution": base,
            "js_divergence": float(js),
        }

    def _detect_performance_drift(
        self,
        X_new_raw: pd.DataFrame,
        X_new_num: pd.DataFrame,
        y_new,
        model: Optional[Any],
        predictor: Optional[Callable[[pd.DataFrame], np.ndarray]],
    ) -> Dict[str, Any]:
        """قارن الدقة الحالية بدقّة الـbaseline (إن وُجدت)."""
        base_acc = self.baseline_stats.get("train_accuracy", None)
        if base_acc is None or y_new is None:
            return {
                "new_accuracy": None,
                "baseline_accuracy": base_acc,
                "accuracy_degradation": None,
            }

        try:
            if predictor is not None:
                y_pred = predictor(X_new_raw)
            elif model is not None and hasattr(model, "predict"):
                y_pred = model.predict(X_new_num)
            else:
                return {
                    "new_accuracy": None,
                    "baseline_accuracy": base_acc,
                    "accuracy_degradation": None,
                }

            new_acc = float(accuracy_score(y_new, y_pred))
        except Exception:
            new_acc = None

        deg = (
            (base_acc - new_acc)
            if (new_acc is not None and base_acc is not None)
            else None
        )
        return {
            "new_accuracy": new_acc,
            "baseline_accuracy": (float(base_acc) if base_acc is not None else None),
            "accuracy_degradation": (float(deg) if deg is not None else None),
        }

    def _detect_feature_drift(self, X_new_num: pd.DataFrame) -> List[Dict[str, Any]]:
        """Z-score (انحراف الوسط) مقارنةً بمتوسط/انحراف baseline."""
        out: List[Dict[str, Any]] = []
        mu = self.baseline_stats.get("feature_means", {})
        sd = self.baseline_stats.get("feature_stds", {})
        for col in X_new_num.columns:
            if col not in mu:
                continue
            new_m = float(X_new_num[col].mean())
            b_m = float(mu[col])
            b_s = float(sd.get(col, 1.0)) or 1.0
            z = abs(new_m - b_m) / (b_s if b_s != 0 else 1.0)
            if z > 3.0:
                out.append(
                    {
                        "feature": col,
                        "z_score": float(z),
                        "new_mean": new_m,
                        "baseline_mean": b_m,
                    }
                )
        return out

    # ---------- Math ----------
    def _calculate_psi(
        self, baseline: np.ndarray, current: np.ndarray, eps: float = 1e-10
    ) -> float:
        baseline = baseline.astype(float) + eps
        current = current.astype(float) + eps
        p = baseline / baseline.sum()
        q = current / current.sum()
        return float(np.sum((q - p) * np.log(q / p)))

    def _calculate_js_divergence(
        self, p: np.ndarray, q: np.ndarray, eps: float = 1e-10
    ) -> float:
        p = np.asarray(p, float) + eps
        p = p / p.sum()
        q = np.asarray(q, float) + eps
        q = q / q.sum()
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))
        return float(0.5 * (kl_pm + kl_qm))

    # ---------- QoL ----------
    def summarize_last(self) -> Optional[Dict[str, Any]]:
        if not self.monitoring_history:
            return None
        last = self.monitoring_history[-1]
        return {
            "timestamp": last["timestamp"],
            "overall_drift_score": last["overall_drift_score"],
            "alert_triggered": last["alert_triggered"],
            "psi_mean": last["data_drift"].get("psi_mean", 0.0),
            "js_divergence": last["label_drift"].get("js_divergence", None),
            "acc_degradation": last["performance_drift"].get(
                "accuracy_degradation", None
            ),
            "top_drifted_features": sorted(
                last["data_drift"].get("psi_scores", {}).items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:5],
        }

    def update_and_check(
        self, X_new, y_new=None, *, model=None, predictor=None
    ) -> bool:
        """اختصار: شغّل الكشف وأرجع هل أُطلق إنذار."""
        res = self.detect_drift(X_new, y_new, model=model, predictor=predictor)
        return bool(res["alert_triggered"])

    # ---------- Persistence ----------
    def save_baseline(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.baseline_stats, f, ensure_ascii=False, indent=2)

    def load_baseline(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.baseline_stats = json.load(f)

    def save_history(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.monitoring_history, f, ensure_ascii=False, indent=2)

    def load_history(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.monitoring_history = json.load(f)


# ================================================================================================
# Part 7: Main Pipeline
# ================================================================================================


class PhishingDetectionPipeline:
    """Complete production pipeline for phishing detection (TF-IDF or numeric)"""

    def __init__(self, config: ModelConfig):
        self.config = config

        # Core components (from Parts 1..6)
        self.preprocessor = DataPreprocessor(config)
        self.leakage_detector = LeakageDetector()
        self.trainer = AdvancedModelTrainer(config)
        self.robustness_tester = RobustnessTestSuite()
        self.explainer = ModelExplainer()
        self.drift_monitor = DriftMonitor(config)

        # State
        self.best_model: Any = None
        self.best_model_name: Optional[str] = None
        self.best_threshold: float = 0.5

        self.results: Dict[str, Any] = {}
        self.calibrated_models: Dict[str, Any] = {}
        self.model_thresholds: Dict[str, float] = {}

        # TF-IDF path state (EnhancedPhishingTrainer)
        self.use_tfidf: bool = False
        self.tfidf_trainer: Any = None
        self.feature_pipeline: Any = None  # vectorizer/featurizer used by TF-IDF path

    # ---------- helpers ----------
    def _num_only(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.select_dtypes(include=[np.number, bool]).copy()
        if "label" in X.columns:
            X = X.drop(columns=["label"])
        return X

    def save_artifacts(self, output_dir: str) -> None:
        """
        Save all production artifacts to `output_dir`.

        يَحفظ:
        - best_model.pkl            ← أفضل نموذج بعد المعايرة
        - data_cleaner.pkl          ← DataPreprocessor بكامل إحصاءاته
        - feature_pipeline.pkl      ← في حالة استخدام TF-IDF
        - results.json              ← ملخّص النتائج/المقاييس/التهيئة
        - validation_results.csv    ← مقارنة النماذج على مجموعة التحقق
        - feature_importance.csv    ← أهميات الميزات (إن وُجدت)
        - shap_top_features.csv     ← أهم ميزات SHAP (إن وُجدت)
        - errors_sample.csv         ← عيّنة من أخطاء الاختبار (إن وُجدت)
        - drift_baseline.json       ← خط الأساس للمراقبة
        - drift_history.json        ← سجّل جلسات المراقبة السابقة (إن وُجدت)
        - versions.txt              ← إصدارات الحِزم لسهولة التتبع
        - model_metadata.json       ← بيانات ميتا مفيدة للنشر والاستدعاء لاحقًا
        """
        from pathlib import Path
        import json
        import numpy as np
        import pandas as pd
        import joblib
        import sklearn  # للإصدار فقط
        import sys

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1) حِفظ المنظّف (Preprocessor) + أفضل نموذج
        if self.preprocessor is None:
            raise ValueError(
                "DataPreprocessor is None; run the pipeline before saving artifacts."
            )
        joblib.dump(self.preprocessor, out / "data_cleaner.pkl")

        if self.best_model is None:
            raise ValueError(
                "best_model is None; run the pipeline before saving artifacts."
            )
        joblib.dump(self.best_model, out / "best_model.pkl")

        # 2) في حالة TF-IDF احفظ الـ feature_pipeline
        try:
            if (
                getattr(self, "use_tfidf", False)
                and getattr(self, "feature_pipeline", None) is not None
            ):
                joblib.dump(self.feature_pipeline, out / "feature_pipeline.pkl")
        except Exception as e:
            print(f"⚠️ Skipped saving feature_pipeline.pkl: {e}")

        # 3) تجهيز results.json (مع تحويل آمن لأنواع NumPy/Pandas)
        def _to_jsonable(obj):
            # NumPy scalar
            if isinstance(obj, (np.floating, np.integer, np.bool_)):
                return obj.item()
            # NumPy array
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # pandas
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            if isinstance(obj, pd.Series):
                return obj.to_list()
            # أي شيء آخر يُترك كما هو
            return obj

        results_payload = {
            "best_model_name": self.best_model_name,
            "best_threshold": float(getattr(self, "best_threshold", 0.5)),
            "use_tfidf": bool(getattr(self, "use_tfidf", False)),
            "validation_comparison": _to_jsonable(
                getattr(self, "results", {}).get("validation_comparison", [])
            ),
            "test_metrics": _to_jsonable(
                getattr(self, "results", {}).get("test_metrics", {})
            ),
            "bootstrap_ci": _to_jsonable(
                getattr(self, "results", {}).get("bootstrap_ci", None)
            ),
            "robustness_results": _to_jsonable(
                getattr(self, "results", {}).get("robustness_results", None)
            ),
            "feature_importance": _to_jsonable(
                getattr(self, "results", {}).get("feature_importance", None)
            ),
            "shap_top_features": _to_jsonable(
                getattr(self, "results", {}).get("shap_top_features", None)
            ),
            "tfidf_top_coefficients": _to_jsonable(
                getattr(self, "results", {}).get("tfidf_top_coefficients", None)
            ),
            "errors_sample": _to_jsonable(
                getattr(self, "results", {}).get("errors_sample", None)
            ),
            "training_time_seconds": float(
                getattr(self, "results", {}).get("training_time_seconds", 0.0)
            ),
            "timestamp": getattr(self, "results", {}).get("timestamp", None),
        }

        # أضف thresholds لكل نموذج إن كانت متوفّرة
        try:
            if getattr(self, "model_thresholds", None):
                results_payload["model_thresholds"] = {
                    k: float(v) for k, v in self.model_thresholds.items()
                }
        except Exception:
            pass

        # إحصاءات الـ Preprocessor المهمة (تفيد الـ API)
        try:
            if hasattr(self.preprocessor, "feature_stats"):
                fs = self.preprocessor.feature_stats or {}
                results_payload["preprocessor_feature_stats"] = {
                    "feature_order": fs.get("feature_order", None),
                    "removed_low_var": fs.get("removed_low_var", []),
                }
        except Exception:
            pass

        with open(out / "results.json", "w", encoding="utf-8") as f:
            json.dump(
                results_payload, f, ensure_ascii=False, indent=2, default=_to_jsonable
            )

        # 4) CSVs اختيارية (لو موجودة في self.results)
        try:
            if self.results.get("validation_comparison"):
                pd.DataFrame(self.results["validation_comparison"]).to_csv(
                    out / "validation_results.csv", index=False, encoding="utf-8"
                )
        except Exception as e:
            print(f"⚠️ Could not write validation_results.csv: {e}")

        try:
            fi_obj = self.results.get("feature_importance")
            if fi_obj:
                # قد تكون dict-of-lists أو list-of-dicts
                fi_df = (
                    pd.DataFrame(fi_obj)
                    if isinstance(fi_obj, dict)
                    else pd.DataFrame(fi_obj)
                )
                if not fi_df.empty:
                    fi_df.to_csv(
                        out / "feature_importance.csv", index=False, encoding="utf-8"
                    )
        except Exception as e:
            print(f"⚠️ Could not write feature_importance.csv: {e}")

        try:
            shap_obj = self.results.get("shap_top_features")
            if shap_obj:
                shap_df = pd.DataFrame(shap_obj)
                if not shap_df.empty:
                    shap_df.to_csv(
                        out / "shap_top_features.csv", index=False, encoding="utf-8"
                    )
        except Exception as e:
            print(f"⚠️ Could not write shap_top_features.csv: {e}")

        try:
            err_obj = self.results.get("errors_sample")
            if err_obj:
                err_df = pd.DataFrame(err_obj)
                if not err_df.empty:
                    err_df.to_csv(
                        out / "errors_sample.csv", index=False, encoding="utf-8"
                    )
        except Exception as e:
            print(f"⚠️ Could not write errors_sample.csv: {e}")

        # 5) حفظ Baseline/History للانجراف
        try:
            if getattr(self, "drift_monitor", None) is not None:
                # baseline محفوظ داخل الـ DriftMonitor أيضًا، نصدّره للملفات
                self.drift_monitor.save_baseline(str(out / "drift_baseline.json"))
                self.drift_monitor.save_history(str(out / "drift_history.json"))
        except Exception as e:
            print(f"⚠️ Drift files not saved: {e}")

        # 6) ملف الإصدارات (يساعد في إعادة الإنتاج)
        try:
            import xgboost as _xgb  # noqa
        except Exception:
            _xgb = None
        try:
            import lightgbm as _lgb  # noqa
        except Exception:
            _lgb = None
        try:
            import catboost as _cb  # noqa
        except Exception:
            _cb = None
        try:
            import shap as _shap  # noqa
        except Exception:
            _shap = None

        versions = [
            f"python: {sys.version.split()[0]}",
            f"numpy: {np.__version__}",
            f"pandas: {pd.__version__}",
            f"scikit-learn: {sklearn.__version__}",
            f"xgboost: {getattr(_xgb, '__version__', 'N/A')}",
            f"lightgbm: {getattr(_lgb, '__version__', 'N/A')}",
            f"catboost: {getattr(_cb, '__version__', 'N/A')}",
            f"shap: {getattr(_shap, '__version__', 'N/A')}",
        ]
        try:
            (out / "versions.txt").write_text("\n".join(versions), encoding="utf-8")
        except Exception:
            pass

        # 7) ملف ميتاداتا مساعد للاستهلاك لاحقًا
        try:
            meta = {
                "best_model_name": self.best_model_name,
                "best_threshold": float(self.best_threshold),
                "use_tfidf": bool(getattr(self, "use_tfidf", False)),
                "feature_order": (
                    list(self.preprocessor.feature_stats.get("feature_order", []) or [])
                    if hasattr(self.preprocessor, "feature_stats")
                    else None
                ),
            }
            with open(out / "model_metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Could not write model_metadata.json: {e}")

        print(f"✅ Artifacts saved to: {out.resolve()}")

    def _make_predict_fn(self):
        """
        Callable يلفّ كامل خطّ التنبؤ:
        - TF-IDF: feature_pipeline -> (SVD لنماذج الشجر) -> calibrated -> threshold
        - Numeric: تعقيم أسماء الأعمدة بنفس mapping التدريب -> calibrated -> threshold
        يُستخدم للـ DriftMonitor ولأي تقييم خارجي على DataFrame خام.
        """
        if self.use_tfidf:
            trainer_ref = self.tfidf_trainer

            def _predict(X_raw: pd.DataFrame):
                X = trainer_ref.feature_pipeline.transform(X_raw)
                base = getattr(
                    self.best_model,
                    "base_estimator",
                    getattr(self.best_model, "estimator", self.best_model),
                )
                is_tree = base.__class__.__name__ in (
                    "XGBClassifier",
                    "LGBMClassifier",
                    "RandomForestClassifier",
                )
                if (
                    is_tree
                    and getattr(trainer_ref, "_svd", None) is not None
                    and getattr(trainer_ref.config, "use_svd", False)
                ):
                    X = trainer_ref._svd.transform(X)
                proba = self.best_model.predict_proba(X)[:, 1]
                return (proba >= self.best_threshold).astype(int)

            return _predict
        else:
            trainer_ref = self.trainer

            def _predict(X_raw: pd.DataFrame):
                Xn = trainer_ref._num_only(X_raw)
                Xn = Xn.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                Xn = trainer_ref._sanitize_feature_names_apply(Xn)
                proba = self.best_model.predict_proba(Xn)[:, 1]
                return (proba >= self.best_threshold).astype(int)

            return _predict

    # ---------- main ----------
    def run_complete_pipeline(
        self, train_path: str, val_path: str, test_path: str
    ) -> Dict[str, Any]:
        """Run the complete training and evaluation pipeline (clean → leakage → train → test → robustness → SHAP → drift)."""
        print(
            """
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║        🛡️ Production-Ready Phishing Detection System — Full Pipeline       ║
    ╚════════════════════════════════════════════════════════════════════════════╝
            """
        )
        start_time = time.time()

        # 1) Load & preprocess
        print("\n" + "=" * 100)
        print("📂 STAGE 1: Data Loading and Preprocessing")
        print("=" * 100)

        train_df = pd.read_csv(train_path)
        val_df   = pd.read_csv(val_path)
        test_df  = pd.read_csv(test_path)

        print(f"\n📊 Initial shapes → Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")

        train_df = self.preprocessor.clean_data(train_df, is_train=True)
        val_df   = self.preprocessor.clean_data(val_df,   is_train=False)
        test_df  = self.preprocessor.clean_data(test_df,  is_train=False)

        # 2) Leakage detection
        print("\n" + "=" * 100)
        print("🔒 STAGE 2: Leakage Detection and Removal")
        print("=" * 100)
        train_df, val_df, test_df = self.leakage_detector.detect_and_remove_leakage(
            train_df, val_df, test_df
        )

        # Align feature columns safely
        feature_cols = sorted(list((set(train_df.columns) & set(val_df.columns) & set(test_df.columns)) - {"label"}))
        if not feature_cols:
            raise ValueError("No common feature columns across splits after preprocessing/leakage filtering.")

        # ---- normalize labels to {0,1} for all splits ----
        def _labels_to01(s: pd.Series) -> pd.Series:
            if pd.api.types.is_numeric_dtype(s):
                arr = s.astype(float)
                return (arr > 0).astype(int)
            t = s.astype(str).str.strip().str.lower()
            pos = {"1","true","phish","spam","fraud","malicious","malware","phishing"}
            neg = {"0","false","ham","legit","benign","safe","ham_email"}
            mapped = pd.Series(np.where(t.isin(pos), 1, np.where(t.isin(neg), 0, np.nan)), index=s.index)
            if mapped.isna().any():
                unknown = sorted(t[mapped.isna()].unique().tolist())[:10]
                print(f"⚠️ Unrecognized labels found (up to 10): {unknown}. Defaulting unknowns to 0.")
                mapped = mapped.fillna(0)
            return mapped.astype(int)

        X_train, y_train_raw = train_df[feature_cols], train_df["label"]
        X_val,   y_val_raw   = val_df[feature_cols],   val_df["label"]
        X_test,  y_test_raw  = test_df[feature_cols],  test_df["label"]

        y_train = _labels_to01(y_train_raw)
        y_val   = _labels_to01(y_val_raw)
        y_test  = _labels_to01(y_test_raw)

        print(f"\n✅ Labels normalized → train pos={int((y_train==1).sum()):,} | neg={int((y_train==0).sum()):,}")

        # 3) Training
        print("\n" + "=" * 100)
        print("🚀 STAGE 3: Model Training and Optimization")
        print("=" * 100)

        want_tfidf = str(os.getenv("USE_TFIDF", "0")).lower() in ("1", "true", "yes")
        self.use_tfidf = bool(want_tfidf and (EnhancedPhishingTrainer is not None))
        self.tfidf_trainer = None  # for safety

        # حاول ربط sanitizer (لو متاح) مع الـ robustness tester
        try:
            self.robustness_tester._sanitize_feature_names_apply = self.trainer._sanitize_feature_names_apply
        except Exception:
            pass

        if want_tfidf and EnhancedPhishingTrainer is None:
            print("⚠️ USE_TFIDF=1 but enhanced_trainer.py not found — falling back to numeric features.")

        if self.use_tfidf:
            print("⚙️ Using Enhanced TF-IDF trainer (USE_TFIDF=1).")
            self.tfidf_trainer = EnhancedPhishingTrainer(self.config)

            results_df = self.tfidf_trainer.train_enhanced(X_train, y_train, X_val, y_val)

            self.best_model_name = self.tfidf_trainer.best_model_name
            self.best_model      = self.tfidf_trainer.best_model
            self.best_threshold  = float(self.tfidf_trainer.best_threshold)
            self.feature_pipeline = self.tfidf_trainer.feature_pipeline

            # Sync maps for downstream compatibility
            self.calibrated_models = dict(getattr(self.tfidf_trainer, "calibrated_models", {}))
            self.model_thresholds  = dict(getattr(self.tfidf_trainer, "model_thresholds", {}))
            try:
                self.trainer.calibrated_models = dict(self.calibrated_models)
                self.trainer.best_thresholds   = dict(self.model_thresholds)
            except Exception:
                pass
        else:
            # numeric-only training path (uses _fit_with_optional_eval → we will fix XGB there below)
            results_df = self.trainer.train_with_temporal_cv(X_train, y_train, X_val, y_val)  # يستدعي _fit_with_optional_eval داخلياً 

            if (results_df.empty or "Recall_at_FPR" not in results_df or "Model" not in results_df):
                raise ValueError("Training results missing required columns.")
            best_idx = results_df["Recall_at_FPR"].astype(float).idxmax()
            self.best_model_name = str(results_df.loc[best_idx, "Model"])

            calmap = getattr(self.trainer, "calibrated_models", {})
            self.best_model = calmap.get(self.best_model_name, None) or getattr(self.trainer, "models", {}).get(self.best_model_name, None)
            if self.best_model is None:
                raise KeyError(f"Best model '{self.best_model_name}' not found.")

            thrmap = getattr(self.trainer, "best_thresholds", {})
            if self.best_model_name not in thrmap:
                raise KeyError(f"No threshold recorded for best model '{self.best_model_name}'.")
            self.best_threshold = float(thrmap[self.best_model_name])

        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   Optimal Threshold: {self.best_threshold:.3f}")

        # 4) Test evaluation
        print("\n" + "=" * 100)
        print("🧪 STAGE 4: Test Set Evaluation")
        print("=" * 100)

        if self.use_tfidf:
            # ✅ الـ transform بترجع (X_lr, X_tree) - نختار حسب النموذج
            X_test_lr, X_test_tree = self.feature_pipeline.transform(X_test)
            
            # اختيار الـ features المناسبة
            if self.best_model_name == 'LR':
                X_in = X_test_lr
            else:
                # Tree models: استخدم X_tree (SVD-reduced)
                X_in = X_test_tree
                
                # ✅ تحويل لـ Dense للنماذج الشجرية
                if sp.issparse(X_in):  # ✅ استخدم sp.issparse بدلاً من issparse
                    X_in = X_in.toarray()
            
            y_proba = self.best_model.predict_proba(X_in)[:, 1]
        else:
            # numeric path (الكود الموجود - صحيح)
            X_test_num = self.trainer._num_only(X_test)
            X_test_num = X_test_num.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            X_test_num = self.trainer._sanitize_feature_names_apply(X_test_num)
            y_proba = self.best_model.predict_proba(X_test_num)[:, 1]

        y_pred = (y_proba >= self.best_threshold).astype(int)

        test_metrics = self.trainer._calculate_comprehensive_metrics(y_test, y_pred, y_proba)

        # Bootstrap CI (اختياري)
        try:
            ci = bootstrap_metrics_summary(
                y_test, y_proba, self.best_threshold, self.config.target_fpr,
                n_boot=int(os.getenv("BOOT_N", "500")),
            )
        except Exception:
            ci = None

        print(
            f"\n📊 Test Performance → "
            f"Acc {test_metrics['Accuracy']:.4f} | Prec {test_metrics['Precision']:.4f} | "
            f"Rec {test_metrics['Recall']:.4f} | F1 {test_metrics['F1']:.4f} | "
            f"ROC-AUC {test_metrics['ROC-AUC']:.4f} | PR-AUC {test_metrics['PR-AUC']:.4f} | "
            f"Recall@FPR≤{self.config.target_fpr}: {test_metrics['Recall_at_FPR']:.4f} | "
            f"Brier {test_metrics['Brier_Score']:.4f}"
        )

        # 5) Robustness testing (skip for TF-IDF sparse)
        print("\n" + "=" * 100)
        print("💪 STAGE 5: Robustness Testing")
        print("=" * 100)
        if self.use_tfidf:
            robustness_results = {"skipped": True, "reason": "TF-IDF sparse feature space"}
            print("ℹ️ Skipping robustness suite for TF-IDF models.")
        else:
            robustness_results = self.robustness_tester.run_all_tests(self.best_model, X_test, y_test)

        # 6) Explainability
        print("\n" + "=" * 100)
        print("🔍 STAGE 6: Model Explainability")
        print("=" * 100)
        shap_df = feature_importance_df = errors_df = top_coef_df = None
        if self.use_tfidf:
            try:
                top_coef_df = self.tfidf_trainer.top_coefficients(k=30)
            except Exception:
                top_coef_df = None
            print("ℹ️ SHAP skipped for TF-IDF; reporting LR top coefficients when available.")
        else:
            shap_df, feature_importance_df, errors_df = self.explainer.explain_model(
                self.best_model, X_train, X_test, feature_cols,
                y_train=y_train, y_test=y_test, threshold=self.best_threshold,
            )

        # 7) Drift monitoring baseline
        print("\n" + "=" * 100)
        print("📊 STAGE 7: Drift Monitoring Setup")
        print("=" * 100)
        predict_fn = self._make_predict_fn()
        self.drift_monitor.establish_baseline(X_train, y_train, predictor=predict_fn)

        # 8) Report
        elapsed = time.time() - start_time
        self.results = {
            "best_model_name": self.best_model_name,
            "best_threshold": float(self.best_threshold),
            "validation_comparison": results_df.to_dict(orient="records"),
            "test_metrics": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in test_metrics.items()},
            "bootstrap_ci": ci,
            "robustness_results": robustness_results,
            "feature_importance": (None if feature_importance_df is None or (hasattr(feature_importance_df, "empty") and feature_importance_df.empty)
                                else feature_importance_df.to_dict(orient="list")),
            "shap_top_features": (None if shap_df is None or (hasattr(shap_df, "empty") and shap_df.empty)
                                else shap_df.head(20).to_dict(orient="records")),
            "tfidf_top_coefficients": (None if top_coef_df is None or len(top_coef_df) == 0
                                    else top_coef_df.to_dict(orient="records")),
            "errors_sample": (None if errors_df is None or (hasattr(errors_df, "empty") and errors_df.empty)
                            else errors_df.head(50).to_dict(orient="records")),
            "training_time_seconds": float(elapsed),
            "timestamp": datetime.now().isoformat(),
        }

        print("\n" + "=" * 100)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 100)
        print(f"\n⏱️ Total execution time: {elapsed:.2f} sec")
        try:
            print("\nTop models by Recall@FPR:")
            print(results_df.sort_values("Recall_at_FPR", ascending=False).head(3))
        except Exception:
            pass

        return self.results



# =========================== Dataset Ingestion (archives + folders + multi-files) ===========================
from pathlib import Path
import os, re, json, csv, shutil
import numpy as np
import pandas as pd


def _prepare_datasets_from_any_source(base_dir: str) -> None:
    """
    Ingests datasets from BASE_DIR (files / folders / archives), unifies schema,
    deduplicates safely, stratified-splits (70/15/15), and saves canonical split files.

    Will SKIP if canonical split CSVs already exist.
    """

    # ---------- local consts ----------
    base = Path(base_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    try:
        script_dir = Path(__file__).resolve().parent
    except Exception:
        script_dir = Path.cwd()

    extracted_root = base / "_extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)

    # Canonical output names (use these everywhere)
    train_csv = base / "blue_multi_big_split_train.csv"
    val_csv = base / "blue_multi_big_split_val.csv"
    test_csv = base / "blue_multi_big_split_test.csv"

    # ---------- early exit if splits already exist ----------
    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        print("✅ Found existing split CSVs. Skipping ingestion.")
        return

    # Accept legacy names too; if found, copy/rename to canonical then return
    legacy = {
        "train": [base / "train_data.csv", base / "train.csv"],
        "val": [base / "val_data.csv", base / "valid.csv", base / "validation.csv"],
        "test": [base / "test_data.csv", base / "test.csv"],
    }
    if any(p.exists() for lst in legacy.values() for p in lst):

        def _pick_first(paths):
            for p in paths:
                if p.exists():
                    return p
            return None

        tr, va, te = (
            _pick_first(legacy["train"]),
            _pick_first(legacy["val"]),
            _pick_first(legacy["test"]),
        )
        if tr and va and te:
            shutil.copy2(tr, train_csv)
            shutil.copy2(va, val_csv)
            shutil.copy2(te, test_csv)
            print("✅ Found legacy split files; normalized to canonical names.")
            return

    # ---------- helpers (local, no globals) ----------
    def _dedupe_columns_local(df: pd.DataFrame) -> pd.DataFrame:
        # drop duplicate-named columns to avoid pandas concat InvalidIndexError
        if not df.columns.is_unique:
            df = df.loc[:, ~df.columns.duplicated()].copy()
        df.columns = [str(c).strip() for c in df.columns]
        return df

    def _is_archive(p: Path) -> bool:
        s = p.suffix.lower()
        return (
            s in {".zip", ".rar", ".7z", ".tar", ".gz"}
            or p.name.lower().endswith(".tar.gz")
            or p.name.lower().endswith(".tgz")
        )

    def _extract_archive(arc: Path, dest: Path):
        dest.mkdir(parents=True, exist_ok=True)
        s = arc.suffix.lower()
        try:
            if s == ".zip":
                import zipfile

                with zipfile.ZipFile(str(arc), "r") as zf:
                    zf.extractall(dest)
            elif s in {".tar", ".gz"} or arc.name.lower().endswith((".tar.gz", ".tgz")):
                import tarfile

                with tarfile.open(str(arc), "r:*") as tf:
                    tf.extractall(dest)
            elif s == ".rar":
                try:
                    import rarfile

                    with rarfile.RarFile(str(arc)) as rf:
                        rf.extractall(dest)
                except Exception as e:
                    print(
                        f"⚠️ RAR extraction failed ({e}). Install 'rarfile' + 'unrar'."
                    )
            elif s == ".7z":
                try:
                    import py7zr

                    with py7zr.SevenZipFile(str(arc), mode="r") as z:
                        z.extractall(path=dest)
                except Exception as e:
                    print(f"⚠️ 7z extraction failed ({e}). Install 'py7zr'.")
        except Exception as e:
            print(f"⚠️ Failed to extract {arc.name}: {e}")

    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        # unify to label/text/url/timestamp when possible
        cols = {str(c).lower().strip(): c for c in df.columns}

        # SMS classic
        if "v1" in cols and "v2" in cols:
            df = df.rename(columns={cols["v1"]: "label", cols["v2"]: "text"})

        # label
        for k in [
            "label",
            "labels",
            "class",
            "result",
            "target",
            "email type",
            "email_type",
            "type",
            "category",
            "y",
            "is_spam",
            "is_phish",
            "is_phishing",
            "phishing",
        ]:
            if k in cols and cols[k] != "label":
                df = df.rename(columns={cols[k]: "label"})
                break

        # text
        for k in [
            "text",
            "body",
            "message",
            "content",
            "email text",
            "email_text",
            "msg",
            "sms",
            "payload",
            "raw",
            "title",
        ]:
            if k in cols and cols[k] != "text":
                df = df.rename(columns={cols[k]: "text"})
                break

        # url
        for k in ["url", "link", "href", "domain"]:
            if k in cols and cols[k] != "url":
                df = df.rename(columns={cols[k]: "url"})
                break

        # timestamp
        for k in ["timestamp", "date", "datetime", "time"]:
            if k in cols and cols[k] != "timestamp":
                df = df.rename(columns={cols[k]: "timestamp"})
                break

        return df

    def _read_any_file(p: Path) -> pd.DataFrame | None:
        name = p.name.lower()
        try:
            if name == "smsspamcollection":
                df = pd.read_csv(p, sep="\t", header=None, names=["label", "text"])
            elif p.suffix.lower() == ".csv":
                try:
                    df = pd.read_csv(
                        p,
                        encoding="utf-8",
                        engine="python",
                        on_bad_lines="skip",
                        low_memory=False,
                    )
                except UnicodeDecodeError:
                    df = pd.read_csv(
                        p,
                        encoding="latin-1",
                        engine="python",
                        on_bad_lines="skip",
                        low_memory=False,
                    )
            elif p.suffix.lower() == ".tsv":
                df = pd.read_csv(
                    p, sep="\t", engine="python", on_bad_lines="skip", low_memory=False
                )
            elif p.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(p)
            elif p.suffix.lower() == ".jsonl":
                df = pd.read_json(p, lines=True)
            elif p.suffix.lower() == ".json":
                df = pd.read_json(p)
            elif p.suffix.lower() == ".parquet":
                try:
                    df = pd.read_parquet(p)  # requires pyarrow or fastparquet
                except Exception as e:
                    print(f"⚠️ Skipping parquet {p.name}: {e}")
                    return None
            else:
                return None
        except Exception as e:
            print(f"⚠️ Failed to read {p.name}: {e}")
            return None

        # drop unnamed cols and dedupe names
        for c in list(df.columns):
            if str(c).lower().startswith("unnamed"):
                df = df.drop(columns=[c])
        df = _dedupe_columns_local(df)
        df = _standardize_columns(df)
        return df

    # ---------- collect sources (files/folders/archives) ----------
    def _iter_candidate_paths():
        # env override
        env_path = os.getenv("DATASET_PATH")
        if env_path:
            p = Path(env_path)
            if p.exists():
                yield p

        # common file names in base / script_dir
        common = [
            "dataset.zip",
            "dataset.rar",
            "dataset.7z",
            "dataset.tar.gz",
            "PhishingData.csv",
            "Phishing_Email.csv",
            "SMSSpamCollection",
        ]
        for root in (base, script_dir):
            for name in common:
                p = root / name
                if p.exists():
                    yield p

        # everything else in base & script_dir
        for root in (base, script_dir):
            for p in root.glob("*"):
                if p.is_dir():
                    yield p
                elif p.is_file() and (
                    _is_archive(p)
                    or p.suffix.lower()
                    in {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl", ".parquet"}
                    or p.name.lower() == "smsspamcollection"
                ):
                    yield p

    # 1) extract archives we find
    extracted_dirs: list[Path] = []
    for src in _iter_candidate_paths():
        if src.is_file() and _is_archive(src):
            dest = extracted_root / src.stem
            print(f"📦 Extracting: {src}")
            _extract_archive(src, dest)
            if dest.exists():
                extracted_dirs.append(dest)

    # 2) enumerate readable files from base/script_dir and extracted folders
    all_targets: list[Path] = []

    def _collect_from_dir(d: Path):
        for p in d.rglob("*"):
            if p.is_file() and (
                p.suffix.lower()
                in {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl", ".parquet"}
                or p.name.lower() == "smsspamcollection"
            ):
                all_targets.append(p)

    for src in _iter_candidate_paths():
        if src.is_dir():
            _collect_from_dir(src)
        elif src.is_file():
            all_targets.append(src)
    for d in extracted_dirs:
        _collect_from_dir(d)

    # de-duplicate same absolute path
    uniq_targets, seen = [], set()
    for p in all_targets:
        r = str(p.resolve())
        if r not in seen:
            uniq_targets.append(p)
            seen.add(r)

    if not uniq_targets:
        print(
            "⚠️ No readable dataset files were found. Put your data in BASE_DIR or archive it."
        )
        return

    print(f"🔎 Found {len(uniq_targets)} candidate files to ingest.")

    # 3) read + standardize + keep useful cols
    frames: list[pd.DataFrame] = []
    for p in uniq_targets:
        df = _read_any_file(p)
        if df is None or df.empty:
            continue

        # ensure lower names then dedupe
        df.columns = [str(c).lower().strip() for c in df.columns]
        df = _dedupe_columns_local(df)

        # build text if missing
        if "text" not in df.columns:
            text_candidates = [
                c
                for c in df.columns
                if c
                in (
                    "body",
                    "content",
                    "url",
                    "subject",
                    "message",
                    "email",
                    "raw",
                    "payload",
                    "title",
                )
            ]
            if text_candidates:
                df["text"] = (
                    df[text_candidates].astype(str).fillna("").agg(" ".join, axis=1)
                )
            else:
                obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
                df["text"] = (
                    df[obj_cols].astype(str).fillna("").agg(" ".join, axis=1)
                    if obj_cols
                    else ""
                )

        frames.append(df)

    if not frames:
        print("⚠️ Nothing could be read after standardization.")
        return

    # SAFE concat (after per-frame dedupe)
    df_all = pd.concat(
        [_dedupe_columns_local(f) for f in frames], ignore_index=True, sort=False
    )

    # 4) basic cleaning
    if "text" in df_all.columns:
        df_all["text"] = (
            df_all["text"]
            .astype(str)
            .fillna("")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # map label -> {0,1}
    if "label" not in df_all.columns:
        raise ValueError("No 'label' column found in any source.")

    lab = df_all["label"].astype(str).str.strip().str.lower()
    direct_map = {
        "1": 1,
        "true": 1,
        "yes": 1,
        "spam": 1,
        "phishing": 1,
        "malicious": 1,
        "fraud": 1,
        "attack": 1,
        "bad": 1,
        "phishing email": 1,
        "spam email": 1,
        "0": 0,
        "false": 0,
        "no": 0,
        "ham": 0,
        "legitimate": 0,
        "benign": 0,
        "good": 0,
        "normal": 0,
        "legitimate email": 0,
        "not spam": 0,
    }
    lab_mapped = lab.map(direct_map)

    is_phish = lab_mapped.isna() & lab.str.contains(
        r"(phish|spam|malicious|fraud|attack|scam|fake|decept)", regex=True
    )
    is_legit = lab_mapped.isna() & lab.str.contains(
        r"(legit|ham|benign|safe|normal|clean|valid)", regex=True
    )
    lab_mapped = np.where(is_phish, 1, np.where(is_legit, 0, lab_mapped)).astype(
        "float"
    )

    df_all["label"] = pd.to_numeric(lab_mapped, errors="coerce")
    before = len(df_all)
    df_all = df_all.dropna(subset=["label"]).copy()
    df_all["label"] = df_all["label"].astype(int)
    dropped = before - len(df_all)
    if dropped > 0:
        print(f"🧹 Dropped {dropped} rows with unknown labels.")

    # require text or url
    if "text" not in df_all.columns and "url" not in df_all.columns:
        raise ValueError("Dataset must contain at least 'text' or 'url' column.")

    # drop empties/duplicates
    if "text" in df_all.columns:
        df_all = df_all[df_all["text"].astype(str).str.len() > 0].copy()
    dup_subset = [c for c in ["text", "url"] if c in df_all.columns]
    if dup_subset:
        df_all = df_all.drop_duplicates(subset=dup_subset).reset_index(drop=True)

    # timestamp fallback
    if "timestamp" not in df_all.columns:
        df_all["timestamp"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(
            np.arange(len(df_all)), unit="h"
        )

    # 5) stratified split 70/15/15
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df_all, test_size=0.30, stratify=df_all["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    # 6) save canonical outputs
    train_df.to_csv(train_csv, index=False, encoding="utf-8")
    val_df.to_csv(val_csv, index=False, encoding="utf-8")
    test_df.to_csv(test_csv, index=False, encoding="utf-8")
    print(f"✅ Saved splits to: {base}")
    print(f"   Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")


# ================================================================================================
# Main Execution (+ dataset preparer)
# ================================================================================================

from pathlib import Path
import os
import numpy as np
import pandas as pd

# ملاحظات:
# - نعتمد على الدالة _prepare_datasets_from_any_source(BASE_DIR) المعرّفة أعلاه.
# - الدالة _normalize_single_file_dataset(base_dir) موجودة أسفل هذا التعليق (لا تغيّر اسمها).


def _normalize_single_file_dataset(base_dir: str):
    """
    يجهّز داتا من ملف واحد إلى ثلاث ملفات: train/val/test داخل BASE_DIR.
    يبحث عن ملفات مرشّحة: SMSSpamCollection / PhishingData.csv / Phishing_Email.csv
    أو أي CSV يحتوي label + (text/url). يقسم 70/15/15 مع تنظيف مرن.
    """
    import sys, csv
    from sklearn.model_selection import (
        train_test_split,
    )  # استيراد محلي لتجنّب مشاكل الترتيب

    # تكبير حد حجم الحقل للصفوف الطويلة
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

    base = Path(base_dir)
    train_csv = base / "train_data.csv"
    val_csv = base / "val_data.csv"
    test_csv = base / "test_data.csv"

    # لو الملفات موجودة مسبقًا، لا تعيد التحضير (إلا إذا FORCE_PREPARE=1)
    if (
        train_csv.exists()
        and val_csv.exists()
        and test_csv.exists()
        and str(os.getenv("FORCE_PREPARE", "0")).lower() not in ("1", "true", "yes")
    ):
        print(
            "✅ Found existing train/val/test CSVs. Skipping single-file preparation."
        )
        return

    script_dir = Path(__file__).resolve().parent

    # لو المستخدم حدّد مصدر معيّن بالبيئة، أعطه أولوية
    dataset_env = os.getenv("DATASET_PATH")
    env_candidate = [Path(dataset_env)] if dataset_env else []

    # مرشحين داخل BASE_DIR ومجلد السكربت
    candidates = env_candidate + [
        base / "PhishingData.csv",
        base / "Phishing_Email.csv",
        base / "SMSSpamCollection",
        script_dir / "PhishingData.csv",
        script_dir / "Phishing_Email.csv",
        script_dir / "SMSSpamCollection",
    ]
    candidates += [
        p
        for p in base.glob("*.csv")
        if p.name.lower() not in {"train_data.csv", "val_data.csv", "test_data.csv"}
    ]
    candidates += [
        p
        for p in script_dir.glob("*.csv")
        if p.name.lower() not in {"train_data.csv", "val_data.csv", "test_data.csv"}
    ]

    src = next((c for c in candidates if c.exists()), None)
    if src is None:
        print(
            "⚠️ No single-source CSV found to prepare. Put your raw CSV in BASE_DIR or next to App.py."
        )
        return

    print(f"📂 Preparing dataset from: {src}")

    # قراءة مرنة
    if src.name == "SMSSpamCollection":
        df = pd.read_csv(src, sep="\t", header=None, names=["label", "text"])
        df["label"] = df["label"].astype(str).str.lower().map({"ham": 0, "spam": 1})
    else:
        try:
            df = pd.read_csv(
                src, encoding="utf-8", engine="python", on_bad_lines="skip"
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                src, encoding="latin-1", engine="python", on_bad_lines="skip"
            )

    # حذف الأعمدة المؤشرة تلقائيًا
    for c in list(df.columns):
        if str(c).lower().startswith("unnamed"):
            df = df.drop(columns=[c])

    # توحيد أسماء الأعمدة
    cols = {c.lower().strip(): c for c in df.columns}
    if "v1" in cols and "v2" in cols:
        df = df.rename(columns={cols["v1"]: "label", cols["v2"]: "text"})

    for k in [
        "label",
        "labels",
        "class",
        "result",
        "target",
        "email type",
        "email_type",
        "type",
        "category",
        "y",
        "is_spam",
    ]:
        if k in cols and cols[k] != "label":
            df = df.rename(columns={cols[k]: "label"})
            break
    if "label" not in df.columns:
        raise ValueError(
            f"Dataset must contain a 'label' column. Found: {list(df.columns)[:20]}"
        )

    for k in [
        "text",
        "body",
        "message",
        "content",
        "email text",
        "email_text",
        "msg",
        "sms",
    ]:
        if k in cols and cols[k] != "text":
            df = df.rename(columns={cols[k]: "text"})
            break
    for k in ["url", "link", "href", "domain"]:
        if k in cols and cols[k] != "url":
            df = df.rename(columns={cols[k]: "url"})
            break

    if "text" in df.columns:
        df["text"] = (
            df["text"]
            .astype(str)
            .fillna("")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # تحويل label إلى 0/1 بمرونة
    lab = df["label"].astype(str).str.strip().str.lower()
    direct_map = {
        "spam": 1,
        "phishing": 1,
        "malicious": 1,
        "fraud": 1,
        "attack": 1,
        "bad": 1,
        "phishing email": 1,
        "spam email": 1,
        "1": 1,
        "true": 1,
        "yes": 1,
        "ham": 0,
        "legitimate": 0,
        "benign": 0,
        "good": 0,
        "normal": 0,
        "legitimate email": 0,
        "not spam": 0,
        "0": 0,
        "false": 0,
        "no": 0,
    }
    lab_mapped = lab.map(direct_map)
    is_phish = lab_mapped.isna() & lab.str.contains(
        r"(phish|spam|malicious|fraud|attack|scam|fake|decept)", regex=True
    )
    is_legit = lab_mapped.isna() & lab.str.contains(
        r"(legit|ham|benign|safe|normal|clean|valid)", regex=True
    )
    lab_mapped = np.where(is_phish, 1, np.where(is_legit, 0, lab_mapped)).astype(
        "float"
    )

    df["label"] = pd.to_numeric(lab_mapped, errors="coerce")
    before = len(df)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    dropped = before - len(df)
    if dropped > 0:
        print(
            f"🧹 Dropped {dropped} rows with unknown labels (after flexible mapping)."
        )
    if len(df) == 0:
        raise ValueError(
            "No rows left after mapping labels. Please check the label values in your CSV."
        )

    if "text" not in df.columns and "url" not in df.columns:
        raise ValueError("Dataset must contain at least 'text' or 'url' column.")
    if "text" in df.columns:
        df = df[df["text"].astype(str).str.len() > 0].copy()

    dup_subset = [c for c in ["text", "url"] if c in df.columns]
    if dup_subset:
        df = df.drop_duplicates(subset=dup_subset).reset_index(drop=True)

    ts_cols = {c.lower(): c for c in df.columns}
    if not any(k in ts_cols for k in ["timestamp", "date", "datetime", "time"]):
        df["timestamp"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(
            np.arange(len(df)), unit="h"
        )

    base.mkdir(parents=True, exist_ok=True)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    train_df.to_csv(train_csv, index=False, encoding="utf-8")
    val_df.to_csv(val_csv, index=False, encoding="utf-8")
    test_df.to_csv(test_csv, index=False, encoding="utf-8")
    print(f"✅ Saved train/val/test CSVs into: {base_dir}")


# ضعها فوق _ensure_splits_or_prepare مباشرةً
def _prefer_existing_splits(base_dir):
    import os
    data_dir = os.environ.get("DATASET_DIR", os.path.join(base_dir, "data"))
    candidates = {
        "train": ["merged_plus_synth_split_train_uniq.csv", "train.csv"],
        "val":   ["merged_plus_synth_split_val_uniq.csv",   "val.csv"],
        "test":  ["merged_plus_synth_split_test_uniq.csv",  "test.csv"],
    }
    paths = {}
    for split, names in candidates.items():
        for n in names:
            p = os.path.join(data_dir, n)
            if os.path.exists(p):
                paths[split] = p
                break
    return (paths.get("train"), paths.get("val"), paths.get("test")) if len(paths) == 3 else None



def _ensure_splits_or_prepare(base_dir: str):
    """
    Guarantees train/val/test CSVs exist and returns their paths.
    Accepts either:
      - train_data.csv / val_data.csv / test_data.csv   (canonical)
      - blue_multi_big_split_{train,val,test}.csv       (alt)
    If only the alt set exists, copy/normalize to the canonical names.
    """
    import os, shutil
    from pathlib import Path

    # ⬅️ هنا كان الخطأ: استخدم base_dir بدل BASE_DIR
    paths = _prefer_existing_splits(base_dir)
    if paths:
        return paths  # استخدم الـ splits الجاهزة فوراً

    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    # Canonical names
    tr_c = base / "train_data.csv"
    va_c = base / "val_data.csv"
    te_c = base / "test_data.csv"

    # Alternate names written by the other ingester
    tr_b = base / "blue_multi_big_split_train.csv"
    va_b = base / "blue_multi_big_split_val.csv"
    te_b = base / "blue_multi_big_split_test.csv"

    def _normalize_from_blue():
        if tr_b.exists() and va_b.exists() and te_b.exists():
            shutil.copy2(tr_b, tr_c)
            shutil.copy2(va_b, va_c)
            shutil.copy2(te_b, te_c)
            print("ℹ️ Normalized blue_multi_big_split_* → train/val/test CSVs.")

    # 1) If canonical already present and not forcing, use them
    force = str(os.getenv("FORCE_PREPARE", "0")).lower() in ("1", "true", "yes")
    if tr_c.exists() and va_c.exists() and te_c.exists() and not force:
        print(f"✅ Using existing splits in: {base_dir}")
        return str(tr_c), str(va_c), str(te_c)

    # 2) If only blue_* exist, normalize to canonical and return
    if tr_b.exists() and va_b.exists() and te_b.exists():
        _normalize_from_blue()
        return str(tr_c), str(va_c), str(te_c)

    # 3) Try multi-source ingestion (may create either scheme)
    try:
        print("🧩 Preparing datasets (multi-source ingestion)…")
        _prepare_datasets_from_any_source(str(base))  # your existing function
    except Exception as e:
        print(f"ℹ️ Multi-source ingestion raised: {e}")

    # 4) Re-check: prefer canonical; otherwise normalize from blue_*
    if tr_c.exists() and va_c.exists() and te_c.exists():
        return str(tr_c), str(va_c), str(te_c)
    if tr_b.exists() and va_b.exists() and te_b.exists():
        _normalize_from_blue()
        return str(tr_c), str(va_c), str(te_c)

    # 5) Fallback: single-file normalizer
    print("↩️ Falling back to single-file normalizer…")
    _normalize_single_file_dataset(str(base))

    if tr_c.exists() and va_c.exists() and te_c.exists():
        return str(tr_c), str(va_c), str(te_c)

    raise FileNotFoundError(
        "No train/val/test CSVs could be created. "
        "Place a labeled CSV (with 'label' + text/url) in BASE_DIR or set DATASET_PATH."
    )


def main():
    """Main execution function"""
    from datetime import datetime  # فقط للزينة في الطباعة

    # Configuration
    config = ModelConfig(
        target_fpr=0.003,
        cost_fp=1.0,
        cost_fn=10.0,
        early_stopping_rounds=50,
        cv_folds=5,
        time_series_cv=True,
        calibration_method="isotonic",
        min_recall_threshold=0.80,
        drift_threshold=0.1,
    )

    # BASE_DIR: من البيئة أو بجانب الملف (Cross-platform)
    default_base = Path(__file__).parent.joinpath("PhishingData").resolve()
    BASE_DIR = os.getenv("BASE_DIR", str(default_base))
    if not Path(BASE_DIR).exists():
        BASE_DIR = str(default_base)
    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
    print(f"\n📁 BASE_DIR: {BASE_DIR}")

    # جهّز أو استخدم السبلت
    train_path, val_path, test_path = _ensure_splits_or_prepare(BASE_DIR)
    output_dir = str(Path(BASE_DIR) / "artifacts")

    # Run pipeline
    pipeline = PhishingDetectionPipeline(config)

    try:
        results = pipeline.run_complete_pipeline(train_path, val_path, test_path)
        pipeline.save_artifacts(output_dir)
        generate_final_report(results, output_dir)

        # (اختياري) بطاقة نموذج سريعة
        try:
            create_model_card(results, str(Path(output_dir) / "model_card.md"))
        except Exception as _:
            pass

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback

        traceback.print_exc()


# ================================================================================================
# Utility Functions
# ================================================================================================

from pathlib import Path
from typing import Optional, Dict, Any
import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
import joblib


def generate_final_report(results: dict, output_dir: str) -> None:
    """Generate concise final report and save it to artifacts."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    training_time = float(
        results.get("training_time_seconds", results.get("training_time", 0.0))
    )

    report = f"""
========================  FINAL REPORT  ========================

Generated: {results.get('timestamp')}
Training Time: {training_time:.2f} seconds

BEST MODEL
----------
Model: {results.get('best_model_name')}
Optimal Threshold: {float(results.get('best_threshold', 0.5)):.3f}

TEST PERFORMANCE
----------------
Accuracy : {results['test_metrics']['Accuracy']:.4f}
Precision: {results['test_metrics']['Precision']:.4f}
Recall   : {results['test_metrics']['Recall']:.4f}
F1 Score : {results['test_metrics']['F1']:.4f}
ROC-AUC  : {results['test_metrics']['ROC-AUC']:.4f}
PR-AUC   : {results['test_metrics']['PR-AUC']:.4f}
Recall@FPR≤0.3%: {results['test_metrics']['Recall_at_FPR']:.4f}
Brier Score   : {results['test_metrics']['Brier_Score']:.4f}

Confusion Matrix:
TN: {int(results['test_metrics']['TN'])}  FP: {int(results['test_metrics']['FP'])}
FN: {int(results['test_metrics']['FN'])}  TP: {int(results['test_metrics']['TP'])}

===============================================================
""".strip(
        "\n"
    )

    print("\n" + report)
    out_txt = Path(output_dir) / "final_report.txt"
    out_txt.write_text(report, encoding="utf-8")
    print(f"\n✅ Report saved to {out_txt}")


def calculate_performance_at_different_thresholds(y_true, y_proba) -> pd.DataFrame:
    """Compute metrics on thresholds 0.1..0.9 (step 0.1)."""
    thresholds = np.arange(0.1, 1.0, 0.1, dtype=float)
    rows = []
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append(
            {
                "threshold": float(thr),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred),
                "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
                "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_calibration_curve(y_true, y_proba, n_bins: int = 10) -> Dict[str, Any]:
    """
    Return calibration curve values + ECE/MCE (no plotting here).
    Safe against degenerate inputs.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # sklearn قد يرمي خطأ لو القيم غير متنوعة
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    except Exception:
        # قِيَم افتراضية فارغة
        frac_pos, mean_pred = np.array([], dtype=float), np.array([], dtype=float)

    ece = expected_calibration_error(y_true, y_proba, n_bins)
    mce = maximum_calibration_error(y_true, y_proba, n_bins)

    return {
        "fraction_of_positives": frac_pos,
        "mean_predicted_value": mean_pred,
        "ece": float(ece),
        "mce": float(mce),
    }


def expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_true.size == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_proba > lo) & (y_proba <= hi)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_proba[mask].mean()
        ece += abs(conf - acc) * (mask.mean())
    return float(ece)


def maximum_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    """Maximum Calibration Error (MCE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_true.size == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_proba > lo) & (y_proba <= hi)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_proba[mask].mean()
        mce = max(mce, abs(conf - acc))
    return float(mce)


def create_model_card(model_info: Dict[str, Any], output_path: str) -> str:
    """Create a simple markdown model card and save to output_path."""
    training_time = float(
        model_info.get("training_time_seconds", model_info.get("training_time", 0.0))
    )
    card = f"""
# Model Card: Phishing Detection System

## Model Details
- **Model Type**: {model_info.get('best_model_name')}
- **Version**: 1.0.0
- **Date Trained**: {model_info.get('timestamp', '')}
- **Training Time**: {training_time:.2f} seconds
- **Optimal Threshold**: {float(model_info.get('best_threshold', 0.5)):.3f}

## Intended Use
- **Primary**: Detect phishing in emails/SMS/messages.
- **Users**: Security teams, email providers, enterprises.
- **Out-of-Scope**: Non-phishing cyber threats.

## Performance (Test)
- **Accuracy**: {model_info['test_metrics']['Accuracy']:.4f}
- **Precision**: {model_info['test_metrics']['Precision']:.4f}
- **Recall**: {model_info['test_metrics']['Recall']:.4f}
- **F1**: {model_info['test_metrics']['F1']:.4f}
- **ROC-AUC**: {model_info['test_metrics']['ROC-AUC']:.4f}
- **PR-AUC**: {model_info['test_metrics']['PR-AUC']:.4f}
- **Recall@FPR≤0.3%**: {model_info['test_metrics']['Recall_at_FPR']:.4f}

## Limitations
- New attack styles may degrade performance.
- Retrain periodically (≈ every 30 days).
- Performance may drop on non-English content.
- Susceptible to sophisticated adversarial attacks.

## Governance
- **Drift Monitoring**: Weekly
- **Performance Review**: Monthly
- **Security Audit**: Quarterly
- **Retraining**: 30 days

## Contact
- **Team**: Security ML Team
- **Email**: security-ml@company.com
    """.strip(
        "\n"
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card, encoding="utf-8")
    return card


# ================= Additional Utilities: Bootstrap CI =================


def _recall_at_fpr(y_true, y_proba, target_fpr: float = 0.003) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    mask = fpr <= target_fpr
    if not np.any(mask):
        return 0.0
    return float(tpr[mask][-1])


def _bootstrap_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo = np.quantile(values, alpha / 2)
    hi = np.quantile(values, 1 - alpha / 2)
    return float(lo), float(hi)


def bootstrap_metrics_summary(
    y_true,
    y_proba,
    thr: float,
    target_fpr: float = 0.003,
    n_boot: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Bootstrap CI for: ROC-AUC, PR-AUC, Recall@FPR, F1, Precision, Recall, Accuracy.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)

    auc_vals, pr_vals, r_fpr_vals, f1_vals, p_vals, r_vals, acc_vals = (
        [] for _ in range(7)
    )
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_proba[idx]
        yb = (yp >= thr).astype(int)

        # may fail on constant arrays → guard per metric
        try:
            auc_vals.append(roc_auc_score(yt, yp))
        except Exception:
            pass
        try:
            pr_vals.append(average_precision_score(yt, yp))
        except Exception:
            pass
        try:
            r_fpr_vals.append(_recall_at_fpr(yt, yp, target_fpr))
        except Exception:
            pass

        acc_vals.append(accuracy_score(yt, yb))
        p_vals.append(precision_score(yt, yb, zero_division=0))
        r_vals.append(recall_score(yt, yb))
        f1_vals.append(f1_score(yt, yb))

    def pack(arr):
        arr = np.array(arr, dtype=float)
        if arr.size == 0:
            return None
        lo, hi = _bootstrap_ci(arr)
        return {"mean": float(arr.mean()), "ci95": [lo, hi]}

    return {
        "ROC_AUC": pack(auc_vals),
        "PR_AUC": pack(pr_vals),
        "Recall_at_FPR": pack(r_fpr_vals),
        "Accuracy": pack(acc_vals),
        "Precision": pack(p_vals),
        "Recall": pack(r_vals),
        "F1": pack(f1_vals),
        "n_boot": int(n_boot),
    }


# ================================================================================================
# API Interface for Production (TF-IDF aware + Abstain Zone)
# ================================================================================================


class PhishingDetectionAPI:
    """
    Production inference API.
    - Supports numeric-feature models or TF-IDF feature_pipeline.pkl (if present).
    - Reads threshold/metadata from results.json (config_path).
    - Abstain zone: [threshold - band, threshold + band] → returns -1.
    """

    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
        config_path: str,
        feature_pipeline_path: Optional[str] = None,
    ):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.threshold: float = float(self.config.get("best_threshold", 0.5))
        self.abstain_band: float = float(
            os.getenv("ABSTAIN_BAND", str(self.config.get("abstain_band", 0.05)))
        )
        self.t_low: float = max(0.0, self.threshold - self.abstain_band)
        self.t_high: float = min(1.0, self.threshold + self.abstain_band)

        self.feature_pipeline = None
        if feature_pipeline_path and os.path.exists(feature_pipeline_path):
            self.feature_pipeline = joblib.load(feature_pipeline_path)

    # -------- convenience loader --------
    @classmethod
    def from_artifacts(cls, artifacts_dir: str) -> "PhishingDetectionAPI":
        """Load API from a directory produced by pipeline.save_artifacts()."""
        adir = Path(artifacts_dir)
        model = adir / "best_model.pkl"
        cleaner = adir / "data_cleaner.pkl"
        config = adir / "results.json"
        fp = adir / "feature_pipeline.pkl"
        return cls(
            str(model), str(cleaner), str(config), str(fp) if fp.exists() else None
        )

    # ---------- helpers ----------
    def _num_only(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.select_dtypes(include=[np.number, bool]).copy()
        if "label" in X.columns:
            X = X.drop(columns=["label"])
        return X

    def _feature_order(self) -> Optional[list[str]]:
        """Try to recover the training feature order for numeric fallback."""
        base = getattr(
            self.model, "base_estimator", getattr(self.model, "estimator", self.model)
        )
        if hasattr(base, "feature_names_in_"):
            return list(base.feature_names_in_)
        feat_order = getattr(self.preprocessor, "feature_stats", {}).get(
            "feature_order"
        )
        return list(feat_order) if feat_order is not None else None

    def _transform_features(self, data: pd.DataFrame):
        """Apply the same preprocessing/feature mapping used in training."""
        processed = self.preprocessor.clean_data(data.copy(), is_train=False)

        if self.feature_pipeline is not None:
            try:
                return self.feature_pipeline.transform(processed)
            except Exception:
                # fallback إلى numeric لو حصل عدم توافق
                pass

        X_num = self._num_only(processed)
        order = self._feature_order()
        if order is not None:
            X_num = X_num.reindex(columns=order, fill_value=0.0)
        return X_num

    def _proba(self, X) -> np.ndarray:
        """Robust probability getter (supports models without predict_proba)."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        if hasattr(self.model, "decision_function"):
            z = self.model.decision_function(X).astype(float)
            # logistic squashing
            return 1.0 / (1.0 + np.exp(-z))
        # worst-case: use predictions as hard probs
        return self.model.predict(X).astype(float)

    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict on a dataframe of raw features.
        Returns dictionary with predictions, probabilities, and threshold info.
        """
        X_feat = self._transform_features(data)
        probs = self._proba(X_feat)

        preds = np.where(
            probs <= self.t_low, 0, np.where(probs >= self.t_high, 1, -1)
        ).astype(int)
        return {
            "predictions": preds.tolist(),
            "probabilities": probs.astype(float).tolist(),
            "threshold_used": float(self.threshold),
            "abstain_band": float(self.abstain_band),
            "t_low": float(self.t_low),
            "t_high": float(self.t_high),
        }

    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        return self.predict(pd.DataFrame([features]))

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "model_loaded": bool(self.model is not None),
            "preprocessor_loaded": bool(self.preprocessor is not None),
            "feature_pipeline_loaded": bool(self.feature_pipeline is not None),
            "threshold": float(self.threshold),
            "abstain_band": float(self.abstain_band),
        }


# ================================================================================================
# Entry Point
# ================================================================================================

if __name__ == "__main__":
    main()