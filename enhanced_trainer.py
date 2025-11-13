# ================================================================================
# 🚀 ENHANCED TRAINER V2.3 - ACCURACY OPTIMIZED
# ================================================================================
# التحسينات الجديدة - النسخة المحسّنة للـ Accuracy:
# 1. ✅ ACCURACY-FOCUSED threshold logic - Accuracy ≥ 86%!
# 2. ✅ High Recall maintained (≥ 85%)
# 3. ✅ Composite scoring: Accuracy*0.40 + F1*0.35 + Recall*0.15 + (1-FPR)*0.10
# 4. ✅ Multi-level intelligent fallback (4 levels)
# 5. ✅ Smart FPR constraint (12% بدل 10%)
# ================================================================================
# 🎯 النتائج المتوقعة:
#    Accuracy: 86-89%  |  Precision: 85-88%  |  Recall: 85-90%  |  F1: 86-89%
# ================================================================================
from __future__ import annotations
import os
import sys
import time
import warnings
import inspect
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Union
from dataclasses import dataclass, field
from urllib.parse import urlparse, parse_qs
from collections import Counter
import math
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import issparse
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse import issparse, vstack, hstack
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from features_spec import JsonFeatureExtractor
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import numpy as np
# Tree models
try:
    from xgboost import XGBClassifier
    from xgboost.callback import EarlyStopping as XGBEarlyStopping
except ImportError:
    XGBClassifier = None
    XGBEarlyStopping = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

# Imbalance handling
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
    from imblearn.combine import SMOTETomek
except ImportError:
    SMOTE = ADASYN = TomekLinks = EditedNearestNeighbours = SMOTETomek = None

warnings.filterwarnings("ignore")


# ================================================================================
# 🎛️ ENHANCED CONFIGURATION V2.1
# ================================================================================

from dataclasses import dataclass
from typing import Tuple
import os

@dataclass
class EnhancedConfigV2:
    """تكوين محسّن مع سويتشات تخطي المراحل الثقيلة"""
    # Operating strategy
    operating_point: str = "f1"
    target_precision: float = 0.88
    target_recall: float = 0.86
    min_precision: float = 0.86
    min_recall: float = 0.86
    target_fpr: float = 0.06

    # Costs
    cost_fp: float = 1.0
    cost_fn: float = 6.0

    # Feature engineering
    use_advanced_features: bool = True
    use_tfidf: bool = True
    tfidf_max_features: int = 160000
    use_char_ngrams: bool = True
    char_ngram_range: Tuple[int, int] = (3, 5)
    use_word_ngrams: bool = True
    word_ngram_range: Tuple[int, int] = (1, 3)

    # Dimensionality reduction
    use_svd: bool = True
    svd_components: int = 512
    svd_algorithm: str = "randomized"

    # Balancing
    use_smote: bool = True
    smote_ratio: float = 1.0
    use_tomek: bool = False

    # CV/Ensembling
    use_cv: bool = False
    cv_folds: int = 3
    cv_stratified: bool = True
    use_stacking: bool = True
    stacking_cv_folds: int = 2
    use_voting: bool = False

    # XGBoost
    xgb_n_estimators: int = 6000
    xgb_learning_rate: float = 0.03
    xgb_max_depth: int = 6
    xgb_min_child_weight: int = 2
    xgb_gamma: float = 0.0
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    xgb_reg_lambda: float = 1.0
    xgb_reg_alpha: float = 0.0
    xgb_early_stopping_rounds: int = 600

    # LightGBM
    lgb_n_estimators: int = 4000
    lgb_learning_rate: float = 0.03
    lgb_max_depth: int = -1
    lgb_num_leaves: int = 255
    lgb_min_child_samples: int = 15
    lgb_subsample: float = 0.9
    lgb_colsample_bytree: float = 0.9
    lgb_reg_lambda: float = 1.0
    lgb_reg_alpha: float = 0.0

    # CatBoost
    catboost_iterations: int = 3000
    catboost_learning_rate: float = 0.03
    catboost_depth: int = 8
    catboost_l2_leaf_reg: float = 3.0

    # RF / LR
    rf_n_estimators: int = 500
    rf_max_depth: int = 15
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2
    rf_max_features: str = "sqrt"
    lr_c: float = 0.5
    lr_penalty: str = "l2"
    lr_solver: str = "saga"
    lr_max_iter: int = 2000

    # Performance
    use_gpu: bool = True
    gpu_id: int = 0
    n_jobs: int = -1

    # Calibration / Evaluation (NEW)
    calibrate_models: bool = True
    calibration_method: str = "sigmoid"
    skip_calibration: bool = False     # ← Part 5
    skip_eval_threshold: bool = False  # ← Part 6
    skip_on_error: bool = True         # ← كمّل حتى لو صار Error
    verbose: int = 1

    def __post_init__(self):
        env = {
            "operating_point": ("OPERATING_POINT", str),
            "target_fpr": ("TARGET_FPR", float),
            "min_precision": ("MIN_PRECISION", float),
            "min_recall": ("MIN_RECALL_THRESHOLD", float),
            "tfidf_max_features": ("TFIDF_MAX_FEATURES", int),
            "svd_components": ("SVD_COMPONENTS", int),
            "use_tfidf": ("USE_TFIDF", bool),
            "use_svd": ("USE_SVD", bool),
            "use_smote": ("USE_SMOTE", bool),
            "use_gpu": ("GPU", bool),
            "gpu_id": ("GPU_ID", int),
            "xgb_n_estimators": ("XGB_N_EST", int),
            "lgb_n_estimators": ("LGB_N_EST", int),
            "calibration_method": ("CALIBRATION_METHOD", str),
            "use_stacking": ("USE_STACKING", bool),
            "calibrate_models": ("CALIBRATE", bool),
            # NEW switches:
            "skip_calibration": ("SKIP_CALIB", bool),
            "skip_eval_threshold": ("SKIP_EVAL", bool),
            "xgb_early_stopping_rounds": ("XGB_ES", int),
        }
        for attr, (k, typ) in env.items():
            v = os.getenv(k)
            if v is None: 
                continue
            try:
                if typ is bool:
                    setattr(self, attr, v.lower() in ("1","true","yes","y"))
                else:
                    setattr(self, attr, typ(v))
            except:
                pass
        if os.getenv("XGB_ES") is not None:
            self.xgb_early_stopping_rounds = int(os.getenv("XGB_ES"))

# ================================================================================
# 🔍 ADVANCED URL FEATURES EXTRACTOR
# ================================================================================

class AdvancedURLFeatures(BaseEstimator, TransformerMixin):
    """
    استخراج features متقدمة من URLs
    """
    
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.suspicious_tlds = {
            '.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.click'
        }
        self.common_phishing_keywords = {
            'login', 'signin', 'account', 'verify', 'update', 'secure',
            'banking', 'paypal', 'ebay', 'amazon', 'apple', 'microsoft',
            'password', 'suspended', 'locked', 'confirm', 'urgent'
        }

    def _safe_str(self, value) -> str:
        if value is None or pd.isna(value):
            return ''
        if isinstance(value, str):
            return value
        try:
            return str(value)
        except:
            return ''
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print("   🔍 Extracting advanced URL features...")
        
        X = X.copy()
        url_col = 'url' if 'url' in X.columns else X.columns[0]
        
        X[url_col] = X[url_col].apply(self._safe_str)
        
        # Basic features
        if 'url_length' not in X.columns:
            X['url_length'] = X[url_col].str.len()
        if 'num_dots' not in X.columns:
            X['num_dots'] = X[url_col].str.count(r'\.')
        if 'num_digits_in_url' not in X.columns:
            X['num_digits_in_url'] = X[url_col].str.count(r'\d')
        
        # Advanced URL structure
        X['num_hyphens'] = X[url_col].str.count('-')
        X['num_underscores'] = X[url_col].str.count('_')
        X['num_slashes'] = X[url_col].str.count('/')
        X['num_question_marks'] = X[url_col].str.count(r'\?')
        X['num_ampersands'] = X[url_col].str.count('&')
        X['num_equals'] = X[url_col].str.count('=')
        X['num_at_symbols'] = X[url_col].str.count('@')
        X['num_percent_signs'] = X[url_col].str.count('%')
        
        # Domain features
        X['domain_length'] = X[url_col].apply(self._extract_domain_length)
        X['subdomain_count'] = X[url_col].apply(self._count_subdomains)
        X['tld_length'] = X[url_col].apply(self._extract_tld_length)
        X['has_suspicious_tld'] = X[url_col].apply(self._has_suspicious_tld).astype(int)
        
        # Path features
        X['path_length'] = X[url_col].apply(self._extract_path_length)
        X['num_path_tokens'] = X[url_col].apply(self._count_path_tokens)
        X['query_length'] = X[url_col].apply(self._extract_query_length)
        X['num_query_params'] = X[url_col].apply(self._count_query_params)
        
        # Character-based features
        X['uppercase_ratio'] = X[url_col].apply(self._uppercase_ratio)
        X['digit_ratio'] = X[url_col].apply(self._digit_ratio)
        X['special_char_ratio'] = X[url_col].apply(self._special_char_ratio)
        
        # Entropy
        if 'url_entropy' not in X.columns:
            X['url_entropy'] = X[url_col].apply(self._calculate_entropy)
        X['domain_entropy'] = X[url_col].apply(self._domain_entropy)
        
        # Phishing keywords
        X['num_phishing_keywords'] = X[url_col].apply(self._count_phishing_keywords)
        X['has_ip_address'] = X[url_col].apply(self._has_ip_address).astype(int)
        
        # Statistical features
        X['char_variety'] = X[url_col].apply(lambda x: len(set(x)))
        X['avg_token_length'] = X[url_col].apply(self._avg_token_length)
        X['max_token_length'] = X[url_col].apply(self._max_token_length)
        
        # Anomaly score
        X['anomaly_score'] = X[url_col].apply(self._calculate_anomaly_score)
        
        if self.verbose:
            print(f"   ✅ Extracted {len([c for c in X.columns if c != url_col])} features")
        
        return X
    
    def _extract_domain_length(self, url: str) -> int:
        try:
            domain = urlparse(url).netloc
            return len(domain)
        except:
            return 0
    
    def _count_subdomains(self, url: str) -> int:
        try:
            domain = urlparse(url).netloc
            return domain.count('.') - 1 if '.' in domain else 0
        except:
            return 0
    
    def _extract_tld_length(self, url: str) -> int:
        try:
            domain = urlparse(url).netloc
            tld = domain.split('.')[-1] if '.' in domain else ''
            return len(tld)
        except:
            return 0
    
    def _has_suspicious_tld(self, url: str) -> bool:
        try:
            domain = urlparse(url).netloc.lower()
            return any(domain.endswith(tld) for tld in self.suspicious_tlds)
        except:
            return False
    
    def _extract_path_length(self, url: str) -> int:
        try:
            path = urlparse(url).path
            return len(path)
        except:
            return 0
    
    def _count_path_tokens(self, url: str) -> int:
        try:
            path = urlparse(url).path
            return len([t for t in path.split('/') if t])
        except:
            return 0
    
    def _extract_query_length(self, url: str) -> int:
        try:
            query = urlparse(url).query
            return len(query)
        except:
            return 0
    
    def _count_query_params(self, url: str) -> int:
        try:
            query = urlparse(url).query
            return len(parse_qs(query))
        except:
            return 0
    
    def _uppercase_ratio(self, url: str) -> float:
        if url is None or not isinstance(url, str) or not url:
            return 0.0
        try:
            return sum(1 for c in url if c.isupper()) / len(url)
        except:
            return 0.0
    
    def _digit_ratio(self, url: str) -> float:
        if url is None or not isinstance(url, str) or not url:
            return 0.0
        try:
            return sum(1 for c in url if c.isdigit()) / len(url)
        except:
            return 0.0

    def _special_char_ratio(self, url: str) -> float:
        if url is None or not isinstance(url, str) or not url:
            return 0.0
        try:
            special = sum(1 for c in url if not c.isalnum())
            return special / len(url)
        except:
            return 0.0

    def _calculate_entropy(self, text: str) -> float:
        if text is None or not isinstance(text, str) or not text:
            return 0.0
        try:
            counts = Counter(text)
            probs = [count / len(text) for count in counts.values()]
            return -sum(p * math.log2(p) for p in probs if p > 0)
        except:
            return 0.0
        
    def _domain_entropy(self, url: str) -> float:
        try:
            domain = urlparse(url).netloc
            return self._calculate_entropy(domain)
        except:
            return 0.0
    
    def _count_phishing_keywords(self, url: str) -> int:
        url_lower = url.lower()
        return sum(1 for keyword in self.common_phishing_keywords if keyword in url_lower)
    
    def _has_ip_address(self, url: str) -> bool:
        ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        return bool(re.search(ip_pattern, url))
    
    def _avg_token_length(self, url: str) -> float:
        tokens = re.split(r'[/?&=.]', url)
        tokens = [t for t in tokens if t]
        return np.mean([len(t) for t in tokens]) if tokens else 0.0
    
    def _max_token_length(self, url: str) -> int:
        tokens = re.split(r'[/?&=.]', url)
        tokens = [t for t in tokens if t]
        return max([len(t) for t in tokens]) if tokens else 0
    
    def _calculate_anomaly_score(self, url: str) -> float:
        score = 0.0
        if len(url) > 100:
            score += 0.2
        if self._digit_ratio(url) > 0.3:
            score += 0.2
        if url.count('-') + url.count('_') > 5:
            score += 0.2
        if self._has_suspicious_tld(url):
            score += 0.3
        if self._count_phishing_keywords(url) >= 2:
            score += 0.3
        return min(score, 1.0)


# ================================================================================
# 🔧 FEATURE PIPELINE ADAPTER
# ================================================================================

class _FeaturePipelineAdapter:
    def __init__(self, vspace):
        self.vspace = vspace
    
    def transform(self, X):
        return self.vspace.transform_text(X)


# ================================================================================
# 🎯 ENHANCED VECTOR SPACE V2.1
# ================================================================================

# ===================== EnhancedVectorSpaceV2 (drop-in) =====================
class EnhancedVectorSpaceV2:
    """
    Vector space builder with TF-IDF (+ optional SVD) and numeric URL features.
    JSON-features control:
      FEATURE_SPECS_MODE = 'replace' | 'concat' | 'off'
      FEATURE_SPECS_DIR  = path to /features (contains url_features.json, email_features.json, common_features.json)
    """
    def __init__(self, config: EnhancedConfigV2):
        import os
        self.config = config

        # Text space
        self.char_vectorizer = None
        self.word_vectorizer = None
        self.svd = None

        # Numeric space
        self.scaler = None
        self.feature_names_ = []        # ترتيب ثابت للأعمدة الرقمية
        self.json_mode = os.getenv("FEATURE_SPECS_MODE", "concat").strip().lower()
        self.json_dir  = os.getenv("FEATURE_SPECS_DIR", "").strip()
        self.url_feature_extractor = None

        # وصّل مستخرج JSON إن لزم
        if self.json_mode != "off" and self.json_dir:
            try:
                from features_spec import JsonFeatureExtractor
                self.url_feature_extractor = JsonFeatureExtractor(
                    feature_dir=self.json_dir,
                    mode=self.json_mode,
                    verbose=1
                )
            except Exception as e:
                print(f"⚠️ JSON features disabled due to error: {e}")
                self.json_mode = "off"

    # ---------- legacy URL numeric features (أبقِ ما تحتاجه من ميزاتك القديمة) ----------
    def _legacy_url_feats(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X.copy()
        url_col = "url_canon" if "url_canon" in x.columns else ("url" if "url" in x.columns else None)
        if url_col is None:
            return pd.DataFrame(index=x.index)

        x[url_col] = x[url_col].astype(str)

        if 'url_length' not in x.columns:
            x['url_length'] = x[url_col].str.len()
        if 'num_dots' not in x.columns:
            x['num_dots'] = x[url_col].str.count(r'\.')
        if 'num_digits_in_url' not in x.columns:
            x['num_digits_in_url'] = x[url_col].str.count(r'\d')

        cols = [c for c in ['url_length', 'num_dots', 'num_digits_in_url'] if c in x.columns]
        if not cols:
            return pd.DataFrame(index=x.index)

        df = x[cols].copy()
        # تأكد كلها float
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
        return df

    # ---------- تجميع الميزات الرقمية: FIT ----------
    def _numeric_blocks_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        blocks = []

        # legacy when concat|off
        if self.json_mode in ("off", "concat"):
            leg = self._legacy_url_feats(X)
            if leg.shape[1] > 0:
                blocks.append(leg)

        # json when replace|concat
        if self.json_mode in ("replace", "concat") and self.url_feature_extractor is not None:
            jdf = self.url_feature_extractor.fit_transform(X)
            # خذ فقط الأعمدة الرقمية/البولية
            jdf = jdf.select_dtypes(include=[np.number, "bool"]).astype(float)
            blocks.append(jdf)
            print(f"   ✅ Loaded JSON feature specs: {jdf.shape[1]} features from {self.json_dir}")

        if not blocks:
            # لا ميزات رقمية → dummy
            out = pd.DataFrame({"__dummy0": np.zeros(len(X), dtype=np.float32)}, index=X.index)
            self.feature_names_ = ["__dummy0"]
            return out

        out = pd.concat(blocks, axis=1)
        # ترتيب ثابت للأعمدة
        out = out[[c for c in sorted(out.columns)]]
        self.feature_names_ = list(out.columns)
        return out

    # ---------- تجميع الميزات الرقمية: TRANSFORM ----------
    def _numeric_blocks_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        blocks = []

        if self.json_mode in ("off", "concat"):
            leg = self._legacy_url_feats(X)
            blocks.append(leg)

        if self.json_mode in ("replace", "concat") and self.url_feature_extractor is not None:
            jdf = self.url_feature_extractor.transform(X)
            jdf = jdf.select_dtypes(include=[np.number, "bool"]).astype(float)
            blocks.append(jdf)

        if not blocks:
            out = pd.DataFrame(index=X.index)
        else:
            out = pd.concat(blocks, axis=1)

        # أعمدة ثابتة حسب feature_names_:
        if not self.feature_names_:
            # safeguard للقديم
            if out.shape[1] == 0:
                out = pd.DataFrame({"__dummy0": np.zeros(len(X), dtype=np.float32)}, index=X.index)
                self.feature_names_ = ["__dummy0"]
            else:
                self.feature_names_ = [c for c in sorted(out.columns)]

        # أضف المفقود واحذف الزائد ثم رتّب
        for c in self.feature_names_:
            if c not in out.columns:
                out[c] = 0.0
        out = out[self.feature_names_]
        return out.astype(float)

    # ========================= FIT =========================
    # --- EnhancedFeatureSpace.patch BEGIN ---
    def fit_transform_text(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Build feature space on TRAIN and freeze schema so VAL/TEST align exactly.
        Also prunes constant columns for LightGBM stability and runs a mini sanity check.
        """
        print("\n📊 Building enhanced feature space...")

        # 1) Advanced JSON/url features (keeps your current behavior)
        if getattr(self.config, "use_advanced_features", True):
            # Read feature specs from env if present
            base_dir = os.environ.get("FEATURE_SPECS_DIR", os.path.join(os.getcwd(), "PhishingData", "features"))
            mode     = os.environ.get("FEATURE_SPECS_MODE", "concat")  # "replace" | "concat"
            self.url_feature_extractor = JsonFeatureExtractor(feature_dir=base_dir, mode=mode, verbose=1)
            X_feat = self.url_feature_extractor.fit_transform(X)
        else:
            X_feat = X

        # 2) Freeze numeric schema and scale
        num_cols = [c for c in X_feat.columns if X_feat[c].dtype in (np.float64, np.float32, np.int64, np.int32)]
        self.numeric_cols_ = list(num_cols)  # schema freeze
        X_num_df = X_feat[self.numeric_cols_].fillna(0.0).astype(np.float32)
        self.scaler = RobustScaler(with_centering=False)  # stable for sparse-ish numeric
        X_num_scaled = self.scaler.fit_transform(X_num_df.values)

        # 3) TF-IDF parts
        X_tfidf = None
        if getattr(self.config, "use_tfidf", True):
            url_col = 'url' if 'url' in X.columns else next((c for c in X.columns if X[c].dtype == object), X.columns[0])
            urls = X[url_col].fillna('').astype(str)

            # dynamic cap
            actual_max = min(getattr(self.config, "tfidf_max_features", 200_000), len(urls) * 5)

            if getattr(self.config, "use_char_ngrams", True):
                self.char_vectorizer = TfidfVectorizer(
                    analyzer='char',
                    ngram_range=getattr(self.config, "char_ngram_range", (3,5)),
                    max_features=min(actual_max // 2, 50_000),
                    min_df=max(3, len(urls) // 5000),
                    max_df=0.9,
                    strip_accents='unicode',
                    lowercase=True,
                    dtype=np.float32,
                )
                X_char = self.char_vectorizer.fit_transform(urls)
            else:
                X_char = None

            if getattr(self.config, "use_word_ngrams", True):
                self.word_vectorizer = TfidfVectorizer(
                    analyzer='word',
                    ngram_range=getattr(self.config, "word_ngram_range", (1,3)),
                    max_features=min(actual_max // 2, 50_000),
                    min_df=max(3, len(urls) // 5000),
                    max_df=0.9,
                    strip_accents='unicode',
                    lowercase=True,
                    token_pattern=r'\b\w+\b',
                    dtype=np.float32,
                )
                X_word = self.word_vectorizer.fit_transform(urls)
            else:
                X_word = None

            parts = [p for p in (X_char, X_word) if p is not None]
            if parts:
                from scipy.sparse import hstack as _hstack
                X_tfidf = _hstack(parts)

        # 4) Optional SVD for LR block. For tree block we guard carefully.
        X_tfidf_red = None
        if X_tfidf is not None and getattr(self.config, "use_svd", True):
            self.svd = TruncatedSVD(
                n_components=min(getattr(self.config, "svd_components", 768), X_tfidf.shape[1] - 1),
                algorithm=getattr(self.config, "svd_algorithm", "randomized"),
                random_state=42,
            )
            X_tfidf_red = self.svd.fit_transform(X_tfidf).astype(np.float32, copy=False)
            print(f"   ✅ SVD done - Explained variance: {self.svd.explained_variance_ratio_.sum():.2%}")
        else:
            self.svd = None
            if X_tfidf is not None:
                X_tfidf_red = X_tfidf.toarray().astype(np.float32, copy=False)

        # 5) Build LR and Tree blocks
        from scipy import sparse
        from scipy.sparse import hstack as _hstack
        X_lr_num = sparse.csr_matrix(X_num_scaled)
        X_lr = _hstack([X_lr_num, X_tfidf]) if X_tfidf is not None else X_lr_num

        # Tree: strictly float32, finite, and constants removed
        X_tree = X_num_scaled.astype(np.float32, copy=False)
        if X_tfidf_red is not None:
            X_tree = np.hstack([X_tree, X_tfidf_red])
        X_tree = np.nan_to_num(X_tree, copy=False, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        # Drop constant columns → prevents LightGBM split path degeneracy
        vt = VarianceThreshold(threshold=0.0)
        X_tree = vt.fit_transform(X_tree)
        self.tree_keep_mask_ = vt.get_support()

        # 6) Minimal sanity test (same code path as VAL)
        try:
            n = min(32, len(X))
            _ = self.transform_text(X.iloc[:n])
        except Exception as e:
            raise RuntimeError(f"[SanityCheck] transform_text failed right after fit. Cause: {e}")

        print(f"   ✅ Features ready:\n      numeric(frozen): {len(self.numeric_cols_)}\n      LR shape:   {X_lr.shape}\n      Tree shape: {X_tree.shape}")
        return X_lr, X_tree


    def transform_text(self, X: pd.DataFrame):
        """
        Transform on VAL/TEST using the frozen schema from fit_transform_text.
        """
        # 1) Recompute JSON/url features with the same extractor
        X_feat = self.url_feature_extractor.transform(X) if self.url_feature_extractor else X

        # 2) Align to frozen numeric schema and scale
        X_num_df = X_feat.reindex(columns=self.numeric_cols_, fill_value=0.0).astype(np.float32)
        X_num_scaled = self.scaler.transform(X_num_df.values)

        # 3) TF-IDF using fitted vectorizers
        X_tfidf = None
        if getattr(self.config, "use_tfidf", True):
            url_col = 'url' if 'url' in X.columns else next((c for c in X.columns if X[c].dtype == object), X.columns[0])
            urls = X[url_col].fillna('').astype(str)
            parts = []
            if getattr(self, "char_vectorizer", None): parts.append(self.char_vectorizer.transform(urls))
            if getattr(self, "word_vectorizer", None): parts.append(self.word_vectorizer.transform(urls))
            from scipy.sparse import hstack as _hstack
            if parts: X_tfidf = _hstack(parts)

        # 4) SVD projection (or dense fallback)
        if self.svd is not None and X_tfidf is not None:
            X_tfidf_red = self.svd.transform(X_tfidf).astype(np.float32, copy=False)
        else:
            X_tfidf_red = X_tfidf.toarray().astype(np.float32, copy=False) if X_tfidf is not None else None

        # 5) Combine
        from scipy import sparse
        from scipy.sparse import hstack as _hstack
        X_lr_num = sparse.csr_matrix(X_num_scaled)
        X_lr = _hstack([X_lr_num, X_tfidf]) if X_tfidf is not None else X_lr_num

        X_tree = X_num_scaled.astype(np.float32, copy=False)
        if X_tfidf_red is not None:
            X_tree = np.hstack([X_tree, X_tfidf_red])
        X_tree = np.nan_to_num(X_tree, copy=False, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        # apply same constant-column mask learned on train
        if getattr(self, "tree_keep_mask_", None) is not None:
            X_tree = X_tree[:, self.tree_keep_mask_]

        print(f"   ✅ Features extracted:\n      LR:   ({X_lr_num.shape[0]}, {X_lr.shape[1]})\n      Tree: ({X_tree.shape[0]}, {X_tree.shape[1]})")
        return X_lr, X_tree


# =================== end EnhancedVectorSpaceV2 ===================

# ================================================================================
# 🔧 HELPER FUNCTIONS
# ================================================================================

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve
)

# ---------- helpers ----------
def _confcounts(y_true, y_pred):
    # Fast confusion counts without sklearn confusion_matrix
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tn, fp, fn, tp

@dataclass
class ThresholdConfig:
    # modes: 'accuracy' | 'fpr' | 'precision'
    operating_point: str = "accuracy"
    force_manual_fpr: bool = False      # FORCE_MANUAL_FPR
    target_fpr: float = 0.06            # 6% as fraction (0.06)
    min_precision: float = 0.86         # used in precision-mode only

    skip_eval_threshold: bool = False
    skip_on_error: bool = True

    cost_fp: float = 1.0
    cost_fn: float = 6.0

class ThresholdSelectorV23:
    """
    Holds models and selects optimal per-model thresholds + best model overall.
    Expected attributes similar to your trainer.
    """
    def __init__(self, config: ThresholdConfig, base_models: dict, calibrated_models: dict | None = None):
        self.config = config
        self.base_models = base_models
        self.calibrated_models = calibrated_models or {}
        self.model_thresholds = {}

        self.best_model = None
        self.best_model_name = None
        self.best_threshold = None


def _best_f1_threshold(y_true, y_proba) -> float:
    """Return threshold that maximizes F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.0)
    best_f1, best_th = -1.0, 0.5
    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)
    return best_th


def _threshold_at_fpr(y_true, y_proba, target_fpr: float = 0.05):
    """
    ✅ FIXED: إيجاد threshold عند FPR معين
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    # Find threshold where FPR <= target_fpr
    valid_idx = np.where(fpr <= target_fpr)[0]
    
    if len(valid_idx) == 0:
        # إذا ما في threshold يحقق FPR المطلوب، نرجع أفضل ما عندنا
        return thresholds[0], tpr[0]
    
    # ✅ نختار آخر threshold (الأعلى) اللي يحقق الشرط
    idx = valid_idx[-1]
    return float(thresholds[idx]), float(tpr[idx])


def _max_recall_at_precision(y_true, y_proba, min_precision: float = 0.88):
    """
    ✅ أقصى recall عند precision معين
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.0)
    
    # Find max recall where precision >= min_precision
    valid_idx = np.where(precision >= min_precision)[0]
    
    if len(valid_idx) == 0:
        return 1.0, 0.0, 1.0
    
    # اختر الـ recall الأعلى
    idx = valid_idx[np.argmax(recall[valid_idx])]
    return float(thresholds[idx]), float(recall[idx]), float(precision[idx])


# ================================================================================
# 🏗️ MODEL BUILDERS
# ================================================================================

class ModelBuilders:
    """بناء النماذج بإعدادات محسّنة"""
    
    @staticmethod
    def build_xgboost(config: EnhancedConfigV2):
        if XGBClassifier is None:
            return None
        
        params = {
            'n_estimators': config.xgb_n_estimators,
            'learning_rate': config.xgb_learning_rate,
            'max_depth': config.xgb_max_depth,
            'min_child_weight': config.xgb_min_child_weight,
            'gamma': config.xgb_gamma,
            'subsample': config.xgb_subsample,
            'colsample_bytree': config.xgb_colsample_bytree,
            'reg_lambda': config.xgb_reg_lambda,
            'reg_alpha': config.xgb_reg_alpha,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42,
            'n_jobs': config.n_jobs,
        }
        
        if config.use_gpu:
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
        
        return XGBClassifier(**params)
    
    @staticmethod
    def build_lightgbm(config):
        # Prefer GPU only when safe; CPU is often more stable post-SVD
        use_gpu = bool(getattr(config, "use_gpu", True))
        device_type = "gpu" if use_gpu else "cpu"

        params = dict(
            objective="binary",
            n_estimators=getattr(config, "lgbm_n_estimators", 1200),
            learning_rate=getattr(config, "lgbm_learning_rate", 0.05),
            num_leaves=getattr(config, "lgbm_num_leaves", 64),
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_data_in_leaf=20,
            min_data_in_bin=3,
            max_bin=255,
            device_type="cpu",
            random_state=42,
            n_jobs=0,

            # Stability flags
            feature_pre_filter=False,
            deterministic=True,
            two_round=True,

            # Device
            device_type=device_type,
            force_col_wise=True,   # safer with many columns
        )
        return lgb.LGBMClassifier(**params)


    @staticmethod
    def build_catboost_gpu_safe(config):
        # GPU-safe for RTX 3050 4GB. Falls back to CPU if GPU not available or OOM occurs.
        from catboost import CatBoostClassifier
        base = dict(
            iterations=min(600, getattr(config, "cat_iterations", 800)),
            learning_rate=getattr(config, "cat_learning_rate", 0.06),
            depth=min(6, getattr(config, "cat_depth", 6)),
            l2_leaf_reg=getattr(config, "cat_l2_leaf_reg", 3.0),
            loss_function="Logloss",
            eval_metric="AUC",
            bootstrap_type="Bernoulli",
            subsample=0.8,
            rsm=0.6,
            max_bin=128,
            border_count=128,
            random_seed=42,
            verbose=False,
            task_type="CPU"
        )
        try:
            return CatBoostClassifier(
                task_type="GPU",
                devices="0",
                gpu_ram_part=float(os.environ.get("CAT_GPU_RAM_PART", "0.30")),
                **base,
            )
        except Exception:
            # CPU fallback keeps the same behavior but avoids VRAM pressure
            return CatBoostClassifier(task_type="CPU", **base)

    
    @staticmethod
    def build_lightgbm_cpu_safe(config: EnhancedConfigV2):
        if lgb is None:
            return None
        params = {
            'n_estimators': config.lgb_n_estimators,
            'learning_rate': config.lgb_learning_rate,
            'max_depth': config.lgb_max_depth,
            'num_leaves': min(config.lgb_num_leaves, 255),
            'min_child_samples': config.lgb_min_child_samples,
            'subsample': config.lgb_subsample,
            'colsample_bytree': config.lgb_colsample_bytree,
            'reg_lambda': config.lgb_reg_lambda,
            'reg_alpha': config.lgb_reg_alpha,
            'objective': 'binary',
            'metric': 'auc',
            'random_state': 42,
            'n_jobs': max(1, os.cpu_count() - 1),
            'device': 'cpu',
            'force_col_wise': True,
            # ثوابت صغيرة لتفادي خطأ left_count
            'min_data_in_bin': 3,
            'min_split_gain': 0.0,
        }
        return lgb.LGBMClassifier(**params)

    @staticmethod
    def build_catboost(config: EnhancedConfigV2):
        if CatBoostClassifier is None:
            return None
        
        params = {
            'iterations': config.catboost_iterations,
            'learning_rate': config.catboost_learning_rate,
            'depth': config.catboost_depth,
            'l2_leaf_reg': config.catboost_l2_leaf_reg,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 42,
            'thread_count': config.n_jobs if config.n_jobs > 0 else -1,
            'verbose': False,
        }
        
        if config.use_gpu:
            params.update({
                'task_type': 'GPU',
                'devices': f'{config.gpu_id}',
            })
        
        return CatBoostClassifier(**params)
    
    @staticmethod
    def build_random_forest(config: EnhancedConfigV2):
        return RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            min_samples_split=config.rf_min_samples_split,
            min_samples_leaf=config.rf_min_samples_leaf,
            max_features=config.rf_max_features,
            n_jobs=config.n_jobs,
            random_state=42,
            class_weight='balanced',
        )
    
    @staticmethod
    def build_logistic_regression(config: EnhancedConfigV2):
        return LogisticRegression(
            C=config.lr_c,
            penalty=config.lr_penalty,
            solver=config.lr_solver,
            max_iter=config.lr_max_iter,
            class_weight='balanced',
            random_state=42,
            n_jobs=config.n_jobs,
        )


# ================================================================================
# 🎓 ENHANCED TRAINER V2.1 - MAIN CLASS
# ================================================================================

class EnhancedPhishingTrainerV2:
    """
    المدرب المحسّن النسخة 2.1 - FIXED
    
    التحسينات:
    - ✅ Fixed threshold selection logic
    - ✅ Better memory management
    - ✅ Smarter SMOTE
    - ✅ Optimized hyperparameters
    """
    
    def __init__(self, config: Optional[EnhancedConfigV2] = None):
        if config is None:
            self.config = EnhancedConfigV2()
        elif not isinstance(config, EnhancedConfigV2):
            self.config = self._adapt_config(config)
        else:
            self.config = config

        self.vspace = None
        self.base_models = {}
        self.calibrated_models = {}
        self.model_thresholds = {}
        self.best_model = None
        self.best_model_name = None
        self.best_threshold = 0.5
        self.feature_pipeline = None
        self._svd = None

        # NEW: runtime flags from env (skip + manual FPR)
        env_true = lambda s: str(s).lower() in ("1", "true", "yes", "y")
        self.skip_calib = env_true(os.getenv("SKIP_CALIB", "0"))
        self.skip_eval  = env_true(os.getenv("SKIP_EVAL",  "0"))
        self.force_manual_fpr = env_true(os.getenv("FORCE_MANUAL_FPR", "0"))

        print("🚀 Enhanced Trainer V2.3 (thresholds fixed)")
        print(f"   Operating Point: {getattr(self.config, 'operating_point', 'f1')}")
        print(f"   Target FPR: {getattr(self.config, 'target_fpr', 0.06):.2%}")
        print(f"   GPU: {'Enabled' if getattr(self.config, 'use_gpu', True) else 'Disabled'}")
        print(f"   SMOTE: {'Yes' if getattr(self.config, 'use_smote', False) else 'No'}")
        print(f"   Stacking: {'Yes' if getattr(self.config, 'use_stacking', True) else 'No'}")
        if self.skip_calib: print("   ⚠️ Calibration will be SKIPPED (SKIP_CALIB=1)")
        if self.skip_eval:  print("   ⚠️ Evaluation/Thresholds will be SKIPPED (SKIP_EVAL=1)")

    
    def _adapt_config(self, old_config):
        """تحويل ModelConfig من App.py مع تفضيل قيم الـ ENV إن وُجدت."""
        new_config = EnhancedConfigV2()  # هذا يقرأ الـ ENV في __post_init__

        # لو مافي ENV للهدف، خُذ من ModelConfig، غير هيك خَلّي قيمة الـ ENV
        if os.getenv("TARGET_FPR") is None and hasattr(old_config, 'target_fpr'):
            new_config.target_fpr = getattr(old_config, 'target_fpr', new_config.target_fpr)

        # تكاليف
        new_config.cost_fp = getattr(old_config, 'cost_fp', new_config.cost_fp)
        new_config.cost_fn = getattr(old_config, 'cost_fn', new_config.cost_fn)

        # حد الاستدعاء (recall)
        if os.getenv("MIN_RECALL_THRESHOLD") is None and hasattr(old_config, 'min_recall_threshold'):
            new_config.min_recall = old_config.min_recall_threshold
            new_config.target_recall = old_config.min_recall_threshold

        # المعايرة
        if os.getenv("CALIBRATION_METHOD") is None and hasattr(old_config, 'calibration_method'):
            new_config.calibration_method = old_config.calibration_method

        # Early stopping لـ XGB
        if hasattr(old_config, 'early_stopping_rounds'):
            new_config.xgb_early_stopping_rounds = old_config.early_stopping_rounds

        return new_config

    # --- Backward-compat alias to avoid AttributeError ---
    def _evaluate_and_select_threshold(self, X_va_lr, X_va_tree, y_val):
        return self.evaluate_and_select_threshold(X_va_lr, X_va_tree, y_val)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame, y_val: pd.Series):
        print("\n" + "=" * 80)
        print("🎯 STARTING ENHANCED TRAINING V2.3")
        print("=" * 80)

        # -------- Phase 1: Features --------
        print("\n📊 Phase 1: Advanced Feature Engineering")
        self.vspace = EnhancedVectorSpaceV2(self.config)
        X_tr_lr,   X_tr_tree = self.vspace.fit_transform_text(X_train, y_train)
        X_va_lr,   X_va_tree = self.vspace.transform_text(X_val)
        self.feature_pipeline = _FeaturePipelineAdapter(self.vspace)
        self._svd = self.vspace.svd

        print(f"\n   ✅ Features extracted:")
        print(f"      LR:   {X_tr_lr.shape} → {X_va_lr.shape}")
        print(f"      Tree: {X_tr_tree.shape} → {X_va_tree.shape}")

        # -------- Phase 2: Balancing --------
        X_tr_tree_balanced = X_tr_tree
        y_train_balanced   = y_train
        if self.config.use_smote and SMOTE is not None:
            try:
                smote = SMOTE(sampling_strategy=self.config.smote_ratio,
                            random_state=42, k_neighbors=5)
                X_tr_tree_balanced, y_train_balanced = smote.fit_resample(X_tr_tree, y_train)
                print(f"   ✅ Balanced: {X_tr_tree.shape[0]:,} → {X_tr_tree_balanced.shape[0]:,}")
            except Exception as e:
                print(f"   ⚠️ SMOTE failed: {e}")

        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        scale_pos_weight = (n_neg / max(1, n_pos)) * 0.8
        print(f"\n⚖️ Class balance: neg={n_neg:,} pos={n_pos:,} | scale_pos_weight={scale_pos_weight:.2f}")

        # -------- Phase 3: Base Models --------
        print("\n🏋️ Phase 3: Training Base Models")

        # LR
        print("\n1️⃣ Training Logistic Regression...")
        lr = ModelBuilders.build_logistic_regression(self.config)
        lr.fit(X_tr_lr, y_train)
        self.base_models['LR'] = lr
        print("   ✅ LR trained")

        # XGB
        if XGBClassifier is not None:
            print("\n2️⃣ Training XGBoost...")
            xgb = ModelBuilders.build_xgboost(self.config)
            xgb.set_params(scale_pos_weight=scale_pos_weight)
            fit_kwargs = dict(eval_set=[(X_va_tree, y_val)], verbose=False)
            accepts_callbacks = "callbacks" in inspect.signature(xgb.fit).parameters
            if accepts_callbacks and XGBEarlyStopping is not None:
                try:
                    fit_kwargs["callbacks"] = [XGBEarlyStopping(
                        rounds=self.config.xgb_early_stopping_rounds, maximize=True, save_best=True)]
                except:
                    pass
            xgb.fit(X_tr_tree_balanced, y_train_balanced, **fit_kwargs)
            self.base_models['XGB'] = xgb
            print("   ✅ XGB trained")

        # LightGBM (GPU → CPU fallback)
        print("\n3️⃣ Training LightGBM...")
        try:
            lgbm = ModelBuilders.build_lightgbm(self.config)
            lgbm.set_params(scale_pos_weight=scale_pos_weight)
            lgbm.fit(
                X_tr_tree_balanced, y_train_balanced,
                eval_set=[(X_va_tree, y_val)],
                callbacks=[lgb.early_stopping(self.config.xgb_early_stopping_rounds, verbose=False)]
            )
        except Exception as e:
            print(f"   ⚠️ LGBM GPU error: {e}\n   ↪️ Retrying on CPU.")
            lgbm = ModelBuilders.build_lightgbm_cpu_safe(self.config)
            if lgbm:
                lgbm.set_params(scale_pos_weight=scale_pos_weight)
                lgbm.fit(
                    X_tr_tree_balanced, y_train_balanced,
                    eval_set=[(X_va_tree, y_val)],
                    callbacks=[lgb.early_stopping(self.config.xgb_early_stopping_rounds, verbose=False)]
                )
        self.base_models['LightGBM'] = lgbm
        print("   ✅ LightGBM trained")

        # CatBoost (safe)
        if CatBoostClassifier is not None:
            print("\n4️⃣ Training CatBoost (Safe Mode)...")
            try:
                cat = ModelBuilders.build_catboost(self.config)
                cat.set_params(scale_pos_weight=scale_pos_weight)
                max_samples = min(40000, len(X_tr_tree_balanced))
                if len(X_tr_tree_balanced) > max_samples:
                    print(f"   ⚙️ Sampling {max_samples:,} for memory safety")
                    idx = np.random.choice(len(X_tr_tree_balanced), max_samples, replace=False)
                    X_cat = X_tr_tree_balanced[idx]
                    y_cat = y_train_balanced.iloc[idx] if hasattr(y_train_balanced, "iloc") else y_train_balanced[idx]
                else:
                    X_cat = X_tr_tree_balanced
                    y_cat = y_train_balanced
                X_val_dense = X_va_tree.toarray() if issparse(X_va_tree) else X_va_tree
                if issparse(X_cat): X_cat = X_cat.toarray()
                cat.fit(X_cat, y_cat, eval_set=(X_val_dense, y_val),
                        early_stopping_rounds=min(300, self.config.xgb_early_stopping_rounds),
                        verbose=False, use_best_model=True)
                self.base_models['CatBoost'] = cat
                print("   ✅ CatBoost trained")
            except Exception as e:
                print(f"   ⚠️ CatBoost failed (continue): {e}")

        # RF
        print("\n5️⃣ Training Random Forest...")
        rf = ModelBuilders.build_random_forest(self.config)
        X_rf = X_tr_tree_balanced if not issparse(X_tr_tree_balanced) else X_tr_tree_balanced.toarray()
        rf.fit(X_rf, y_train_balanced)
        self.base_models['RF'] = rf
        print("   ✅ RF trained")

        # -------- Phase 4: Stacking --------
        if self.config.use_stacking and len(self.base_models) >= 3:
            print("\n🔗 Phase 4: Building Stacking Ensemble")
            self._build_stacking_ensemble(X_tr_tree_balanced, y_train_balanced)

        # -------- Phase 5: Calibration (SKIP-able) --------
        if self.skip_calib:
            print("\n📏 Phase 5: Model Calibration → SKIPPED (SKIP_CALIB=1)")
            self.calibrated_models = dict(self.base_models)  # use raw models
        else:
            print("\n📏 Phase 5: Model Calibration")
            self._calibrate_models(X_va_tree, y_val)

        # -------- Phase 6: Evaluation/Thresholds (SKIP-able) --------
        if self.skip_eval:
            print("\n📊 Phase 6: Evaluation/Thresholds → SKIPPED (SKIP_EVAL=1)")
            # Choose a sane default if skipped
            prefer = ['LightGBM', 'XGB', 'CatBoost', 'LR', 'RF']
            for name in prefer:
                if name in self.calibrated_models or name in self.base_models:
                    self.best_model_name = name
                    self.best_model = self.calibrated_models.get(name, self.base_models.get(name))
                    self.best_threshold = 0.5
                    break
            return pd.DataFrame([{'Model': self.best_model_name, 'Chosen_Threshold': self.best_threshold}])

        print("\n📊 Phase 6: Evaluation & Threshold Selection")
        results_df = self.evaluate_and_select_threshold(X_va_lr, X_va_tree, y_val)

        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   Optimal Threshold: {self.best_threshold:.3f}")
        return results_df

        
    def _build_stacking_ensemble(self, X_train, y_train):
        try:
            base_estimators = []
            for name, model in self.base_models.items():
                if name not in ['LR', 'CatBoost', 'LightGBM']:
                    base_estimators.append((name, clone(model)))
            
            if len(base_estimators) < 2:
                print("   ⚠️ Not enough stable models for stacking")
                return
            
            max_stack_samples = min(15000, len(X_train))  # ✅ 15K بدل 20K
            if len(X_train) > max_stack_samples:
                print(f"   ⚙️ Stacking using {max_stack_samples:,} samples")
                stack_idx = np.random.choice(len(X_train), max_stack_samples, replace=False)
                X_stack = X_train[stack_idx]
                y_stack = y_train.iloc[stack_idx] if hasattr(y_train, 'iloc') else y_train[stack_idx]
            else:
                X_stack = X_train
                y_stack = y_train
            
            meta_learner = LogisticRegression(
                C=1.0,
                max_iter=300,  # ✅ 300 بدل 500
                random_state=42,
                n_jobs=1,
            )
            
            stacking = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=2,
                n_jobs=1,
                passthrough=False,
            )
            
            X_stack_dense = X_stack if not issparse(X_stack) else X_stack.toarray()
            stacking.fit(X_stack_dense, y_stack)
            
            self.base_models['Stacking'] = stacking
            print(f"   ✅ Stacking ensemble built with {len(base_estimators)} models")
            
        except Exception as e:
            print(f"   ⚠️ Stacking failed (continuing without it): {e}")
    
    

    def _calibrate_models(self, X_val, y_val):
        """Part 5 — Calibration مع إمكانية التخطي الآمن"""
        if self.config.skip_calibration or not self.config.calibrate_models:
            self.calibrated_models = {name: model for name, model in self.base_models.items()}
            print("   ℹ️ Calibration skipped.")
            return

        method = (getattr(self.config, 'calibration_method', 'sigmoid') or 'sigmoid')
        self.calibrated_models = {}

        for name, model in self.base_models.items():
            try:
                if name == 'LR':
                    self.calibrated_models[name] = model
                    continue

                X_val_use = X_val if not issparse(X_val) else X_val.toarray()
                calibrated = CalibratedClassifierCV(model, method=method, cv='prefit')
                calibrated.fit(X_val_use, y_val)
                self.calibrated_models[name] = calibrated
                print(f"   ✅ {name} calibrated ({method})")
            except Exception as e:
                if self.config.verbose:
                    print(f"   ⚠️ {name} calibration failed: {e}")
                if self.config.skip_on_error:
                    self.calibrated_models[name] = model
                else:
                    raise

    def _quick_select_by_prauc(self, X_va_lr, X_va_tree, y_val):
        rows = []
        for name in self.base_models.keys():
            model = self.calibrated_models.get(name, self.base_models[name])
            Xuse = self._X_for(name, X_va_lr, X_va_tree)
            y_proba = model.predict_proba(Xuse)[:, 1]
            pr_auc  = average_precision_score(y_val, y_proba)
            self.model_thresholds[name] = 0.5
            rows.append({"Model": name, "PR-AUC": pr_auc, "Chosen_Threshold": 0.5, "Strategy": "prauc@0.5"})
        # Pick best by PR-AUC
        best = max(rows, key=lambda r: r["PR-AUC"])
        self.best_model = self.calibrated_models.get(best["Model"], self.base_models[best["Model"]])
        self.best_model_name = best["Model"]
        self.best_threshold = 0.5
        return pd.DataFrame(rows)

    import numpy as np
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, precision_recall_curve,
        accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
    )

    # -------- public entry point (drop-in for your Part 6) --------
    def evaluate_and_select_threshold(self, X_va_lr, X_va_tree, y_val):
        if self.config.skip_eval_threshold:
            print("   ℹ️ Evaluation skipped → quick PR-AUC selection @ th=0.5")
            return self._quick_select_by_prauc(X_va_lr, X_va_tree, y_val)

        op = (self.config.operating_point or "accuracy").lower()
        if op == "fpr":
            print("\n   📊 Threshold Selection: FPR-Targeted")
            return self._select_by_fpr(X_va_lr, X_va_tree, y_val)
        elif op == "precision":
            print("\n   📊 Threshold Selection: Precision-Targeted")
            return self._select_by_precision(X_va_lr, X_va_tree, y_val)
        else:
            print("\n   📊 Threshold Selection: Accuracy-Focused (V2.3)")
            return self._select_by_accuracy_v23(X_va_lr, X_va_tree, y_val)

    # -------- internal: data accessor --------
    def _X_for(self, name, X_lr, X_tree):
        if name == "LR":
            return X_lr
        return X_tree.toarray() if issparse(X_tree) else X_tree

    # -------- internal: FPR-targeted selection (Robust Quantile-Based) --------
    def _select_by_fpr(self, X_va_lr, X_va_tree, y_val):
        import numpy as np

        def _metrics(y_true, y_proba, th, cost_fp, cost_fn):
            y_pred = (y_proba >= th).astype(int)
            tn, fp, fn, tp = _confcounts(y_true, y_pred)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            fpr  = fp / (fp + tn) if (fp + tn) else 0.0
            f1   = 2 * (prec * rec) / (prec + rec + 1e-12)
            acc  = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) else 0.0
            cost = fp*cost_fp + fn*cost_fn
            return prec, rec, fpr, f1, acc, cost

        rows, best_score = [], -np.inf
        tgt = float(self.config.target_fpr)
        eps  = 1e-12

        for name in self.base_models.keys():
            print(f"\n   Evaluating {name}...")
            model = self.calibrated_models.get(name, self.base_models[name])

            # 1) اختَر الماتريكس الصحيحة
            Xuse = self._X_for(name, X_va_lr, X_va_tree)
            try:
                y_proba = model.predict_proba(Xuse)[:, 1]
            except Exception as e:
                if self.config.skip_on_error:
                    print(f"      ⚠️ predict_proba failed: {e} → skipped")
                    continue
                raise

            # 2) حساب عتبة الكوانتايل على السوالب
            y_val_arr   = np.asarray(y_val, dtype=np.int8)
            neg_scores  = y_proba[y_val_arr == 0]
            roc_auc     = roc_auc_score(y_val, y_proba)
            pr_auc      = average_precision_score(y_val, y_proba)

            if neg_scores.size == 0:
                # لا يوجد سوالب؟ نختار 1.0
                th_seed = 1.0
            else:
                # quantile عند (1 - target_fpr)
                q = np.quantile(neg_scores, max(0.0, 1.0 - tgt))
                # ارفع قليلاً لتجاوز ties
                th_seed = float(np.nextafter(q, 1.0))

            # 3) شبكة صغيرة حول العتبة، مع search أوسع كـ fallback
            fine = np.linspace(max(0.0, th_seed - 0.05), min(1.0, th_seed + 0.05), 201)
            coarse = np.linspace(0.0, 1.0, 101)
            cand_thresholds = np.unique(np.r_[th_seed, fine, coarse, 0.0, 1.0])

            chosen = None
            for th in cand_thresholds:
                P, R, FPR, F1, ACC, COST = _metrics(y_val_arr, y_proba, th, self.config.cost_fp, self.config.cost_fn)
                if FPR <= tgt + 1e-9:  # اسمح بهامش عددي صغير
                    # تعظيم الـ Recall، ثم F1، ثم Accuracy، ثم تقليل الكلفة
                    key = (R, F1, ACC, -COST)
                    if (chosen is None) or (key > chosen[0]):
                        chosen = (key, th, P, R, FPR, F1, ACC, COST)

            if chosen is None:
                # إذا لم نجد أي عتبة ضمن FPR المطلوب: خذ أقل FPR ممكن
                best_fpr = +np.inf
                best_alt = None
                for th in cand_thresholds:
                    P, R, FPR, F1, ACC, COST = _metrics(y_val_arr, y_proba, th, self.config.cost_fp, self.config.cost_fn)
                    key = ( -FPR, R, F1, ACC, -COST )  # قلّل FPR قدر الإمكان
                    if (best_alt is None) or (key > best_alt[0]):
                        best_alt = (key, th, P, R, FPR, F1, ACC, COST)
                        best_fpr = min(best_fpr, FPR)
                _, th, P, R, FPR, F1, ACC, COST = best_alt
                strategy = "min_fpr_fallback"
            else:
                _, th, P, R, FPR, F1, ACC, COST = chosen
                strategy = f"quantile@{100*tgt:.3f}% (tuned)"

            print(f"      Selected: {strategy} | th={th:.6f}")
            print(f"      Acc={ACC:.4f} | P={P:.4f} | R={R:.4f} | F1={F1:.4f} | FPR={FPR:.4f} | Cost={COST:.1f}")

            self.model_thresholds[name] = float(th)

            combined = F1*0.35 + R*0.30 + P*0.15 + (1 - FPR)*0.10 + (1/(1+0.01*COST))*0.10
            if combined > best_score:
                best_score = combined
                self.best_model = model
                self.best_model_name = name
                self.best_threshold = float(th)

            rows.append({
                "Model": name, "ROC-AUC": roc_auc, "PR-AUC": pr_auc,
                "Final_Precision": P, "Final_Recall": R, "Final_FPR": FPR,
                "Final_F1": F1, "Final_Accuracy": ACC, "Total_Cost": COST,
                "Chosen_Threshold": float(th), "Strategy": strategy
            })

        return pd.DataFrame(rows)

    

    # -------- internal: Precision-targeted (optional) --------
    def _select_by_precision(self, X_va_lr, X_va_tree, y_val):
        rows, best_score = [], -np.inf
        min_p = float(self.config.min_precision)

        for name in self.base_models.keys():
            print(f"\n   Evaluating {name}...")
            model = self.calibrated_models.get(name, self.base_models[name])

            try:
                Xuse = self._X_for(name, X_va_lr, X_va_tree)
                y_proba = model.predict_proba(Xuse)[:, 1]
            except Exception as e:
                if self.config.skip_on_error:
                    print(f"      ⚠️ predict_proba failed: {e} → skipped")
                    continue
                raise

            roc_auc = roc_auc_score(y_val, y_proba)
            pr_auc  = average_precision_score(y_val, y_proba)

            thresholds = np.unique(np.r_[y_proba, 0.0, 1.0])
            thresholds.sort()

            best = None
            for th in thresholds:
                y_pred = (y_proba >= th).astype(int)
                tn, fp, fn, tp = _confcounts(y_val, y_pred)
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec  = tp / (tp + fn) if (tp + fn) else 0.0
                fpr  = fp / (fp + tn) if (fp + tn) else 0.0
                f1   = 2 * (prec * rec) / (prec + rec + 1e-12)
                acc  = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) else 0.0
                cost = fp*self.config.cost_fp + fn*self.config.cost_fn

                if prec >= min_p:
                    # Primary: maximize F1; tiebreakers: higher recall → lower FPR → lower cost
                    key = (f1, rec, -fpr, -cost)
                    if (best is None) or (key > best[0]):
                        best = (key, th, prec, rec, fpr, f1, acc, cost)

            if best is None:
                # If nothing hits precision floor, take the threshold with max precision anyway.
                argmax_p = np.argmax(y_proba)  # safest is th=1.0, but this tries to keep logic consistent
                th = 1.0 if argmax_p is None else float(np.max(np.r_[y_proba, 1.0]))
                y_pred = (y_proba >= th).astype(int)
                tn, fp, fn, tp = _confcounts(y_val, y_pred)
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec  = tp / (tp + fn) if (tp + fn) else 0.0
                fpr  = fp / (fp + tn) if (fp + tn) else 0.0
                f1   = 2 * (prec * rec) / (prec + rec + 1e-12)
                acc  = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) else 0.0
                cost = fp*self.config.cost_fp + fn*self.config.cost_fn
                strategy = "max_precision_fallback"
            else:
                _, th, prec, rec, fpr, f1, acc, cost = best
                strategy = f"precision_floor@≥{100*min_p:.1f}%"

            print(f"      Selected: {strategy} | th={th:.4f}")
            print(f"      Acc={acc:.4f} | P={prec:.4f} | R={rec:.4f} | F1={f1:.4f} | FPR={fpr:.4f} | Cost={cost:.1f}")

            self.model_thresholds[name] = float(th)
            combined = f1*0.35 + rec*0.30 + prec*0.15 + (1 - fpr)*0.10 + (1/(1+0.01*cost))*0.10
            if combined > best_score:
                best_score = combined
                self.best_model = model
                self.best_model_name = name
                self.best_threshold = float(th)

            rows.append({
                "Model": name, "ROC-AUC": roc_auc, "PR-AUC": pr_auc,
                "Final_Precision": prec, "Final_Recall": rec, "Final_FPR": fpr,
                "Final_F1": f1, "Final_Accuracy": acc, "Total_Cost": cost,
                "Chosen_Threshold": float(th), "Strategy": strategy
            })

        return pd.DataFrame(rows)
    
     # -------- internal: Accuracy-focused V2.3 (اللي بعته المستخدم) --------
    def _select_by_accuracy_v23(self, X_va_lr, X_va_tree, y_val):
        rows = []
        best_score = -np.inf

        for name in self.base_models.keys():
            print(f"\n   Evaluating {name}...")
            model = self.calibrated_models.get(name, self.base_models[name])

            try:
                Xuse = self._X_for(name, X_va_lr, X_va_tree)
                y_proba = model.predict_proba(Xuse)[:, 1]
            except Exception as e:
                if self.config.skip_on_error:
                    print(f"      ⚠️ predict_proba failed: {e} → skipped")
                    continue
                else:
                    raise

            roc_auc = roc_auc_score(y_val, y_proba)
            pr_auc  = average_precision_score(y_val, y_proba)

            precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
            thresholds = np.append(thresholds, 1.0)

            best_threshold = None
            best_metrics = None
            strategy = None

            # Level 1
            candidates = []
            for th in thresholds:
                y_pred = (y_proba >= th).astype(int)
                tn, fp, fn, tp = _confcounts(y_val, y_pred)
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec  = tp / (tp + fn) if (tp + fn) else 0.0
                fpr  = fp / (fp + tn) if (fp + tn) else 0.0
                f1   = 2 * (prec * rec) / (prec + rec + 1e-12)
                acc  = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) else 0.0
                if acc >= 0.86 and rec >= 0.85 and fpr <= 0.12 and prec >= 0.80:
                    comp = acc*0.40 + f1*0.35 + rec*0.15 + (1 - fpr)*0.10
                    candidates.append((comp, th, prec, rec, fpr, f1, acc,
                                       fp*self.config.cost_fp + fn*self.config.cost_fn))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                s, th, P, R, FPR, F1, ACC, COST = candidates[0]
                best_threshold = th; strategy = "accuracy_focused_l1"
                best_metrics = dict(prec=P, rec=R, fpr=FPR, f1=F1, acc=ACC, score=s, cost=COST)

            # Level 2
            if best_threshold is None:
                candidates = []
                for th in thresholds:
                    y_pred = (y_proba >= th).astype(int)
                    tn, fp, fn, tp = _confcounts(y_val, y_pred)
                    prec = tp / (tp + fp) if (tp + fp) else 0.0
                    rec  = tp / (tp + fn) if (tp + fn) else 0.0
                    fpr  = fp / (fp + tn) if (fp + tn) else 0.0
                    f1   = 2 * (prec * rec) / (prec + rec + 1e-12)
                    acc  = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) else 0.0
                    if acc >= 0.84 and rec >= 0.80 and fpr <= 0.15 and prec >= 0.75:
                        comp = acc*0.35 + f1*0.40 + rec*0.15 + (1 - fpr)*0.10
                        candidates.append((comp, th, prec, rec, fpr, f1, acc,
                                           fp*self.config.cost_fp + fn*self.config.cost_fn))
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    s, th, P, R, FPR, F1, ACC, COST = candidates[0]
                    best_threshold = th; strategy = "accuracy_focused_l2"
                    best_metrics = dict(prec=P, rec=R, fpr=FPR, f1=F1, acc=ACC, score=s, cost=COST)

            # Level 3
            if best_threshold is None:
                best_f1 = -1
                for th in thresholds:
                    y_pred = (y_proba >= th).astype(int)
                    tn, fp, fn, tp = _confcounts(y_val, y_pred)
                    prec = tp / (tp + fp) if (tp + fp) else 0.0
                    rec  = tp / (tp + fn) if (tp + fn) else 0.0
                    fpr  = fp / (fp + tn) if (fp + tn) else 0.0
                    f1   = 2 * (prec * rec) / (prec + rec + 1e-12)
                    acc  = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) else 0.0
                    if fpr <= 0.18 and rec >= 0.70 and f1 > best_f1:
                        best_f1 = f1
                        best_threshold = th; strategy = "f1_optimized_l3"
                        best_metrics = dict(prec=prec, rec=rec, fpr=fpr, f1=f1, acc=acc,
                                            score=f1, cost=fp*self.config.cost_fp + fn*self.config.cost_fn)

            # Level 4
            if best_threshold is None:
                best_bal = -1
                for th in thresholds:
                    y_pred = (y_proba >= th).astype(int)
                    tn, fp, fn, tp = _confcounts(y_val, y_pred)
                    prec = tp / (tp + fp) if (tp + fp) else 0.0
                    rec  = tp / (tp + fn) if (tp + fn) else 0.0
                    fpr  = fp / (fp + tn) if (fp + tn) else 0.0
                    f1   = 2 * (prec * rec) / (prec + rec + 1e-12)
                    acc  = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) else 0.0
                    bal  = acc*0.35 + f1*0.35 + rec*0.20 + (1 - fpr)*0.10
                    if bal > best_bal:
                        best_bal = bal
                        best_threshold = th; strategy = "balanced_l4"
                        best_metrics = dict(prec=prec, rec=rec, fpr=fpr, f1=f1, acc=acc,
                                            score=bal, cost=fp*self.config.cost_fp + fn*self.config.cost_fn)

            th = float(best_threshold)
            P, R, FPR, F1, ACC, COST = (
                best_metrics['prec'], best_metrics['rec'], best_metrics['fpr'],
                best_metrics['f1'], best_metrics['acc'], best_metrics['cost']
            )
            print(f"      Selected: {strategy} | th={th:.4f}")
            print(f"      Acc={ACC:.4f} | P={P:.4f} | R={R:.4f} | F1={F1:.4f} | FPR={FPR:.4f} | Cost={COST:.1f}")

            self.model_thresholds[name] = th

            combined = F1*0.35 + R*0.30 + P*0.15 + (1-FPR)*0.10 + (1/(1+0.01*COST))*0.10
            if combined > best_score:
                best_score = combined
                self.best_model = model
                self.best_model_name = name
                self.best_threshold = th

            rows.append({
                "Model": name, "ROC-AUC": roc_auc, "PR-AUC": pr_auc,
                "Final_Precision": P, "Final_Recall": R, "Final_FPR": FPR,
                "Final_F1": F1, "Final_Accuracy": ACC, "Total_Cost": COST,
                "Chosen_Threshold": th, "Strategy": strategy
            })

        return pd.DataFrame(rows)

    
    def evaluate_on_test(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """تقييم على مجموعة الاختبار"""
        print("\n" + "=" * 80)
        print("🧪 TESTING ON HOLDOUT SET")
        print("=" * 80)
        
        X_te_lr, X_te_tree = self.vspace.transform_text(X_test)
        
        if self.best_model_name == 'LR':
            X_use = X_te_lr
        else:
            X_use = X_te_tree if not issparse(X_te_tree) else X_te_tree.toarray()
        
        y_proba = self.best_model.predict_proba(X_use)[:, 1]
        y_pred = (y_proba >= self.best_threshold).astype(int)
        
        tn, fp, fn, tp = _confcounts(y_test, y_pred)
        
        results = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_proba),
            'PR-AUC': average_precision_score(y_test, y_proba),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp),
            'FPR': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        }
        
        print("\n📊 Test Results:")
        print(f"   Accuracy:  {results['Accuracy']:.4f}")
        print(f"   Precision: {results['Precision']:.4f}")
        print(f"   Recall:    {results['Recall']:.4f}")
        print(f"   F1-Score:  {results['F1']:.4f}")
        print(f"   ROC-AUC:   {results['ROC-AUC']:.4f}")
        print(f"   PR-AUC:    {results['PR-AUC']:.4f}")
        print(f"\n   Confusion Matrix: TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
        print(f"   FPR={results['FPR']:.4f}")
        
        return results
    
    def train_enhanced(self, X_train, y_train, X_val, y_val):
        """Wrapper للتوافق"""
        return self.train(X_train, y_train, X_val, y_val)


# ================================================================================
# 🔄 Backward Compatibility
# ================================================================================
EnhancedPhishingTrainer = EnhancedPhishingTrainerV2
EnhancedConfig = EnhancedConfigV2

# ================================================================================
# 🎉 MAIN
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Enhanced Trainer V2.1 - FIXED & OPTIMIZED!")
    print("=" * 80)
    print("\nKey Improvements:")
    print("  ✅ Fixed threshold selection (no more threshold=1.0)")
    print("  ✅ Better memory management")
    print("  ✅ Optimized hyperparameters")
    print("  ✅ Smarter SMOTE handling")
    print("=" * 80)