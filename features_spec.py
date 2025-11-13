# features_spec.py
from __future__ import annotations
import os, re, json, math
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, parse_qs
from collections import Counter
import os
from pathlib import Path

# دعم مستخرج JSON
try:
    from features_spec import JsonFeatureExtractor
except Exception:
    JsonFeatureExtractor = None

import numpy as np
import pandas as pd

_AR_RE   = re.compile(r'[\u0600-\u06FF]')
_LAT_RE  = re.compile(r'[A-Za-z]')
_DIG_RE  = re.compile(r'\d')
_PUNCT_RE= re.compile(r'[^\w\s]', re.UNICODE)

def _entropy(s: str) -> float:
    if not s: return 0.0
    cnt = Counter(s)
    n = float(len(s))
    return float(-sum((c/n)*math.log((c/n)+1e-12, 2) for c in cnt.values()))

def _char_ratio(s: str, which: str) -> float:
    if not s: return 0.0
    if which == "arabic":
        m = len(_AR_RE.findall(s))
    elif which == "latin":
        m = len(_LAT_RE.findall(s))
    elif which == "digits":
        m = len(_DIG_RE.findall(s))
    elif which == "punct":
        m = len(_PUNCT_RE.findall(s))
    else:
        m = 0
    return m / max(1, len(s))

def _max_run_len(s: str) -> int:
    if not s: return 0
    best, cur, prev = 1, 1, s[0]
    for ch in s[1:]:
        if ch == prev:
            cur += 1
            if cur > best: best = cur
        else:
            cur = 1
            prev = ch
    return best

def _safe_get_host_path_query(u: str):
    try:
        p = urlsplit(u)
        host  = p.hostname or ""
        path  = p.path or ""
        query = p.query or ""
        return host, path, query
    except Exception:
        return "", "", ""

class JsonFeatureExtractor:
    """
    Reads feature specs from JSON files (url_features.json, email_features.json, common_features.json)
    and appends numeric columns accordingly.
    """
    def __init__(self, feature_dir: str, mode: str = "concat", verbose: int = 1):
        """
        mode: "replace" = تُعيد فقط الأعمدة المُعرّفة في JSON
              "concat"  = تُضيف أعمدة JSON فوق أي أعمدة موجودة مسبقًا
        """
        self.feature_dir = feature_dir
        self.mode = mode
        self.verbose = verbose
        self.specs: Dict[str, Any] = {}
        self.compiled: List[Dict[str, Any]] = []
        self.feature_names_: List[str] = []

    def _load_specs(self):
        def _load_one(name):
            p = os.path.join(self.feature_dir, name)
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None

        self.specs["url"]    = _load_one("url_features.json") or {"features":[]}
        self.specs["email"]  = _load_one("email_features.json") or {"features":[]}
        self.specs["common"] = _load_one("common_features.json") or {"features":[]}

        self.compiled = []
        for group in ["url", "email", "common"]:
            for fs in self.specs[group]["features"]:
                fs = fs.copy()
                if fs.get("type","").startswith("regex"):
                    fs["_re"] = re.compile(fs["pattern"])
                self.compiled.append(fs)

        if self.verbose:
            total = sum(len(self.specs[g]["features"]) for g in self.specs)
            print(f"   ✅ Loaded JSON feature specs: {total} features from {self.feature_dir}")

    def _ensure_parsed_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        if all(c in df.columns for c in ["host","path","query"]):
            return df
        df = df.copy()
        urls = df["url"].fillna("").astype(str) if "url" in df.columns else pd.Series([""]*len(df))
        hpq = urls.apply(_safe_get_host_path_query)
        df["host"]  = hpq.apply(lambda t: t[0])
        df["path"]  = hpq.apply(lambda t: t[1])
        df["query"] = hpq.apply(lambda t: t[2])
        if "domain" not in df.columns:
            # آخِر جزئية بعد آخر نقطة كـ TLD تقريبي
            df["domain"] = df["host"].fillna("").astype(str)
        return df

    def _source_text(self, row: pd.Series, source: str) -> str:
        if source == "url":   return str(row.get("url", "") or "")
        if source == "host":  return str(row.get("host","") or "")
        if source == "path":  return str(row.get("path","") or "")
        if source == "query": return str(row.get("query","") or "")
        if source == "domain":return str(row.get("domain","") or "")
        if source == "email_subject": return str(row.get("email_subject","") or row.get("subject","") or "")
        if source == "email_body":    return str(row.get("email_body","") or row.get("body","") or row.get("text","") or "")
        if source == "text_joined":
            subj = self._source_text(row,"email_subject")
            body = self._source_text(row,"email_body")
            url  = self._source_text(row,"url")
            return f"{subj} {body} {url}".strip()
        return ""

    def _apply_spec_one(self, row: pd.Series, fs: Dict[str, Any]) -> float:
        t = fs["type"]
        src = self._source_text(row, fs.get("source","text_joined"))
        if t == "len":
            return float(len(src))
        if t == "word_count":
            return float(len([w for w in re.findall(r"\b\w+\b", src)]))
        if t == "entropy":
            return float(_entropy(src))
        if t == "regex_count":
            return float(len(fs["_re"].findall(src)))
        if t == "regex_bool":
            return float(1 if fs["_re"].search(src) else 0)
        if t == "char_ratio":
            return float(_char_ratio(src, fs.get("charset","latin")))
        if t == "unique_chars":
            return float(len(set(src)))
        if t == "max_run_len":
            return float(_max_run_len(src))
        if t == "subdomain_count":
            host = self._source_text(row,"host")
            return float(max(0, host.count(".")))
        if t == "tld_len":
            host = self._source_text(row,"host")
            last = host.rsplit(".",1)[-1] if "." in host else host
            return float(len(last))
        if t == "domain_in_list":
            host = self._source_text(row,"host").lower()
            return float(1 if any(host.endswith(d.lower()) for d in fs.get("list",[])) else 0)
        if t == "lookup_bool":
            # نتوقع أعمدة Boolean موجودة (is_rare_domain / umbrella_top1m / majestic_top1m)
            col = fs.get("lookup","")
            return float(1 if row.get(col, False) else 0)
        return 0.0

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._load_specs()
        X = self._ensure_parsed_cols(X)
        feats = {}
        for fs in self.compiled:
            name = fs["name"]
            feats[name] = X.apply(lambda r: self._apply_spec_one(r, fs), axis=1).astype(np.float32)
        out = pd.DataFrame(feats, index=X.index)

        self.feature_names_ = list(out.columns)
        if self.mode == "replace":
            return out
        # concat
        # لا نكرر الأعمدة الموجودة بنفس الاسم
        dup = [c for c in out.columns if c in X.columns]
        if dup:
            out = out.drop(columns=dup, errors="ignore")
        return pd.concat([X, out], axis=1)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # نفس ترتيب التدريب
        X = self._ensure_parsed_cols(X)
        cur = self.fit_transform(X)  # re-run using same specs (pure functions, no state)
        # عندما يكون mode == replace نحتاج فقط أعمدة الميّزات
        want = self.feature_names_
        for c in want:
            if c not in cur.columns:
                cur[c] = 0.0
        cur = cur[want] if self.mode == "replace" else cur
        return cur
