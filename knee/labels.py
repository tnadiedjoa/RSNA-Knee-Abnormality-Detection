"""Report grades (scripts/llm_labels.py) -> soft training labels.

The mapping is fixed a priori (monotone in the grade) instead of fitted on the 58 gold studies,
which stay an independent check. "not_mentioned" gets a per-finding prior: silence usually means
normal. Synovitis is the exception: reports rarely discuss it while annotators mark it on about
half the studies, so an unmentioned synovitis inherits part of the effusion score (the two go
together; a public label set reported the same trick).
"""

import re

import numpy as np
import pandas as pd

from knee.config import DATA_DIR, LABELS

GRADE_SCORE = {"absent": 0.02, "equivocal": 0.25, "mild": 0.6, "moderate": 0.85, "severe": 0.97}
SILENCE_PRIOR = {label: 0.10 for label in LABELS}


def grades_to_soft(grades: pd.DataFrame, synovitis_from_effusion: bool = True) -> pd.DataFrame:
    """grades: StudyInstanceUID + one grade column per label -> soft labels in [0, 1]."""
    out = grades[["StudyInstanceUID"]].copy()
    for label in LABELS:
        g = grades[label]
        out[label] = g.map(GRADE_SCORE).fillna(SILENCE_PRIOR[label]).astype(np.float32)
        out[f"{label}__mentioned"] = g.isin(list(GRADE_SCORE)).astype(np.int8)
    if synovitis_from_effusion:
        silent = out["Synovitis__mentioned"] == 0
        out.loc[silent, "Synovitis"] = (0.1 + 0.5 * out.loc[silent, "Effusion"]).astype(np.float32)
    return out


def load_soft(name: str, **kwargs) -> pd.DataFrame:
    return grades_to_soft(pd.read_parquet(DATA_DIR / "labels" / f"{name}.parquet"), **kwargs)


def gold() -> pd.DataFrame:
    train = pd.read_csv(DATA_DIR / "train.csv")
    return train.dropna(subset=LABELS)[["StudyInstanceUID"] + LABELS].reset_index(drop=True)


# report language from stopword counts (Cyrillic -> Bulgarian, Greek letters -> Greek); 99.5 % agreement with the
# 'idioma' column of the public codeandodev weak-label table on the 4,407 training reports
_STOPWORDS = {
    "en": r"\b(the|and|of|is|with|no|there|are|without|seen)\b",
    "es": r"\b(de|la|el|con|sin|los|las|se|del|en|rodilla|menisco)\b",
    "tr": r"\b(ve|ile|bir|izlendi|mevcut|normal|olarak|menisk[uü]s|diz)\b",
    "hr": r"\b(je|i|se|u|bez|na|koljena|meniskus|vidi|uredan|uredne)\b",
    "de": r"\b(der|die|das|und|mit|ohne|des|im|kein|keine|meniskus|kniegelenk)\b",
    "nl": r"\b(de|het|een|van|en|met|geen|niet|zonder|meniscus|knie)\b",
    "fr": r"\b(le|la|les|des|du|et|sans|une|un|ménisque|genou|pas)\b",
}


def report_language(text: str) -> str:
    t = str(text).lower()
    if len(re.findall(r"[а-яё]{3,}", t)) > 5:
        return "bg"
    if len(re.findall(r"[α-ωά-ώ]{3,}", t)) > 5:
        return "el"
    return max(_STOPWORDS, key=lambda k: len(re.findall(_STOPWORDS[k], t)))


def site_proxy(studies: pd.DataFrame, min_studies: int = 30) -> pd.Series:
    """Reporting site of each training study: report language x scanner maker (the competition gives no site id);
    groups under min_studies pooled as "other". Training-only (the test has no report)."""
    maker = (pd.read_parquet(DATA_DIR / "series_meta.parquet").groupby("study").first().Manufacturer
             .fillna("?").str.split().str[0].str.upper())
    site = studies["Report"].map(report_language) + "|" + studies["StudyInstanceUID"].map(maker).fillna("?")
    counts = site.value_counts()
    return site.where(site.map(counts) >= min_studies, "other")
