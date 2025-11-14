from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import pandas as pd

DATA_ROOT = Path("../data")

# ---- Helpers ---------------------------------------------------------------

def _read(name: str, repo_slug: str) -> pd.DataFrame:
    base = DATA_ROOT / repo_slug / "curated"
    pq = base / f"{name}.parquet"
    csv = base / f"{name}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_csv(csv)

@lru_cache(maxsize=16)
def load_versions(repo_slug: str) -> pd.DataFrame:
    df = _read("versions", repo_slug).copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
    return df

@lru_cache(maxsize=16)
def load_modules(repo_slug: str) -> pd.DataFrame:
    return _read("modules", repo_slug).copy()

@lru_cache(maxsize=16)
def load_edges(repo_slug: str) -> pd.DataFrame:
    return _read("edges", repo_slug).copy()

@lru_cache(maxsize=16)
def load_metrics(repo_slug: str) -> pd.DataFrame:
    return _read("metrics", repo_slug).copy()

@lru_cache(maxsize=16)
def load_drift(repo_slug: str) -> pd.DataFrame:
    return _read("drift_summary", repo_slug).copy()

@lru_cache(maxsize=16)
def load_changes_modules(repo_slug: str) -> pd.DataFrame:
    return _read("changes_modules", repo_slug).copy()

@lru_cache(maxsize=16)
def load_changes_edges(repo_slug: str) -> pd.DataFrame:
    return _read("changes_edges", repo_slug).copy()

@lru_cache(maxsize=16)
def load_changes_metrics(repo_slug: str) -> pd.DataFrame:
    return _read("changes_metrics", repo_slug).copy()

# ---- Convenience -----------------------------------------------------------

def list_tags(repo_slug: str) -> list[str]:
    return load_versions(repo_slug)["tag"].tolist()


def slice_by_tag(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    if "tag" in df.columns:
        return df[df["tag"] == tag].copy()
    return df.copy()


def get_repo_path(repo_slug: str) -> Path:
    return DATA_ROOT / repo_slug / "curated"