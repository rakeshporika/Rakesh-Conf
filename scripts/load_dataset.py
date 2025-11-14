import pandas as pd
from pathlib import Path

def load_dataset(repo_slug: str = "fastapi", curated_dir: str | None = None):
    base = Path(curated_dir) if curated_dir else Path(f"data/{repo_slug}/curated")
    versions = pd.read_parquet(base / "versions.parquet") if (base / "versions.parquet").exists() else pd.read_csv(base / "versions.csv", parse_dates=["date"])
    commits  = pd.read_parquet(base / "commits.parquet") if (base / "commits.parquet").exists() else pd.read_csv(base / "commits.csv", parse_dates=["authored_date","committed_date"])
    files    = pd.read_parquet(base / "files_changed.parquet") if (base / "files_changed.parquet").exists() else pd.read_csv(base / "files_changed.csv")
    return versions, commits, files
