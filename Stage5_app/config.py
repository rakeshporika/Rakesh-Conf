# # Rakesh-Conf/streamlit/config.py
# import os
# from pathlib import Path

# # Prefer Streamlit Secrets (on Streamlit Community Cloud)
# try:
#     import streamlit as st
#     ST_SECRETS = st.secrets
# except Exception:
#     ST_SECRETS = {}

# def _load_local_env_from_repo_root():
#     # config.py is in .../streamlit/config.py → repo root is one level up
#     repo_root = Path(__file__).resolve().parents[1]
#     env_path = repo_root / ".env"
#     if env_path.exists():
#         try:
#             from dotenv import load_dotenv  # pip install python-dotenv
#             # Do not override OS env if already set
#             load_dotenv(dotenv_path=env_path, override=False)
#         except Exception:
#             # dotenv not installed; os.getenv will still read OS env if set
#             pass

# # Load .env when running locally (has no effect on Streamlit Cloud)
# _load_local_env_from_repo_root()

# def get_secret(name: str, default=None):
#     # 1) Streamlit Secrets
#     if name in ST_SECRETS:
#         return ST_SECRETS[name]
#     # 2) Environment variables (.env loaded above or OS env)
#     return os.getenv(name, default)



# config.py
import os
from pathlib import Path

# --- Load .env for local runs (no effect on Streamlit Cloud) ---
def _load_local_env():
    # Adjust this if config.py is not inside repo root
    # Here we assume repo root is the first parent that contains a .env, else current working directory
    possible_roots = [Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent, Path.cwd()]
    for root in possible_roots:
        env_path = root / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv  # pip install python-dotenv
                load_dotenv(dotenv_path=env_path, override=False)
            except Exception:
                pass
            break

_load_local_env()

# --- Safe accessor for Streamlit secrets (does not explode when missing) ---
def _get_from_streamlit(name):
    try:
        import streamlit as st
        try:
            return st.secrets[name]   # works on Streamlit Cloud, or if you set local secrets.toml
        except Exception:
            return None               # no secrets defined -> fall back to env
    except Exception:
        return None                    # streamlit not installed or not running

def get_secret(name: str, default=None):
    val = _get_from_streamlit(name)
    if val is not None:
        return val
    return os.getenv(name, default)
