import json
import os
import pickle
import random
import string
from datetime import datetime

import pandas as pd


def _rand_suffix(k=6):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=k))


def create_run_dir(base_dir="runs"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{_rand_suffix()}"
    run_dir = os.path.join(base_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run_meta(run_dir, meta: dict):
    path = os.path.join(run_dir, "run_meta.json")
    serialisable = _make_json_safe(meta)
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)


def save_tracks_pickle(run_dir, tracks: dict):
    path = os.path.join(run_dir, "tracks.pkl")
    with open(path, "wb") as f:
        pickle.dump(tracks, f)


def write_frames_parquet(run_dir, df_frames: pd.DataFrame):
    path = os.path.join(run_dir, "frames.parquet")
    df_frames.to_parquet(path, index=False)


# ── loaders ──────────────────────────────────────────────────────────────────

def load_run_meta(run_dir):
    path = os.path.join(run_dir, "run_meta.json")
    with open(path) as f:
        return json.load(f)


def load_tracks(run_dir):
    path = os.path.join(run_dir, "tracks.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_frames(run_dir):
    path = os.path.join(run_dir, "frames.parquet")
    return pd.read_parquet(path)


def load_run(run_dir):
    return load_run_meta(run_dir), load_tracks(run_dir), load_frames(run_dir)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_json_safe(obj):
    """Recursively convert numpy / non-serialisable types for JSON."""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
