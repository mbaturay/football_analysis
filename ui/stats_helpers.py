"""
Caching helpers and file-resolution utilities for the Stats dashboard.

All loaders use @st.cache_data keyed on (path, mtime) so stale caches
are automatically invalidated when files change on disk.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st


# ── path resolution ──────────────────────────────────────────────────────────

def resolve_run_path(run_dir: str) -> Path:
    """Accept an absolute path *or* a bare run-id and return a Path."""
    p = Path(run_dir)
    if p.is_absolute() and p.is_dir():
        return p
    # Treat as relative to runs/
    candidate = Path("runs") / run_dir
    if candidate.is_dir():
        return candidate
    return p  # caller will handle missing dir


def stats_path(run_path: Path) -> Path:
    return run_path / "stats"


# ── mtime helper (for cache busting) ────────────────────────────────────────

def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


# ── cached loaders ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_json(path: str, _mtime: float = 0.0) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_parquet(path: str, _mtime: float = 0.0) -> pd.DataFrame:
    return pd.read_parquet(path)


def try_json(sd: Path, name: str) -> dict | None:
    p = sd / name
    if p.is_file():
        return load_json(str(p), _mtime=_mtime(str(p)))
    return None


def try_parquet(sd: Path, name: str) -> pd.DataFrame | None:
    p = sd / name
    if p.is_file():
        return load_parquet(str(p), _mtime=_mtime(str(p)))
    return None


# ── download converters ──────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def json_to_bytes(d: dict) -> bytes:
    return json.dumps(d, indent=2, default=str).encode("utf-8")


# ── expected stats files ─────────────────────────────────────────────────────

EXPECTED_FILES = [
    "quality.json",
    "possession.json",
    "possession_rolling_5min.parquet",
    "possession_chains.parquet",
    "physical_distance.parquet",
    "physical_speed.parquet",
    "physical_bands.parquet",
    "physical_accel.parquet",
    "physical_avgpos.parquet",
    "physical_heatmap.parquet",
    "physical_quality.json",
    "shape_centroid.parquet",
    "shape_dims.parquet",
    "shape_area.parquet",
    "shape_def_line.parquet",
    "shape_line_dist.parquet",
    "ball_frame.parquet",
    "ball_possession_metrics.parquet",
    "ball_territory.json",
    "ball_switches.parquet",
]


def check_files(sd: Path) -> dict[str, bool]:
    """Return {filename: exists} for every expected stats file."""
    return {name: (sd / name).is_file() for name in EXPECTED_FILES}


# ── team naming & colours ───────────────────────────────────────────────────

DEFAULT_TEAM_COLORS = {1: "#4c9aff", 2: "#ff6b6b"}
DEFAULT_TEAM_NAMES = {1: "Team 1", 2: "Team 2"}


def team_names(meta: dict) -> dict[int, str]:
    """Return {team_id: display_name} from meta or defaults."""
    mapping = meta.get("team_id_map") or meta.get("teams")
    if mapping and isinstance(mapping, dict):
        return {int(k): str(v) for k, v in mapping.items()}
    return dict(DEFAULT_TEAM_NAMES)


def team_colors(meta: dict) -> dict[int, str]:
    return dict(DEFAULT_TEAM_COLORS)
