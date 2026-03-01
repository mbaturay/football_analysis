"""
Team shape analytics derived from per-player-per-frame pitch coordinates.

All functions are pure (no Streamlit, no file I/O).
Coordinates: x = court_length axis, y = court_width axis (meters).
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

try:
    from scipy.spatial import ConvexHull, QhullError
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from sklearn.cluster import KMeans as _KMeans
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ── 1. team centroid ────────────────────────────────────────────────────────

def team_centroid(df_players: pd.DataFrame) -> pd.DataFrame:
    """Mean (x, y) per team per frame.

    Returns columns: frame, t, team, cx, cy
    """
    valid = df_players.dropna(subset=["x", "y", "team"])
    if valid.empty:
        return _empty(["frame", "t", "team", "cx", "cy"])

    grp = valid.groupby(["frame", "t", "team"], sort=False)
    out = grp.agg(cx=("x", "mean"), cy=("y", "mean")).reset_index()
    out["cx"] = out["cx"].round(3)
    out["cy"] = out["cy"].round(3)
    return out


# ── 2. team dimensions (width / length) ────────────────────────────────────

def team_dimensions(df_players: pd.DataFrame) -> pd.DataFrame:
    """Spread of each team per frame: width (y-range) and length (x-range).

    Returns columns: frame, t, team, width_m, length_m
    """
    valid = df_players.dropna(subset=["x", "y", "team"])
    if valid.empty:
        return _empty(["frame", "t", "team", "width_m", "length_m"])

    grp = valid.groupby(["frame", "t", "team"], sort=False)
    out = grp.agg(
        width_m=("y", lambda s: s.max() - s.min()),
        length_m=("x", lambda s: s.max() - s.min()),
    ).reset_index()
    out["width_m"] = out["width_m"].round(3)
    out["length_m"] = out["length_m"].round(3)
    return out


# ── 3. team surface area (convex hull) ─────────────────────────────────────

def team_surface_area(
    df_players: pd.DataFrame, sample_every: int = 1
) -> pd.DataFrame:
    """Convex-hull area for each team per frame.

    Heavy computation — use *sample_every* to downsample (e.g. fps for 1 Hz).
    Requires scipy; returns empty DataFrame with a warning if unavailable.

    Returns columns: frame, t, team, area_m2
    """
    if not _HAS_SCIPY:
        warnings.warn("scipy not installed — team_surface_area skipped")
        return _empty(["frame", "t", "team", "area_m2"])

    valid = df_players.dropna(subset=["x", "y", "team"])
    if valid.empty:
        return _empty(["frame", "t", "team", "area_m2"])

    frames = sorted(valid["frame"].unique())
    if sample_every > 1:
        frames = frames[::sample_every]

    sampled = valid[valid["frame"].isin(frames)]
    rows: list[dict] = []

    for (frame, t, team), grp in sampled.groupby(["frame", "t", "team"], sort=False):
        pts = grp[["x", "y"]].values
        area = _hull_area(pts)
        rows.append({"frame": int(frame), "t": float(t), "team": team, "area_m2": area})

    return pd.DataFrame(rows)


def _hull_area(pts: np.ndarray) -> float | None:
    """Convex hull area; None if fewer than 3 unique points."""
    if len(pts) < 3:
        return None
    try:
        return round(float(ConvexHull(pts).volume), 3)  # 2D: .volume = area
    except (QhullError, ValueError):
        return None


# ── 4. defensive line height ───────────────────────────────────────────────

def defensive_line_height(
    df_players: pd.DataFrame,
    meta: dict | None = None,
    sample_every: int = 1,
) -> pd.DataFrame:
    """Average x of the 4 deepest outfield players per team per frame.

    Direction assumption (same as zones.py):
        team 1 attacks +x first half → defensive end is low x
        team 2 attacks −x first half → defensive end is high x
    Halves split at time midpoint.

    Returns columns: frame, t, team, def_line_x
    """
    valid = df_players.dropna(subset=["x", "team"])
    if valid.empty:
        return _empty(["frame", "t", "team", "def_line_x"])

    mid_t = valid["t"].max() / 2

    frames = sorted(valid["frame"].unique())
    if sample_every > 1:
        frames = frames[::sample_every]

    sampled = valid[valid["frame"].isin(frames)]
    rows: list[dict] = []

    for (frame, t, team), grp in sampled.groupby(["frame", "t", "team"], sort=False):
        xs = grp["x"].values
        if len(xs) < 4:
            deepest = xs  # use whatever is available
        else:
            # ascending = True means smallest first
            ascending = _defensive_ascending(int(team), float(t), mid_t)
            sorted_xs = np.sort(xs)
            deepest = sorted_xs[:4] if ascending else sorted_xs[-4:]

        rows.append({
            "frame": int(frame),
            "t": float(t),
            "team": team,
            "def_line_x": round(float(deepest.mean()), 3),
        })

    return pd.DataFrame(rows)


def _defensive_ascending(team: int, t: float, mid_t: float) -> bool:
    """True when the team's defensive end is the low-x side."""
    first_half = t <= mid_t
    if team == 1:
        return first_half       # team 1 defends low-x in 1st half
    return not first_half       # team 2 defends low-x in 2nd half


# ── 5. inter-line distances (3-line k-means) ──────────────────────────────

def line_distances(
    df_players: pd.DataFrame,
    sample_every: int = 1,
) -> pd.DataFrame:
    """Cluster each team's players into 3 lines by x, then compute
    back→mid and mid→front gaps.

    Sampled at *sample_every* frames by default.  Requires sklearn.

    Returns columns: frame, t, team, back_mid_m, mid_front_m
    """
    if not _HAS_SKLEARN:
        warnings.warn("sklearn not installed — line_distances skipped")
        return _empty(["frame", "t", "team", "back_mid_m", "mid_front_m"])

    valid = df_players.dropna(subset=["x", "team"])
    if valid.empty:
        return _empty(["frame", "t", "team", "back_mid_m", "mid_front_m"])

    frames = sorted(valid["frame"].unique())
    if sample_every > 1:
        frames = frames[::sample_every]

    sampled = valid[valid["frame"].isin(frames)]
    rows: list[dict] = []

    for (frame, t, team), grp in sampled.groupby(["frame", "t", "team"], sort=False):
        xs = grp["x"].values.reshape(-1, 1)
        back_mid, mid_front = _three_line_gaps(xs)
        rows.append({
            "frame": int(frame),
            "t": float(t),
            "team": team,
            "back_mid_m": back_mid,
            "mid_front_m": mid_front,
        })

    return pd.DataFrame(rows)


def _three_line_gaps(xs: np.ndarray) -> tuple[float | None, float | None]:
    """K-means k=3 on x positions; return (back→mid, mid→front) gaps."""
    n = len(xs)
    if n < 3:
        return None, None

    k = min(3, n)
    km = _KMeans(n_clusters=k, n_init=1, max_iter=10, random_state=0)
    km.fit(xs)
    centres = sorted(km.cluster_centers_.ravel())

    if k < 3:
        return None, None

    back_mid = round(float(centres[1] - centres[0]), 3)
    mid_front = round(float(centres[2] - centres[1]), 3)
    return back_mid, mid_front


# ── helpers ─────────────────────────────────────────────────────────────────

def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)
