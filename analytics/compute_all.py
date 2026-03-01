"""
Runner that loads run artifacts and writes all analytics.

Usage::

    from analytics.compute_all import compute_possession, compute_physical
    compute_possession("runs/20260301_124219_abc123")
    compute_physical("runs/20260301_124219_abc123")
"""

from __future__ import annotations

import json
import os

from analytics_io import load_run_meta, load_frames
from analytics.possession import (
    compute_possession_summary,
    compute_possession_chains,
    compute_time_to_regain,
)
from analytics.zones import compute_possession_in_zones, compute_field_tilt
from analytics.physical import (
    expand_players,
    distance_covered,
    speed_profile,
    speed_bands,
    accelerations,
    avg_position,
    heatmap_grid,
)
from analytics.shape import (
    team_centroid,
    team_dimensions,
    team_surface_area,
    defensive_line_height,
    line_distances,
)


def compute_possession(run_dir: str) -> dict:
    """Load artifacts, run all possession analytics, write outputs.

    Writes into ``<run_dir>/stats/``:
        - possession.json
        - possession_chains.parquet
        - possession_rolling_5min.parquet

    Returns the full results dict (same content as possession.json plus
    the DataFrames).
    """
    meta = load_run_meta(run_dir)
    df = load_frames(run_dir)
    fps = meta.get("fps", 24)

    # 1. Summary (overall, by-half, rolling)
    summary = compute_possession_summary(df, fps)
    rolling_df = summary.pop("rolling_5min")

    # 2. Chains
    chains_df, chains_summary = compute_possession_chains(df, fps)

    # 3. Time to regain
    regain = compute_time_to_regain(df, fps)

    # 4. Zones
    zone_possession = compute_possession_in_zones(df, meta)

    # 5. Field tilt
    field_tilt = compute_field_tilt(df, meta)

    # ── assemble JSON-safe output ────────────────────────────────────────
    # Strip regain_times lists from JSON (can be large); keep summary stats
    regain_json = {}
    for team, rdata in regain.items():
        regain_json[team] = {
            k: v for k, v in rdata.items() if k != "regain_times"
        }

    output = {
        "possession_overall": summary["overall"],
        "possession_by_half": summary["by_half"],
        "chains_summary": chains_summary,
        "time_to_regain": regain_json,
        "zone_possession": zone_possession,
        "field_tilt": field_tilt,
    }

    # ── write to disk ────────────────────────────────────────────────────
    stats_dir = os.path.join(run_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    json_path = os.path.join(stats_dir, "possession.json")
    with open(json_path, "w") as f:
        json.dump(_make_json_safe(output), f, indent=2)

    rolling_df.to_parquet(
        os.path.join(stats_dir, "possession_rolling_5min.parquet"), index=False
    )
    chains_df.to_parquet(
        os.path.join(stats_dir, "possession_chains.parquet"), index=False
    )

    # Return everything (including DataFrames) for programmatic use
    output["rolling_5min_df"] = rolling_df
    output["chains_df"] = chains_df
    output["time_to_regain_full"] = regain
    return output


def compute_physical(run_dir: str) -> dict:
    """Load artifacts, run all player physical analytics, write outputs.

    Writes into ``<run_dir>/stats/``:
        - physical_distance.parquet
        - physical_speed.parquet
        - physical_bands.parquet
        - physical_accel.parquet
        - physical_avgpos.parquet
        - physical_heatmap.parquet

    Returns a dict with all DataFrames keyed by name.
    """
    meta = load_run_meta(run_dir)
    df = load_frames(run_dir)
    fps = meta.get("fps", 24)

    df_players = expand_players(df)

    df_dist = distance_covered(df_players, fps)
    df_speed = speed_profile(df_players)
    df_bands = speed_bands(df_players, fps)
    df_accel = accelerations(df_players, fps)
    df_avgpos = avg_position(df_players)
    df_heat = heatmap_grid(df_players, meta)

    stats_dir = os.path.join(run_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    parquets = {
        "physical_distance": df_dist,
        "physical_speed": df_speed,
        "physical_bands": df_bands,
        "physical_accel": df_accel,
        "physical_avgpos": df_avgpos,
        "physical_heatmap": df_heat,
    }
    for name, frame in parquets.items():
        frame.to_parquet(os.path.join(stats_dir, f"{name}.parquet"), index=False)

    return parquets


def compute_shape(run_dir: str) -> dict:
    """Load artifacts, run all team shape analytics, write outputs.

    Writes into ``<run_dir>/stats/``:
        - shape_centroid.parquet
        - shape_dims.parquet
        - shape_area.parquet
        - shape_def_line.parquet
        - shape_line_dist.parquet

    Heavy per-frame metrics (area, def line, line distances) are sampled
    at 1 Hz (every *fps* frames) by default.

    Returns a dict with all DataFrames keyed by name.
    """
    meta = load_run_meta(run_dir)
    df = load_frames(run_dir)
    fps = int(meta.get("fps", 24))

    df_players = expand_players(df)

    df_centroid = team_centroid(df_players)
    df_dims = team_dimensions(df_players)
    df_area = team_surface_area(df_players, sample_every=fps)
    df_def_line = defensive_line_height(df_players, meta=meta, sample_every=fps)
    df_line_dist = line_distances(df_players, sample_every=fps)

    stats_dir = os.path.join(run_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    parquets = {
        "shape_centroid": df_centroid,
        "shape_dims": df_dims,
        "shape_area": df_area,
        "shape_def_line": df_def_line,
        "shape_line_dist": df_line_dist,
    }
    for name, frame in parquets.items():
        frame.to_parquet(os.path.join(stats_dir, f"{name}.parquet"), index=False)

    return parquets


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
