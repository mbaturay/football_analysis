"""
Runner that loads run artifacts and writes all possession analytics.

Usage::

    from analytics.compute_all import compute_possession
    compute_possession("runs/20260301_124219_abc123")
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
