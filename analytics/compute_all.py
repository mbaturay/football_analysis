"""
Runner that loads run artifacts and writes all analytics.

Usage::

    from analytics.compute_all import compute_all_stats
    compute_all_stats("runs/20260301_124219_abc123")

Individual runners can still be called separately.
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
    sanitize_speeds,
)
from analytics.shape import (
    team_centroid,
    team_dimensions,
    team_surface_area,
    defensive_line_height,
    line_distances,
)
from analytics.ball_movement import (
    ball_speed_and_distance,
    progression,
    territory,
    switches_of_play,
    directness_index,
)
from analytics.quality import (
    compute_transform_coverage,
    compute_speed_sanity,
    compute_transform_confidence,
    save_transform_debug_images,
    CONFIDENCE_THRESHOLD,
)


# ── quality / confidence ─────────────────────────────────────────────────────

def compute_quality(run_dir: str) -> tuple[dict, float]:
    """Compute quality.json and return (quality_dict, confidence_score).

    Always runs first so other modules can check confidence.
    """
    meta = load_run_meta(run_dir)
    df = load_frames(run_dir)

    coverage = compute_transform_coverage(df, meta)

    # Speed sanity needs expanded players
    df_players = expand_players(df)
    speed_sanity = compute_speed_sanity(df_players)

    # Merge into one quality dict
    quality = {**coverage, **speed_sanity}

    confidence, reasons = compute_transform_confidence(quality)
    quality["transform_confidence"] = confidence
    quality["confidence_reasons"] = reasons

    # Write quality.json
    stats_dir = os.path.join(run_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    with open(os.path.join(stats_dir, "quality.json"), "w") as f:
        json.dump(_make_json_safe(quality), f, indent=2)

    # Debug images (guarded)
    try:
        save_transform_debug_images(run_dir, df, meta)
    except Exception:
        pass

    return quality, confidence


# ── possession ───────────────────────────────────────────────────────────────

def compute_possession(run_dir: str, confidence: float = 1.0) -> dict:
    """Load artifacts, run all possession analytics, write outputs.

    Zone-based metrics (zone possession, field tilt) are skipped if confidence
    is below threshold.  Non-zone metrics always run.
    """
    meta = load_run_meta(run_dir)
    df = load_frames(run_dir)
    fps = meta.get("fps", 24)

    # 1. Summary (overall, by-half, rolling) — always computed
    summary = compute_possession_summary(df, fps)
    rolling_df = summary.pop("rolling_5min")

    # 2. Chains — always computed
    chains_df, chains_summary = compute_possession_chains(df, fps)

    # 3. Time to regain — always computed
    regain = compute_time_to_regain(df, fps)

    # 4. Zones — gated
    zone_possession = compute_possession_in_zones(df, meta, confidence=confidence)

    # 5. Field tilt — gated
    field_tilt = compute_field_tilt(df, meta, confidence=confidence)

    # ── assemble JSON-safe output ────────────────────────────────────────
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

    with open(os.path.join(stats_dir, "possession.json"), "w") as f:
        json.dump(_make_json_safe(output), f, indent=2)

    rolling_df.to_parquet(
        os.path.join(stats_dir, "possession_rolling_5min.parquet"), index=False
    )
    chains_df.to_parquet(
        os.path.join(stats_dir, "possession_chains.parquet"), index=False
    )

    output["rolling_5min_df"] = rolling_df
    output["chains_df"] = chains_df
    output["time_to_regain_full"] = regain
    return output


# ── physical ─────────────────────────────────────────────────────────────────

def compute_physical(run_dir: str) -> dict:
    """Load artifacts, run all player physical analytics, write outputs.

    Speed values are sanitized (impossible speeds clamped to None).
    Writes physical_quality.json with clipping stats.
    """
    meta = load_run_meta(run_dir)
    df = load_frames(run_dir)
    fps = meta.get("fps", 24)

    df_players = expand_players(df)

    # Sanitize speeds before computing any speed-based metrics
    df_players, speed_quality = sanitize_speeds(df_players)

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

    # Write physical quality
    with open(os.path.join(stats_dir, "physical_quality.json"), "w") as f:
        json.dump(_make_json_safe(speed_quality), f, indent=2)

    return parquets


# ── shape ────────────────────────────────────────────────────────────────────

def compute_shape(run_dir: str, confidence: float = 1.0) -> dict:
    """Load artifacts, run all team shape analytics, write outputs.

    Skipped entirely if confidence is below threshold.
    """
    stats_dir = os.path.join(run_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    if confidence < CONFIDENCE_THRESHOLD:
        skipped = {
            "skipped": True,
            "reason": f"Transform confidence {confidence:.0%} below threshold",
            "confidence": confidence,
        }
        with open(os.path.join(stats_dir, "shape_skipped.json"), "w") as f:
            json.dump(skipped, f, indent=2)
        return {}

    meta = load_run_meta(run_dir)
    df = load_frames(run_dir)
    fps = int(meta.get("fps", 24))

    df_players = expand_players(df)

    df_centroid = team_centroid(df_players)
    df_dims = team_dimensions(df_players)
    df_area = team_surface_area(df_players, sample_every=fps)
    df_def_line = defensive_line_height(df_players, meta=meta, sample_every=fps)
    df_line_dist = line_distances(df_players, sample_every=fps)

    parquets = {
        "shape_centroid": df_centroid,
        "shape_dims": df_dims,
        "shape_area": df_area,
        "shape_def_line": df_def_line,
        "shape_line_dist": df_line_dist,
    }
    for name, frame in parquets.items():
        frame.to_parquet(os.path.join(stats_dir, f"{name}.parquet"), index=False)

    # Remove skipped marker if it existed from a previous run
    skipped_path = os.path.join(stats_dir, "shape_skipped.json")
    if os.path.isfile(skipped_path):
        os.remove(skipped_path)

    return parquets


# ── ball movement ────────────────────────────────────────────────────────────

def compute_ball_movement(run_dir: str, confidence: float = 1.0) -> dict:
    """Load artifacts, run all ball-movement analytics, write outputs.

    Ball speed per frame always runs.  Territory is gated by confidence.
    """
    meta = load_run_meta(run_dir)
    df = load_frames(run_dir)
    fps = meta.get("fps", 24)

    # Ball speed per frame — always runs
    df_ball = ball_speed_and_distance(df, fps)

    # Possession chains needed for progression & directness
    from analytics.possession import compute_possession_chains
    chains_df, _ = compute_possession_chains(df, fps)

    # Progression & directness per chain
    df_prog = progression(df, chains_df, meta)
    df_direct = directness_index(df_ball, chains_df, meta)

    if not df_prog.empty and not df_direct.empty:
        df_metrics = df_prog.merge(
            df_direct[["chain_id", "total_dist_m", "directness"]],
            on="chain_id", how="left",
        )
    else:
        df_metrics = df_prog

    # Territory — gated
    territory_dict = territory(df, meta, confidence=confidence)

    # Switches of play
    df_switches = switches_of_play(df, meta)

    # ── write to disk ────────────────────────────────────────────────────
    stats_dir = os.path.join(run_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    df_ball.to_parquet(os.path.join(stats_dir, "ball_frame.parquet"), index=False)
    df_metrics.to_parquet(
        os.path.join(stats_dir, "ball_possession_metrics.parquet"), index=False
    )
    df_switches.to_parquet(
        os.path.join(stats_dir, "ball_switches.parquet"), index=False
    )
    with open(os.path.join(stats_dir, "ball_territory.json"), "w") as f:
        json.dump(_make_json_safe(territory_dict), f, indent=2)

    return {
        "ball_frame": df_ball,
        "ball_possession_metrics": df_metrics,
        "ball_territory": territory_dict,
        "ball_switches": df_switches,
    }


# ── combined runner ──────────────────────────────────────────────────────────

def compute_all_stats(run_dir: str) -> dict:
    """Run quality check first, then all analytics with confidence gating."""
    quality, confidence = compute_quality(run_dir)

    results = {"quality": quality, "confidence": confidence}
    results["possession"] = compute_possession(run_dir, confidence=confidence)
    results["physical"] = compute_physical(run_dir)
    results["shape"] = compute_shape(run_dir, confidence=confidence)
    results["ball_movement"] = compute_ball_movement(run_dir, confidence=confidence)
    return results


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


# ── smoke test steps ─────────────────────────────────────────────────────────
# 1. Run pipeline on a video
# 2. Confirm runs/<id>/stats/quality.json exists with transform_confidence
# 3. Open Stats page and confirm confidence displayed
# 4. Confirm runs/<id>/debug/ contains PNG images
# 5. If confidence < 0.35, confirm shape/zone sections show "skipped" card
