"""
Transform quality, data coverage, and confidence metrics.

All functions are pure (no Streamlit, no file I/O except debug image saving).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


# ── pitch bounds ─────────────────────────────────────────────────────────────

def compute_pitch_bounds(meta: dict) -> tuple[float, float]:
    """Return (length, width) in meters from meta, with safe defaults."""
    L = meta.get("pitch_length_m") or meta.get("court_length") or 105.0
    W = meta.get("pitch_width_m") or meta.get("court_width") or 68.0
    return float(L), float(W)


# ── transform coverage ───────────────────────────────────────────────────────

def compute_transform_coverage(df_frames: pd.DataFrame, meta: dict) -> dict:
    """Compute how much of the data has valid pitch-coordinate positions.

    Returns a JSON-serialisable dict.
    """
    L, W = compute_pitch_bounds(meta)
    total = len(df_frames)
    if total == 0:
        return _empty_coverage()

    # Ball position
    ball_present = df_frames["ball_x"].notna() & df_frames["ball_y"].notna()
    ball_present_n = int(ball_present.sum())
    ball_present_pct = round(ball_present_n / total, 4)

    # Ball in pitch bounds
    in_bounds = (
        ball_present
        & (df_frames["ball_x"] >= 0) & (df_frames["ball_x"] <= L)
        & (df_frames["ball_y"] >= 0) & (df_frames["ball_y"] <= W)
    )
    in_bounds_n = int(in_bounds.sum())
    in_bounds_pct = round(in_bounds_n / ball_present_n, 4) if ball_present_n > 0 else 0.0

    # Out-of-bounds examples
    oob = df_frames[ball_present & ~in_bounds][["frame", "t", "ball_x", "ball_y"]].head(5)
    oob_examples = oob.to_dict("records") if not oob.empty else []

    # Players position coverage (estimate from players JSON column)
    players_present_pct = _estimate_player_coverage(df_frames)

    return {
        "total_frames": total,
        "ball_pos_present_frames": ball_present_n,
        "ball_pos_present_pct": ball_present_pct,
        "ball_pos_in_pitch_bounds_frames": in_bounds_n,
        "ball_pos_in_pitch_bounds_pct": in_bounds_pct,
        "players_pos_present_pct": players_present_pct,
        "any_out_of_bounds_examples": oob_examples,
        "pitch_length_m": L,
        "pitch_width_m": W,
    }


def _estimate_player_coverage(df_frames: pd.DataFrame) -> float:
    """Estimate fraction of player samples with valid (x,y)."""
    try:
        import json as _json
        total = 0
        present = 0
        # Sample up to 200 frames to keep it fast
        sample = df_frames.head(200) if len(df_frames) > 200 else df_frames
        for players_str in sample["players"]:
            if not players_str or players_str == "[]":
                continue
            try:
                players = _json.loads(players_str) if isinstance(players_str, str) else players_str
            except (ValueError, TypeError):
                continue
            for p in players:
                total += 1
                if p.get("x") is not None and p.get("y") is not None:
                    present += 1
        return round(present / total, 4) if total > 0 else 0.0
    except Exception:
        return 0.0


def _empty_coverage() -> dict:
    return {
        "total_frames": 0,
        "ball_pos_present_frames": 0,
        "ball_pos_present_pct": 0.0,
        "ball_pos_in_pitch_bounds_frames": 0,
        "ball_pos_in_pitch_bounds_pct": 0.0,
        "players_pos_present_pct": 0.0,
        "any_out_of_bounds_examples": [],
        "pitch_length_m": 105.0,
        "pitch_width_m": 68.0,
    }


# ── speed sanity ─────────────────────────────────────────────────────────────

def compute_speed_sanity(df_players: pd.DataFrame) -> dict:
    """Check speed distribution for impossible values."""
    if df_players.empty or "speed_kmh" not in df_players.columns:
        return {
            "max_speed_kmh_observed": None,
            "pct_speed_over_40": 0.0,
            "pct_speed_over_50": 0.0,
        }

    speeds = df_players["speed_kmh"].dropna()
    n = len(speeds)
    if n == 0:
        return {
            "max_speed_kmh_observed": None,
            "pct_speed_over_40": 0.0,
            "pct_speed_over_50": 0.0,
        }

    return {
        "max_speed_kmh_observed": round(float(speeds.max()), 2),
        "pct_speed_over_40": round(float((speeds > 40).sum() / n), 4),
        "pct_speed_over_50": round(float((speeds > 50).sum() / n), 4),
    }


# ── transform confidence ────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.35


def compute_transform_confidence(quality_dict: dict) -> tuple[float, list[str]]:
    """Score in [0, 1] indicating how trustworthy the pitch transform is.

    Returns (score, reasons_list).
    """
    score = 1.0
    reasons: list[str] = []

    ball_pct = quality_dict.get("ball_pos_present_pct", 0.0)
    score *= ball_pct
    if ball_pct < 0.5:
        reasons.append(f"Only {ball_pct:.0%} of frames have ball position")

    bounds_pct = quality_dict.get("ball_pos_in_pitch_bounds_pct", 0.0)
    score *= bounds_pct
    if bounds_pct < 0.7:
        reasons.append(f"Only {bounds_pct:.0%} of ball positions are within pitch bounds")

    speed_over_50 = quality_dict.get("pct_speed_over_50", 0.0)
    score *= max(0.0, 1.0 - speed_over_50)
    if speed_over_50 > 0.05:
        reasons.append(f"{speed_over_50:.1%} of speed samples exceed 50 km/h")

    score = max(0.0, min(1.0, score))
    if not reasons:
        reasons.append("Transform looks reasonable")

    return round(score, 4), reasons


# ── debug images ─────────────────────────────────────────────────────────────

def save_transform_debug_images(run_dir: str, df_frames: pd.DataFrame, meta: dict):
    """Write debug PNG scatter/time-series plots into runs/<run_id>/debug/.

    Uses matplotlib (backend Agg) — never shown in Streamlit, only saved.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return  # matplotlib optional

    L, W = compute_pitch_bounds(meta)
    debug_dir = os.path.join(run_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # 1. Ball scatter
    try:
        valid = df_frames.dropna(subset=["ball_x", "ball_y"])
        if not valid.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(valid["ball_x"], valid["ball_y"], s=1, alpha=0.3, c="orange")
            ax.axhline(0, color="white", lw=0.5, ls="--")
            ax.axhline(W, color="white", lw=0.5, ls="--")
            ax.axvline(0, color="white", lw=0.5, ls="--")
            ax.axvline(L, color="white", lw=0.5, ls="--")
            ax.set_xlabel("ball_x (m)")
            ax.set_ylabel("ball_y (m)")
            ax.set_title(f"Ball positions (pitch bounds: {L}x{W} m)")
            ax.set_facecolor("#1a1a2e")
            fig.patch.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            fig.tight_layout()
            fig.savefig(os.path.join(debug_dir, "pitch_scatter_ball.png"), dpi=120)
            plt.close(fig)
    except Exception:
        pass

    # 2. Player scatter (sample 5 seconds)
    try:
        import json as _json
        fps = meta.get("fps", 24)
        sample_frames = 5 * fps
        sample_df = df_frames.head(sample_frames)
        xs, ys, teams = [], [], []
        for players_str in sample_df["players"]:
            if not players_str:
                continue
            try:
                players = _json.loads(players_str) if isinstance(players_str, str) else players_str
            except (ValueError, TypeError):
                continue
            for p in players:
                if p.get("x") is not None and p.get("y") is not None:
                    xs.append(p["x"])
                    ys.append(p["y"])
                    teams.append(p.get("team", 0))
        if xs:
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#4c9aff" if t == 1 else "#ff6b6b" for t in teams]
            ax.scatter(xs, ys, s=2, alpha=0.4, c=colors)
            ax.axhline(0, color="white", lw=0.5, ls="--")
            ax.axhline(W, color="white", lw=0.5, ls="--")
            ax.axvline(0, color="white", lw=0.5, ls="--")
            ax.axvline(L, color="white", lw=0.5, ls="--")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(f"Player positions (first {5}s sample)")
            ax.set_facecolor("#1a1a2e")
            fig.patch.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            fig.tight_layout()
            fig.savefig(os.path.join(debug_dir, "pitch_scatter_players_sample.png"), dpi=120)
            plt.close(fig)
    except Exception:
        pass

    # 3. Ball x over time
    try:
        valid = df_frames.dropna(subset=["ball_x"])
        if not valid.empty:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(valid["t"], valid["ball_x"], lw=0.5, alpha=0.7, color="#ffa726")
            ax.axhline(0, color="white", lw=0.5, ls="--")
            ax.axhline(L, color="white", lw=0.5, ls="--")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("ball_x (m)")
            ax.set_title("Ball X over time (spikes = transform issues)")
            ax.set_facecolor("#1a1a2e")
            fig.patch.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            fig.tight_layout()
            fig.savefig(os.path.join(debug_dir, "pitch_time_series_ballx.png"), dpi=120)
            plt.close(fig)
    except Exception:
        pass
