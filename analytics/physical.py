"""
Player physical analytics derived from per-frame tracking data.

All functions are pure (no Streamlit, no file I/O).
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd


# ── speed bands (km/h) ──────────────────────────────────────────────────────

SPEED_BANDS = [
    ("walk", 0, 7),
    ("jog", 7, 14),
    ("run", 14, 20),
    ("sprint", 20, np.inf),
]


# ── data extraction ─────────────────────────────────────────────────────────

def expand_players(df_frames: pd.DataFrame) -> pd.DataFrame:
    """Explode the JSON ``players`` column into one row per player per frame.

    Returns columns: frame, t, player_id, team, x, y, speed_kmh, distance_m, has_ball
    """
    rows: list[dict] = []
    for _, frow in df_frames.iterrows():
        frame = frow["frame"]
        t = frow["t"]
        try:
            players = json.loads(frow["players"]) if isinstance(frow["players"], str) else frow["players"]
        except (json.JSONDecodeError, TypeError):
            continue
        for p in players:
            rows.append({
                "frame": int(frame),
                "t": float(t),
                "player_id": p.get("id"),
                "team": p.get("team"),
                "x": p.get("x"),
                "y": p.get("y"),
                "speed_kmh": p.get("speed_kmh"),
                "distance_m": p.get("distance_m"),
                "has_ball": bool(p.get("has_ball", False)),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "frame", "t", "player_id", "team", "x", "y",
            "speed_kmh", "distance_m", "has_ball",
        ])
    return df


# ── 1. distance covered ────────────────────────────────────────────────────

def distance_covered(df_players: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Total distance, distance per minute, and minutes played per player.

    ``distance_m`` in the source data is cumulative, so total = max per player.
    """
    if df_players.empty:
        return _empty_df(["player_id", "team", "total_distance_m",
                          "minutes_played", "distance_per_min_m"])

    grouped = df_players.groupby("player_id")

    records = []
    for pid, grp in grouped:
        team = grp["team"].mode().iloc[0] if not grp["team"].mode().empty else None
        dists = grp["distance_m"].dropna()
        total = float(dists.max()) if not dists.empty else 0.0
        minutes_played = len(grp) / fps / 60
        dpm = total / minutes_played if minutes_played > 0 else 0.0
        records.append({
            "player_id": int(pid),
            "team": team,
            "total_distance_m": round(total, 2),
            "minutes_played": round(minutes_played, 2),
            "distance_per_min_m": round(dpm, 2),
        })

    return pd.DataFrame(records).sort_values("total_distance_m", ascending=False).reset_index(drop=True)


# ── 2. speed profile ───────────────────────────────────────────────────────

def speed_profile(df_players: pd.DataFrame) -> pd.DataFrame:
    """Max speed and average active speed per player.

    Active = frames where speed_kmh is not None and >= 1 km/h.
    """
    if df_players.empty:
        return _empty_df(["player_id", "team", "max_speed_kmh", "avg_active_speed_kmh"])

    records = []
    for pid, grp in df_players.groupby("player_id"):
        team = grp["team"].mode().iloc[0] if not grp["team"].mode().empty else None
        speeds = grp["speed_kmh"].dropna()
        max_spd = float(speeds.max()) if not speeds.empty else None
        active = speeds[speeds >= 1.0]
        avg_spd = float(active.mean()) if not active.empty else None
        records.append({
            "player_id": int(pid),
            "team": team,
            "max_speed_kmh": round(max_spd, 2) if max_spd is not None else None,
            "avg_active_speed_kmh": round(avg_spd, 2) if avg_spd is not None else None,
        })

    return pd.DataFrame(records).sort_values("max_speed_kmh", ascending=False, na_position="last").reset_index(drop=True)


# ── 3. speed bands ─────────────────────────────────────────────────────────

def speed_bands(df_players: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Time spent in each speed band per player.

    Returns columns: player_id, team, band, band_lo_kmh, band_hi_kmh, frames, time_in_band_s
    """
    if df_players.empty:
        return _empty_df(["player_id", "team", "band", "band_lo_kmh",
                          "band_hi_kmh", "frames", "time_in_band_s"])

    records = []
    for pid, grp in df_players.groupby("player_id"):
        team = grp["team"].mode().iloc[0] if not grp["team"].mode().empty else None
        speeds = grp["speed_kmh"].dropna()
        for label, lo, hi in SPEED_BANDS:
            count = int(((speeds >= lo) & (speeds < hi)).sum())
            records.append({
                "player_id": int(pid),
                "team": team,
                "band": label,
                "band_lo_kmh": lo,
                "band_hi_kmh": hi if hi != np.inf else None,
                "frames": count,
                "time_in_band_s": round(count / fps, 2),
            })

    return pd.DataFrame(records)


# ── 4. accelerations ───────────────────────────────────────────────────────

_ACCEL_THRESHOLD = 2.5   # m/s^2
_DECEL_THRESHOLD = -2.5  # m/s^2


def accelerations(df_players: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Count high-acceleration and high-deceleration events per player.

    Speed is converted to m/s, then accel = delta_v / delta_t between
    consecutive frames where speed exists.
    """
    if df_players.empty:
        return _empty_df(["player_id", "team", "accel_events", "decel_events",
                          "max_accel_ms2", "max_decel_ms2"])

    dt = 1.0 / fps
    records = []

    for pid, grp in df_players.groupby("player_id"):
        team = grp["team"].mode().iloc[0] if not grp["team"].mode().empty else None
        sub = grp.dropna(subset=["speed_kmh"]).sort_values("frame")
        if len(sub) < 2:
            records.append({
                "player_id": int(pid), "team": team,
                "accel_events": 0, "decel_events": 0,
                "max_accel_ms2": None, "max_decel_ms2": None,
            })
            continue

        v_ms = sub["speed_kmh"].values / 3.6
        frames = sub["frame"].values
        # delta_t accounts for frame gaps
        delta_t = np.diff(frames) / fps
        delta_t[delta_t == 0] = dt  # safety
        accel_arr = np.diff(v_ms) / delta_t

        accel_events = int((accel_arr > _ACCEL_THRESHOLD).sum())
        decel_events = int((accel_arr < _DECEL_THRESHOLD).sum())
        max_acc = float(accel_arr.max()) if len(accel_arr) else None
        max_dec = float(accel_arr.min()) if len(accel_arr) else None

        records.append({
            "player_id": int(pid),
            "team": team,
            "accel_events": accel_events,
            "decel_events": decel_events,
            "max_accel_ms2": round(max_acc, 3) if max_acc is not None else None,
            "max_decel_ms2": round(max_dec, 3) if max_dec is not None else None,
        })

    return pd.DataFrame(records).sort_values("accel_events", ascending=False).reset_index(drop=True)


# ── 5. average position & heatmap ──────────────────────────────────────────

def avg_position(df_players: pd.DataFrame) -> pd.DataFrame:
    """Mean x, y position per player (frames with valid coordinates only)."""
    if df_players.empty:
        return _empty_df(["player_id", "team", "avg_x", "avg_y", "valid_frames"])

    records = []
    for pid, grp in df_players.groupby("player_id"):
        team = grp["team"].mode().iloc[0] if not grp["team"].mode().empty else None
        valid = grp.dropna(subset=["x", "y"])
        if valid.empty:
            records.append({"player_id": int(pid), "team": team,
                            "avg_x": None, "avg_y": None, "valid_frames": 0})
        else:
            records.append({
                "player_id": int(pid),
                "team": team,
                "avg_x": round(float(valid["x"].mean()), 3),
                "avg_y": round(float(valid["y"].mean()), 3),
                "valid_frames": int(len(valid)),
            })

    return pd.DataFrame(records)


def heatmap_grid(
    df_players: pd.DataFrame, meta: dict, bins_x: int = 24, bins_y: int = 16
) -> pd.DataFrame:
    """Discretise player positions into a grid and count frames per bin.

    Returns columns: player_id, team, bin_x, bin_y, count_frames
    """
    court_length = meta.get("court_length", 23.32)
    court_width = meta.get("court_width", 68)

    valid = df_players.dropna(subset=["x", "y"]).copy()
    if valid.empty:
        return _empty_df(["player_id", "team", "bin_x", "bin_y", "count_frames"])

    # Clip to pitch bounds then digitize
    valid["bx"] = np.clip(valid["x"], 0, court_length)
    valid["by"] = np.clip(valid["y"], 0, court_width)
    valid["bin_x"] = np.minimum(
        (valid["bx"] / court_length * bins_x).astype(int), bins_x - 1
    )
    valid["bin_y"] = np.minimum(
        (valid["by"] / court_width * bins_y).astype(int), bins_y - 1
    )

    team_mode = valid.groupby("player_id")["team"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)

    grouped = (
        valid.groupby(["player_id", "bin_x", "bin_y"])
        .size()
        .reset_index(name="count_frames")
    )
    grouped["team"] = grouped["player_id"].map(team_mode)

    return grouped[["player_id", "team", "bin_x", "bin_y", "count_frames"]]


# ── helpers ─────────────────────────────────────────────────────────────────

def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)
