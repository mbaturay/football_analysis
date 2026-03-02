"""
Ball movement analytics derived from per-frame ball position data.

All functions are pure (no Streamlit, no file I/O).
Coordinates: x = court_length axis, y = court_width axis (meters).
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from analytics.zones import classify_third, classify_lane, THIRD_LABELS, LANE_LABELS
from analytics.quality import CONFIDENCE_THRESHOLD


# ── 1. ball speed & distance per frame ──────────────────────────────────────

def ball_speed_and_distance(df_frames: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Per-frame ball speed (m/s) computed from consecutive positions.

    Returns columns: frame, t, ball_x, ball_y, ball_speed_mps
    """
    df = df_frames[["frame", "t", "ball_x", "ball_y"]].copy()
    dt = 1.0 / fps

    dx = df["ball_x"].diff()
    dy = df["ball_y"].diff()

    # NaN where either position is missing
    dist = np.sqrt(dx ** 2 + dy ** 2)
    frame_gap = df["frame"].diff()
    elapsed = frame_gap * dt
    elapsed = elapsed.replace(0, np.nan)

    df["ball_speed_mps"] = (dist / elapsed).round(3)
    # first frame has no delta
    df.loc[df.index[0], "ball_speed_mps"] = np.nan

    return df


# ── 2. progression per possession chain ────────────────────────────────────

def progression(
    df_frames: pd.DataFrame,
    chains_df: pd.DataFrame,
    meta: dict,
) -> pd.DataFrame:
    """Forward meters gained and progression rate per possession chain.

    Direction convention (same as zones/shape):
        team 1 attacks +x first half, −x second half.
        team 2 is opposite.

    Returns columns: chain_id, team, start_t, end_t, duration_s,
                     start_ball_x, end_ball_x, forward_m, progression_rate_mps
    """
    if chains_df.empty or df_frames.empty:
        return _empty([
            "chain_id", "team", "start_t", "end_t", "duration_s",
            "start_ball_x", "end_ball_x", "forward_m", "progression_rate_mps",
        ])

    mid_t = df_frames["t"].max() / 2
    ball_by_frame = df_frames.set_index("frame")[["ball_x"]].to_dict("index")

    rows: list[dict] = []
    for _, ch in chains_df.iterrows():
        sf = int(ch["start_frame"])
        ef = int(ch["end_frame"])
        team = int(ch["team"])
        duration = float(ch["duration_s"])

        bx_start = ball_by_frame.get(sf, {}).get("ball_x")
        bx_end = ball_by_frame.get(ef, {}).get("ball_x")

        if bx_start is None or bx_end is None or math.isnan(bx_start) or math.isnan(bx_end):
            rows.append(_prog_row(ch, None, None, None, None))
            continue

        raw_delta = bx_end - bx_start
        sign = _attack_sign(team, float(ch["start_t"]), mid_t)
        forward_m = round(raw_delta * sign, 3)
        rate = round(forward_m / duration, 3) if duration > 0 else None

        rows.append(_prog_row(ch, bx_start, bx_end, forward_m, rate))

    return pd.DataFrame(rows)


def _prog_row(ch, bx_s, bx_e, fwd, rate):
    return {
        "chain_id": int(ch["chain_id"]),
        "team": int(ch["team"]),
        "start_t": ch["start_t"],
        "end_t": ch["end_t"],
        "duration_s": ch["duration_s"],
        "start_ball_x": round(bx_s, 3) if bx_s is not None else None,
        "end_ball_x": round(bx_e, 3) if bx_e is not None else None,
        "forward_m": fwd,
        "progression_rate_mps": rate,
    }


def _attack_sign(team: int, t: float, mid_t: float) -> int:
    """Return +1 if the team attacks +x, −1 if attacks −x."""
    first_half = t <= mid_t
    if team == 1:
        return 1 if first_half else -1
    return -1 if first_half else 1


# ── 3. territory ───────────────────────────────────────────────────────────

# Standard penalty box: 16.5 m from goal line, 40.32 m wide (centred).
# Scaled to visible pitch section in meta coords.
_PEN_DEPTH_REAL = 16.5   # meters from goal line
_PEN_WIDTH_REAL = 40.32  # meters wide, centred

def territory(
    df_frames: pd.DataFrame,
    meta: dict,
    confidence: float = 1.0,
) -> dict:
    """Ball time in each third (overall & by possessing team), plus danger-zone time.

    Danger zones are approximate penalty-box rectangles at each end.

    Returns a JSON-serialisable dict.
    """
    if confidence < CONFIDENCE_THRESHOLD:
        return {"thirds_overall": {}, "thirds_by_team": {},
                "danger_zone_frames": {}, "penalty_box_approx": {},
                "skipped_frames": len(df_frames),
                "skipped": True, "reason": f"Transform confidence {confidence:.0%} below threshold"}

    court_length = meta.get("court_length", 23.32)
    court_width = meta.get("court_width", 68)

    valid = df_frames.dropna(subset=["ball_x", "ball_y"])
    skipped = len(df_frames) - len(valid)
    total = len(valid)

    # ── thirds overall ──
    thirds_overall = {l: 0 for l in THIRD_LABELS}
    for bx in valid["ball_x"]:
        t = classify_third(bx, court_length)
        if t:
            thirds_overall[t] += 1
    thirds_overall = _normalise(thirds_overall, total)

    # ── thirds by team ──
    teams = sorted(valid["team_in_possession"].dropna().unique())
    thirds_by_team: dict = {}
    for team in teams:
        sub = valid[valid["team_in_possession"] == team]
        counts = {l: 0 for l in THIRD_LABELS}
        for bx in sub["ball_x"]:
            t = classify_third(bx, court_length)
            if t:
                counts[t] += 1
        thirds_by_team[int(team)] = _normalise(counts, len(sub))

    # ── danger zones ──
    pen_depth = min(_PEN_DEPTH_REAL, court_length / 2)
    pen_half_w = _PEN_WIDTH_REAL / 2
    y_mid = court_width / 2
    pen_y_lo = max(y_mid - pen_half_w, 0)
    pen_y_hi = min(y_mid + pen_half_w, court_width)

    danger_low = 0   # ball near x=0 goal
    danger_high = 0  # ball near x=court_length goal
    for _, row in valid.iterrows():
        bx, by = row["ball_x"], row["ball_y"]
        if pen_y_lo <= by <= pen_y_hi:
            if bx <= pen_depth:
                danger_low += 1
            elif bx >= court_length - pen_depth:
                danger_high += 1

    return {
        "thirds_overall": thirds_overall,
        "thirds_by_team": thirds_by_team,
        "danger_zone_frames": {
            "near_x0_goal": danger_low,
            "near_xmax_goal": danger_high,
            "total_valid_frames": total,
            "near_x0_pct": round(danger_low / total, 4) if total else 0,
            "near_xmax_pct": round(danger_high / total, 4) if total else 0,
        },
        "penalty_box_approx": {
            "depth_m": round(pen_depth, 2),
            "y_range": [round(pen_y_lo, 2), round(pen_y_hi, 2)],
        },
        "skipped_frames": skipped,
    }


def _normalise(counts: dict, total: int) -> dict:
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: round(v / total, 4) for k, v in counts.items()}


# ── 4. switches of play ───────────────────────────────────────────────────

def switches_of_play(
    df_frames: pd.DataFrame,
    meta: dict,
    window_seconds: float = 6.0,
) -> pd.DataFrame:
    """Detect ball lane changes (left↔right) within *window_seconds*.

    A switch = ball is in 'left' lane at frame A and 'right' lane (or vice
    versa) at frame B, where B.t − A.t <= window_seconds, with no requirement
    on intermediate positions.

    Returns columns: frame_from, t_from, lane_from, frame_to, t_to, lane_to,
                     elapsed_s, team_in_possession
    """
    court_width = meta.get("court_width", 68)

    valid = df_frames.dropna(subset=["ball_y"]).copy()
    if valid.empty:
        return _empty([
            "frame_from", "t_from", "lane_from",
            "frame_to", "t_to", "lane_to",
            "elapsed_s", "team_in_possession",
        ])

    valid = valid.reset_index(drop=True)
    valid["lane"] = valid["ball_y"].apply(lambda y: classify_lane(y, court_width))

    # Find transitions between outer lanes
    outer = valid[valid["lane"].isin(["left", "right"])].reset_index(drop=True)
    if outer.empty:
        return _empty([
            "frame_from", "t_from", "lane_from",
            "frame_to", "t_to", "lane_to",
            "elapsed_s", "team_in_possession",
        ])

    rows: list[dict] = []
    prev_lane = None
    prev_frame = prev_t = prev_team = None

    for _, r in outer.iterrows():
        lane = r["lane"]
        if prev_lane is not None and lane != prev_lane:
            elapsed = r["t"] - prev_t
            if elapsed <= window_seconds:
                rows.append({
                    "frame_from": int(prev_frame),
                    "t_from": prev_t,
                    "lane_from": prev_lane,
                    "frame_to": int(r["frame"]),
                    "t_to": r["t"],
                    "lane_to": lane,
                    "elapsed_s": round(elapsed, 4),
                    "team_in_possession": prev_team,
                })
        prev_lane = lane
        prev_frame = r["frame"]
        prev_t = r["t"]
        prev_team = r.get("team_in_possession")

    return pd.DataFrame(rows) if rows else _empty([
        "frame_from", "t_from", "lane_from",
        "frame_to", "t_to", "lane_to",
        "elapsed_s", "team_in_possession",
    ])


# ── 5. directness index ───────────────────────────────────────────────────

def directness_index(
    df_ball: pd.DataFrame,
    chains_df: pd.DataFrame,
    meta: dict,
) -> pd.DataFrame:
    """Forward distance / total distance per possession chain.

    *df_ball* must have columns: frame, ball_x, ball_y (use output of
    ball_speed_and_distance or the raw df_frames).

    Returns columns: chain_id, team, duration_s, forward_m, total_dist_m,
                     directness
    """
    if chains_df.empty or df_ball.empty:
        return _empty([
            "chain_id", "team", "duration_s",
            "forward_m", "total_dist_m", "directness",
        ])

    mid_t = df_ball["t"].max() / 2

    # Pre-index ball positions
    ball_indexed = df_ball.set_index("frame")[["ball_x", "ball_y", "t"]].sort_index()

    rows: list[dict] = []
    for _, ch in chains_df.iterrows():
        sf = int(ch["start_frame"])
        ef = int(ch["end_frame"])
        team = int(ch["team"])

        segment = ball_indexed.loc[
            (ball_indexed.index >= sf) & (ball_indexed.index <= ef)
        ].dropna(subset=["ball_x", "ball_y"])

        if len(segment) < 2:
            rows.append({
                "chain_id": int(ch["chain_id"]), "team": team,
                "duration_s": ch["duration_s"],
                "forward_m": None, "total_dist_m": None, "directness": None,
            })
            continue

        xs = segment["ball_x"].values
        ys = segment["ball_y"].values

        # total distance
        dx = np.diff(xs)
        dy = np.diff(ys)
        total_dist = float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))

        # forward distance (direction-normalised)
        sign = _attack_sign(team, float(ch["start_t"]), mid_t)
        forward_m = (xs[-1] - xs[0]) * sign

        directness = abs(forward_m) / total_dist if total_dist > 0 else None

        rows.append({
            "chain_id": int(ch["chain_id"]),
            "team": team,
            "duration_s": ch["duration_s"],
            "forward_m": round(forward_m, 3),
            "total_dist_m": round(total_dist, 3),
            "directness": round(directness, 4) if directness is not None else None,
        })

    return pd.DataFrame(rows)


# ── helpers ─────────────────────────────────────────────────────────────────

def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)
