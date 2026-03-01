"""
Possession analytics computed from frames.parquet run artifacts.

All functions are pure (no Streamlit, no file I/O).
Teams are labelled 1 and 2 (as assigned by TeamAssigner).
"""

from __future__ import annotations

import pandas as pd


# ── 1. Possession summary ───────────────────────────────────────────────────

def compute_possession_summary(df_frames: pd.DataFrame, fps: float) -> dict:
    """Overall, per-half, and rolling 5-minute possession percentages.

    Returns::

        {
            "overall": {1: 0.xx, 2: 0.xx},
            "by_half":  {1: {1: pct, 2: pct}, 2: {1: pct, 2: pct}},
            "rolling_5min": <DataFrame>
        }
    """
    col = "team_in_possession"
    valid = df_frames.dropna(subset=[col])
    teams = sorted(valid[col].unique())

    # --- overall ---
    total = len(valid)
    overall = {}
    for t in teams:
        overall[int(t)] = round((valid[col] == t).sum() / total, 4) if total else 0.0

    # --- by half (split at time midpoint) ---
    mid_t = valid["t"].max() / 2 if not valid.empty else 0
    h1 = valid[valid["t"] <= mid_t]
    h2 = valid[valid["t"] > mid_t]

    by_half = {}
    for half_idx, half_df in enumerate([h1, h2], start=1):
        n = len(half_df)
        by_half[half_idx] = {}
        for t in teams:
            by_half[half_idx][int(t)] = round((half_df[col] == t).sum() / n, 4) if n else 0.0

    # --- rolling 5-minute windows ---
    rolling_df = _rolling_possession(valid, col, teams, window_seconds=300)

    return {
        "overall": overall,
        "by_half": by_half,
        "rolling_5min": rolling_df,
    }


def _rolling_possession(
    df: pd.DataFrame, col: str, teams, window_seconds: float = 300
) -> pd.DataFrame:
    """Non-overlapping windows of *window_seconds* length."""
    if df.empty:
        return pd.DataFrame(
            columns=["window_start_t", "window_end_t"]
            + [f"team{int(t)}_pct" for t in teams]
        )

    max_t = df["t"].max()
    rows = []
    start = 0.0
    while start < max_t:
        end = start + window_seconds
        window = df[(df["t"] >= start) & (df["t"] < end)]
        n = len(window)
        row = {"window_start_t": round(start, 2), "window_end_t": round(end, 2)}
        for t in teams:
            pct = (window[col] == t).sum() / n if n else 0.0
            row[f"team{int(t)}_pct"] = round(pct, 4)
        rows.append(row)
        start = end

    return pd.DataFrame(rows)


# ── 2. Possession chains ────────────────────────────────────────────────────

def compute_possession_chains(
    df_frames: pd.DataFrame, fps: float
) -> tuple[pd.DataFrame, dict]:
    """Identify consecutive-frame possession chains.

    Returns (chains_df, summary_dict).
    """
    col = "team_in_possession"
    valid = df_frames.dropna(subset=[col]).reset_index(drop=True)

    chains: list[dict] = []
    if valid.empty:
        chains_df = pd.DataFrame(
            columns=[
                "chain_id", "team", "start_frame", "end_frame",
                "start_t", "end_t", "duration_s",
            ]
        )
        return chains_df, _empty_chain_summary()

    chain_id = 0
    prev_team = None
    start_frame = start_t = None

    for _, row in valid.iterrows():
        team = int(row[col])
        if team != prev_team:
            # close previous chain
            if prev_team is not None:
                chains.append(_chain_row(chain_id, prev_team, start_frame, prev_frame, start_t, prev_t))
                chain_id += 1
            prev_team = team
            start_frame = int(row["frame"])
            start_t = row["t"]
        prev_frame = int(row["frame"])
        prev_t = row["t"]

    # close last chain
    if prev_team is not None:
        chains.append(_chain_row(chain_id, prev_team, start_frame, prev_frame, start_t, prev_t))

    chains_df = pd.DataFrame(chains)

    # summary
    summary: dict = {}
    for team, grp in chains_df.groupby("team"):
        summary[int(team)] = {
            "avg_possession_duration_s": round(grp["duration_s"].mean(), 3),
            "longest_possession_duration_s": round(grp["duration_s"].max(), 3),
            "number_of_possessions": int(len(grp)),
        }

    return chains_df, summary


def _chain_row(chain_id, team, sf, ef, st, et):
    return {
        "chain_id": chain_id,
        "team": team,
        "start_frame": sf,
        "end_frame": ef,
        "start_t": round(st, 4),
        "end_t": round(et, 4),
        "duration_s": round(et - st, 4),
    }


def _empty_chain_summary():
    return {}


# ── 3. Time to regain possession ────────────────────────────────────────────

def compute_time_to_regain(df_frames: pd.DataFrame, fps: float) -> dict:
    """For each team: when they lose possession, how long until they get it back?

    Returns::

        {
            1: {"avg_regain_s": x, "median_regain_s": x, "regain_times": [...]},
            2: { ... },
        }
    """
    col = "team_in_possession"
    valid = df_frames.dropna(subset=[col]).reset_index(drop=True)

    if valid.empty:
        return {}

    teams = sorted(valid[col].unique())

    # Detect transitions: (team_that_lost, time_lost, team_that_gained)
    transitions: list[tuple] = []
    prev_team = None
    for _, row in valid.iterrows():
        team = int(row[col])
        if prev_team is not None and team != prev_team:
            transitions.append((prev_team, row["t"]))
        prev_team = team

    # For each team, find pairs: lost_at -> regained_at
    regain_times: dict[int, list[float]] = {int(t): [] for t in teams}

    for i, (lost_team, lost_t) in enumerate(transitions):
        # Scan forward for next transition where lost_team regains
        for j in range(i + 1, len(transitions)):
            future_lost_team, future_t = transitions[j]
            if future_lost_team != lost_team:
                # The team that lost at transitions[j] is NOT lost_team,
                # meaning lost_team just regained possession at future_t.
                regain_times[lost_team].append(round(future_t - lost_t, 4))
                break

    result = {}
    for t in teams:
        t = int(t)
        times = regain_times[t]
        if times:
            result[t] = {
                "avg_regain_s": round(sum(times) / len(times), 3),
                "median_regain_s": round(sorted(times)[len(times) // 2], 3),
                "count": len(times),
                "regain_times": times,
            }
        else:
            result[t] = {
                "avg_regain_s": None,
                "median_regain_s": None,
                "count": 0,
                "regain_times": [],
            }

    return result
