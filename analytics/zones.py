"""
Pitch zone definitions and zone-based analytics.

Coordinate convention (from ViewTransformer defaults):
  - x axis: court_length direction (0 .. court_length)
  - y axis: court_width  direction (0 .. court_width)

Thirds are split along x (length); lanes are split along y (width).
"""

from __future__ import annotations

from analytics.quality import CONFIDENCE_THRESHOLD


# ── zone helpers ─────────────────────────────────────────────────────────────

def third_boundaries(court_length: float) -> list[float]:
    """Return [0, 1/3, 2/3, full] breakpoints along x."""
    t = court_length / 3
    return [0.0, t, 2 * t, court_length]


def lane_boundaries(court_width: float) -> list[float]:
    """Return [0, 1/3, 2/3, full] breakpoints along y."""
    t = court_width / 3
    return [0.0, t, 2 * t, court_width]


THIRD_LABELS = ["defensive", "middle", "attacking"]
LANE_LABELS = ["left", "center", "right"]


def classify_third(x: float, court_length: float) -> str | None:
    """Return 'defensive', 'middle', or 'attacking' for a raw x value.

    'defensive' is nearest x=0, 'attacking' is nearest x=court_length.
    Returns None if x is outside [0, court_length].
    """
    if x is None:
        return None
    bounds = third_boundaries(court_length)
    for i, label in enumerate(THIRD_LABELS):
        if bounds[i] <= x <= bounds[i + 1]:
            return label
    return None


def classify_lane(y: float, court_width: float) -> str | None:
    """Return 'left', 'center', or 'right' for a raw y value."""
    if y is None:
        return None
    bounds = lane_boundaries(court_width)
    for i, label in enumerate(LANE_LABELS):
        if bounds[i] <= y <= bounds[i + 1]:
            return label
    return None


# ── zone possession ──────────────────────────────────────────────────────────

def compute_possession_in_zones(df_frames, meta: dict, confidence: float = 1.0) -> dict:
    """Possession share broken down by thirds and lanes for each team.

    Returns::

        {
            "thirds": {
                1: {"defensive": 0.xx, "middle": 0.xx, "attacking": 0.xx},
                2: { ... },
            },
            "lanes": {
                1: {"left": 0.xx, "center": 0.xx, "right": 0.xx},
                2: { ... },
            },
            "skipped_frames": <int>,
        }
    """
    if confidence < CONFIDENCE_THRESHOLD:
        return {"thirds": {}, "lanes": {}, "skipped_frames": len(df_frames),
                "skipped": True, "reason": f"Transform confidence {confidence:.0%} below threshold"}

    court_length = meta.get("court_length", 23.32)
    court_width = meta.get("court_width", 68)

    teams = sorted(df_frames["team_in_possession"].dropna().unique())
    if not teams:
        return {"thirds": {}, "lanes": {}, "skipped_frames": len(df_frames)}

    valid = df_frames.dropna(subset=["ball_x", "ball_y", "team_in_possession"])
    skipped = len(df_frames) - len(valid)

    thirds = {int(t): {l: 0 for l in THIRD_LABELS} for t in teams}
    lanes = {int(t): {l: 0 for l in LANE_LABELS} for t in teams}

    for _, row in valid.iterrows():
        team = int(row["team_in_possession"])
        trd = classify_third(row["ball_x"], court_length)
        lan = classify_lane(row["ball_y"], court_width)
        if trd is not None:
            thirds[team][trd] += 1
        if lan is not None:
            lanes[team][lan] += 1

    # normalise per team
    for t in teams:
        t = int(t)
        total_t = sum(thirds[t].values()) or 1
        thirds[t] = {k: round(v / total_t, 4) for k, v in thirds[t].items()}
        total_l = sum(lanes[t].values()) or 1
        lanes[t] = {k: round(v / total_l, 4) for k, v in lanes[t].items()}

    return {"thirds": thirds, "lanes": lanes, "skipped_frames": skipped}


# ── field tilt ───────────────────────────────────────────────────────────────

def compute_field_tilt(df_frames, meta: dict, confidence: float = 1.0) -> dict:
    """Field tilt: share of possession frames where the ball is in the
    attacking third for each team.

    Assumption (placeholder):
        - team 1 attacks towards x = court_length in the first half,
          towards x = 0 in the second half.
        - team 2 is the opposite.

    The midpoint of time splits the halves.
    """
    if confidence < CONFIDENCE_THRESHOLD:
        return {"field_tilt": {}, "skipped_frames": len(df_frames),
                "skipped": True, "reason": f"Transform confidence {confidence:.0%} below threshold",
                "attack_direction_assumption": _DIRECTION_NOTE}

    court_length = meta.get("court_length", 23.32)
    fps = meta.get("fps", 24)

    valid = df_frames.dropna(subset=["ball_x", "team_in_possession"])
    skipped = len(df_frames) - len(valid)
    if valid.empty:
        return {
            "field_tilt": {},
            "skipped_frames": skipped,
            "attack_direction_assumption": _DIRECTION_NOTE,
        }

    mid_t = valid["t"].max() / 2
    thirds_bound = court_length * 2 / 3  # attacking third starts at 2/3

    tilt_counts = {}  # team -> frames_in_attacking_third
    total_counts = {}  # team -> total_possession_frames

    for _, row in valid.iterrows():
        team = int(row["team_in_possession"])
        bx = row["ball_x"]
        half = 1 if row["t"] <= mid_t else 2

        total_counts[team] = total_counts.get(team, 0) + 1

        in_attacking = False
        if team == 1:
            in_attacking = (bx >= thirds_bound) if half == 1 else (bx <= court_length - thirds_bound)
        elif team == 2:
            in_attacking = (bx <= court_length - thirds_bound) if half == 1 else (bx >= thirds_bound)

        if in_attacking:
            tilt_counts[team] = tilt_counts.get(team, 0) + 1

    field_tilt = {}
    for team in sorted(total_counts):
        field_tilt[team] = round(
            tilt_counts.get(team, 0) / total_counts[team], 4
        )

    return {
        "field_tilt": field_tilt,
        "skipped_frames": skipped,
        "attack_direction_assumption": _DIRECTION_NOTE,
    }


_DIRECTION_NOTE = (
    "team 1 attacks +x in first half, -x in second half; "
    "team 2 is opposite. Halves split at time midpoint."
)
