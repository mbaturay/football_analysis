"""
Pitch calibration helpers: frame extraction, overlay drawing, homography preview.

Pure functions (no Streamlit imports) so they can be tested independently.
"""
from __future__ import annotations

import cv2
import numpy as np


# ── vertex ordering ──────────────────────────────────────────────────────────

VERTEX_LABELS = ["Near Left", "Far Left", "Far Right", "Near Right"]

VERTEX_HELP = (
    "Point order (looking at the camera image):\n"
    "  1. Near Left  — bottom-left corner of visible pitch\n"
    "  2. Far Left   — top-left corner\n"
    "  3. Far Right  — top-right corner\n"
    "  4. Near Right — bottom-right corner\n"
    "\n'Near' = closer to camera (bottom of image).\n"
    "'Far'  = further from camera (top of image)."
)


# ── frame extraction ─────────────────────────────────────────────────────────

def get_video_info(video_path: str) -> dict:
    """Return basic video metadata without reading all frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False}
    info = {
        "ok": True,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def read_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    """Read a single frame (BGR) by index. Returns None on failure."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to RGB for display."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ── point annotation on frame ────────────────────────────────────────────────

POINT_COLORS = [
    (0, 200, 0),    # Near Left — green
    (0, 120, 255),  # Far Left — blue
    (255, 100, 0),  # Far Right — orange
    (200, 0, 200),  # Near Right — purple
]


def draw_points_on_frame(
    frame_rgb: np.ndarray,
    points: list[tuple[int, int]],
    radius: int = 8,
) -> np.ndarray:
    """Draw numbered calibration points on an RGB frame. Returns a copy."""
    img = frame_rgb.copy()
    for i, (px, py) in enumerate(points):
        color = POINT_COLORS[i % len(POINT_COLORS)]
        cv2.circle(img, (int(px), int(py)), radius, color, -1)
        cv2.circle(img, (int(px), int(py)), radius + 2, (255, 255, 255), 2)
        label = f"{i + 1}. {VERTEX_LABELS[i]}" if i < 4 else f"P{i + 1}"
        cv2.putText(
            img, label,
            (int(px) + radius + 4, int(py) + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            img, label,
            (int(px) + radius + 4, int(py) + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
        )
    # Draw polygon outline if 4 points
    if len(points) >= 4:
        pts = np.array(points[:4], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
    return img


# ── homography + overlay ─────────────────────────────────────────────────────

def compute_homography(
    pixel_points: list[tuple[float, float]],
    pitch_length: float,
    pitch_width: float,
) -> np.ndarray | None:
    """Compute homography from 4 image points to pitch coordinates.

    Point order: Near Left, Far Left, Far Right, Near Right.
    Pitch corners (matching order):
        Near Left  → (0, pitch_width)
        Far Left   → (0, 0)
        Far Right  → (pitch_length, 0)
        Near Right → (pitch_length, pitch_width)

    Returns the 3x3 homography matrix or None on failure.
    """
    if len(pixel_points) < 4:
        return None

    src = np.array(pixel_points[:4], dtype=np.float32)
    dst = np.array([
        [0, pitch_width],
        [0, 0],
        [pitch_length, 0],
        [pitch_length, pitch_width],
    ], dtype=np.float32)

    H, status = cv2.findHomography(src, dst)
    return H


def build_pitch_grid_lines(
    pitch_length: float,
    pitch_width: float,
) -> list[list[tuple[float, float]]]:
    """Return a list of polylines (in pitch meters) forming a simple grid.

    Includes:
      - Boundary rectangle
      - Vertical lines at x = 0, L/4, L/2, 3L/4, L
      - Horizontal lines at y = 0, W/3, 2W/3, W
    """
    L, W = pitch_length, pitch_width
    lines: list[list[tuple[float, float]]] = []

    # Boundary
    lines.append([(0, 0), (L, 0), (L, W), (0, W), (0, 0)])

    # Vertical grid
    for frac in [0.25, 0.5, 0.75]:
        x = L * frac
        lines.append([(x, 0), (x, W)])

    # Horizontal grid
    for frac in [1 / 3, 2 / 3]:
        y = W * frac
        lines.append([(0, y), (L, y)])

    # Centre circle approximation (20 points)
    cx, cy = L / 2, W / 2
    r = min(9.15, L / 8, W / 8)  # 9.15 m standard centre circle
    circle_pts = []
    for i in range(21):
        angle = 2 * np.pi * i / 20
        circle_pts.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
    lines.append(circle_pts)

    return lines


def project_grid_onto_frame(
    frame_rgb: np.ndarray,
    pixel_points: list[tuple[float, float]],
    pitch_length: float,
    pitch_width: float,
) -> tuple[np.ndarray, list[str]]:
    """Draw projected pitch grid onto frame. Returns (overlaid_image, warnings)."""
    warnings: list[str] = []
    H = compute_homography(pixel_points, pitch_length, pitch_width)
    if H is None:
        warnings.append("Could not compute homography.")
        return frame_rgb.copy(), warnings

    # Inverse homography: pitch → image
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        warnings.append("Homography is singular — points may be collinear.")
        return frame_rgb.copy(), warnings

    h, w = frame_rgb.shape[:2]
    img = frame_rgb.copy()

    grid_lines = build_pitch_grid_lines(pitch_length, pitch_width)

    for line in grid_lines:
        pitch_pts = np.array(line, dtype=np.float32).reshape(-1, 1, 2)
        img_pts = cv2.perspectiveTransform(pitch_pts, H_inv)
        img_pts_int = img_pts.reshape(-1, 2).astype(np.int32)
        cv2.polylines(img, [img_pts_int], isClosed=False, color=(0, 255, 255), thickness=2)

    # Sanity: check if projected pitch corners are within image bounds
    pitch_corners = np.array([
        [0, 0], [pitch_length, 0],
        [pitch_length, pitch_width], [0, pitch_width],
    ], dtype=np.float32).reshape(-1, 1, 2)
    proj_corners = cv2.perspectiveTransform(pitch_corners, H_inv).reshape(-1, 2)

    for i, (cx, cy) in enumerate(proj_corners):
        corner_names = ["Far-Left", "Far-Right", "Near-Right", "Near-Left"]
        if cx < -w * 0.5 or cx > w * 1.5 or cy < -h * 0.5 or cy > h * 1.5:
            warnings.append(f"Projected {corner_names[i]} corner ({cx:.0f}, {cy:.0f}) far outside image — check point order.")

    return img, warnings


# ── config persistence ───────────────────────────────────────────────────────

def build_calibration_config(
    pitch_length: float,
    pitch_width: float,
    pixel_points: list[tuple[float, float]],
) -> dict:
    """Build a serializable calibration dict."""
    named = {}
    for i, label in enumerate(VERTEX_LABELS):
        key = label.lower().replace(" ", "_")
        named[key] = list(pixel_points[i]) if i < len(pixel_points) else None

    return {
        "pitch_length_m": pitch_length,
        "pitch_width_m": pitch_width,
        "pixel_vertices": [list(p) for p in pixel_points[:4]],
        "pixel_vertices_named": named,
        "vertex_order_note": "near_left, far_left, far_right, near_right (clockwise from bottom-left)",
    }


def save_calibration_json(path: str, calib: dict) -> None:
    """Write calibration dict as JSON."""
    import json
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)


def load_calibration_json(path: str) -> dict | None:
    """Load calibration dict from JSON. Returns None if file doesn't exist."""
    import json
    import os
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)
