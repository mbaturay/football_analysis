"""
Pitch calibration helpers: frame extraction, overlay drawing, homography preview.

Pure functions (no Streamlit imports) so they can be tested independently.
"""
from __future__ import annotations

import cv2
import numpy as np


# ── calibration methods ──────────────────────────────────────────────────────

METHOD_TEMPLATE1 = "Template 1: Halfway + Center Circle"
METHOD_MANUAL = "Manual: 4 Corners"
CALIBRATION_METHODS = [METHOD_TEMPLATE1, METHOD_MANUAL]

# ── vertex / point labels per method ─────────────────────────────────────────

VERTEX_LABELS = ["Near Left", "Far Left", "Far Right", "Near Right"]

TEMPLATE1_LABELS = [
    "Halfway × left touchline",
    "Halfway × right touchline",
    "Centre circle left apex",
    "Centre circle right apex",
]

VERTEX_HELP = (
    "Point order (looking at the camera image):\n"
    "  1. Near Left  — bottom-left corner of visible pitch\n"
    "  2. Far Left   — top-left corner\n"
    "  3. Far Right  — top-right corner\n"
    "  4. Near Right — bottom-right corner\n"
    "\n'Near' = closer to camera (bottom of image).\n"
    "'Far'  = further from camera (top of image)."
)

TEMPLATE1_HELP = (
    "Click 4 points on the halfway line & centre circle:\n"
    "  1. Halfway × left touchline  — where halfway meets left sideline\n"
    "  2. Halfway × right touchline — where halfway meets right sideline\n"
    "  3. Centre circle left apex   — leftmost point of the circle\n"
    "  4. Centre circle right apex  — rightmost point of the circle\n"
    "\n'Left' and 'right' refer to what you see in the image.\n"
    "The circle apexes are the widest points of the circle\n"
    "in the direction perpendicular to the halfway line.\n"
    "\nRecommended: easiest when the camera shows the centre clearly."
)

CENTER_CIRCLE_RADIUS = 9.15  # FIFA standard centre circle radius in meters


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
    labels: list[str] | None = None,
) -> np.ndarray:
    """Draw numbered calibration points on an RGB frame. Returns a copy."""
    img = frame_rgb.copy()
    for i, (px, py) in enumerate(points):
        color = POINT_COLORS[i % len(POINT_COLORS)]
        cv2.circle(img, (int(px), int(py)), radius, color, -1)
        cv2.circle(img, (int(px), int(py)), radius + 2, (255, 255, 255), 2)
        if labels and i < len(labels):
            label = f"{i + 1}. {labels[i]}"
        elif i < 4:
            label = f"{i + 1}. {VERTEX_LABELS[i]}"
        else:
            label = f"P{i + 1}"
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


def build_homography(
    method: str,
    image_points: list[tuple[float, float]],
    pitch_length: float,
    pitch_width: float,
) -> np.ndarray | None:
    """Compute homography from image points to pitch coords for any method.

    Template 1 mapping (Halfway + Center Circle):
        1. Left touchline × halfway  → (L/2, 0)
        2. Right touchline × halfway → (L/2, W)
        3. Centre circle top         → (L/2, W/2 - R)   [top in image = smaller y on pitch]
        4. Centre circle bottom      → (L/2, W/2 + R)

    Manual 4 Corners mapping:
        Same as compute_homography(): NL→(0,W), FL→(0,0), FR→(L,0), NR→(L,W)
    """
    if len(image_points) < 4:
        return None

    src = np.array(image_points[:4], dtype=np.float32)
    L, W, R = pitch_length, pitch_width, CENTER_CIRCLE_RADIUS

    if method == METHOD_TEMPLATE1:
        dst = np.array([
            [L / 2, 0],          # halfway × left touchline
            [L / 2, W],          # halfway × right touchline
            [L / 2 - R, W / 2],  # centre circle left apex (toward left goal)
            [L / 2 + R, W / 2],  # centre circle right apex (toward right goal)
        ], dtype=np.float32)
    else:  # METHOD_MANUAL or fallback
        dst = np.array([
            [0, W],
            [0, 0],
            [L, 0],
            [L, W],
        ], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst)
    return H


def autofill_pixel_corners(
    H: np.ndarray,
    pitch_length: float,
    pitch_width: float,
) -> list[tuple[float, float]] | None:
    """Project the 4 pitch corners back to image coords using H_inv.

    Returns pixel vertices in standard order:
        Near Left→(0,W), Far Left→(0,0), Far Right→(L,0), Near Right→(L,W)
    or None if the homography is singular.
    """
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    L, W = pitch_length, pitch_width
    # Pitch corners in standard order: NL, FL, FR, NR
    pitch_corners = np.array([
        [0, W],      # Near Left
        [0, 0],      # Far Left
        [L, 0],      # Far Right
        [L, W],      # Near Right
    ], dtype=np.float32).reshape(-1, 1, 2)

    img_pts = cv2.perspectiveTransform(pitch_corners, H_inv).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in img_pts]


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
    method: str = METHOD_MANUAL,
) -> tuple[np.ndarray, list[str]]:
    """Draw projected pitch grid onto frame. Returns (overlaid_image, warnings)."""
    warnings: list[str] = []
    H = build_homography(method, pixel_points, pitch_length, pitch_width)
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
    method: str = METHOD_MANUAL,
    image_points: list[tuple[float, float]] | None = None,
    autofilled_corners: list[tuple[float, float]] | None = None,
) -> dict:
    """Build a serializable calibration dict.

    pixel_points: the final 4 corner vertices used for the transform
                  (either manually picked or auto-filled from homography).
    image_points: the raw clicked points (may differ from pixel_points for Template 1).
    autofilled_corners: if auto-fill was used, the projected corners.
    """
    named = {}
    for i, label in enumerate(VERTEX_LABELS):
        key = label.lower().replace(" ", "_")
        named[key] = list(pixel_points[i]) if i < len(pixel_points) else None

    config: dict = {
        "pitch_length_m": pitch_length,
        "pitch_width_m": pitch_width,
        "calibration_method": method,
        "pixel_vertices": [list(p) for p in pixel_points[:4]],
        "pixel_vertices_named": named,
        "vertex_order_note": "near_left, far_left, far_right, near_right (clockwise from bottom-left)",
    }

    if image_points is not None:
        labels = TEMPLATE1_LABELS if method == METHOD_TEMPLATE1 else VERTEX_LABELS
        config["image_points"] = [list(p) for p in image_points[:4]]
        config["image_point_labels"] = labels[:len(image_points)]

    if autofilled_corners is not None:
        config["autofilled_corners"] = [list(p) for p in autofilled_corners]

    return config


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
