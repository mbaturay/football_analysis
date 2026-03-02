import streamlit as st
import tempfile
import os
import numpy as np

st.set_page_config(page_title="Football Analysis", page_icon="⚽", layout="wide")

CALIB_CONFIG_PATH = "calibration.json"


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def _get_video_path() -> str | None:
    """Resolve the current input video path (uploaded or sample)."""
    if "calib_temp_video" in st.session_state and os.path.exists(st.session_state["calib_temp_video"]):
        return st.session_state["calib_temp_video"]
    if uploaded_video is not None:
        # Write uploaded to a stable temp file for calibration reuse
        if "calib_temp_video" not in st.session_state:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tf.write(uploaded_video.read())
            tf.flush()
            tf.close()
            st.session_state["calib_temp_video"] = tf.name
            uploaded_video.seek(0)  # reset for later reads
        return st.session_state["calib_temp_video"]
    if use_sample_video and os.path.exists("input_videos/08fd33_4.mp4"):
        return "input_videos/08fd33_4.mp4"
    return None


# ── Load saved calibration (if any) ─────────────────────────────────────────

def _load_saved_calibration() -> dict | None:
    from ui.calibration import load_calibration_json
    return load_calibration_json(CALIB_CONFIG_PATH)

_saved_calib = _load_saved_calibration()

# Resolve defaults from saved calibration
_def_pitch_length = 105.0
_def_pitch_width = 68.0
_def_pv = [[110, 1035], [265, 275], [910, 260], [1640, 915]]

if _saved_calib:
    _def_pitch_length = _saved_calib.get("pitch_length_m", _def_pitch_length)
    _def_pitch_width = _saved_calib.get("pitch_width_m", _def_pitch_width)
    if _saved_calib.get("pixel_vertices"):
        _def_pv = _saved_calib["pixel_vertices"]


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("⚽ Football Analysis")
st.sidebar.markdown("Configure settings and run the analysis pipeline.")

# Video I/O
with st.sidebar.expander("📹 Video I/O", expanded=True):
    uploaded_video = st.file_uploader("Upload input video", type=["mp4", "avi", "mov", "mkv"])
    use_sample_video = st.checkbox("Use sample video (input_videos/08fd33_4.mp4)", value=True)
    output_filename = st.text_input("Output filename", value="output_video.mp4")
    output_fps = st.number_input("Output FPS", min_value=1, max_value=120, value=24)
    output_codec = st.selectbox("Output codec", ["mp4v", "XVID", "H264", "avc1"], index=0)

# Detection & Tracking
with st.sidebar.expander("🎯 Detection & Tracking"):
    model_path = st.text_input("YOLO model path", value="models/best.pt")
    conf_threshold = st.slider("Detection confidence", 0.01, 1.0, 0.1, 0.01)
    batch_size = st.number_input("Batch size", min_value=1, max_value=100, value=20)

# Stubs / Cache
with st.sidebar.expander("💾 Stubs / Cache"):
    use_tracking_stub = st.checkbox("Use tracking stubs", value=True)
    tracking_stub_path = st.text_input("Tracking stub path", value="stubs/track_stubs.pkl")
    use_camera_stub = st.checkbox("Use camera movement stubs", value=True)
    camera_stub_path = st.text_input("Camera movement stub path", value="stubs/camera_movement_stub.pkl")

# Ball Assignment
with st.sidebar.expander("⚽ Ball Assignment"):
    max_player_ball_distance = st.slider("Max player-ball distance (px)", 1, 200, 70)

# Speed & Distance
with st.sidebar.expander("🏃 Speed & Distance"):
    frame_window = st.number_input("Frame window", min_value=1, max_value=50, value=5)
    frame_rate = st.number_input("Video frame rate (FPS)", min_value=1, max_value=120, value=24)

# Camera Movement
with st.sidebar.expander("📷 Camera Movement"):
    minimum_distance = st.slider("Min movement distance (px)", 1, 50, 5)
    lk_win_size = st.number_input("LK window size", min_value=3, max_value=51, value=15, step=2)
    lk_max_level = st.number_input("LK max pyramid levels", min_value=0, max_value=10, value=2)
    corner_max_corners = st.number_input("Max corners to detect", min_value=10, max_value=500, value=100)
    corner_quality_level = st.slider("Corner quality level", 0.01, 1.0, 0.3, 0.01)
    corner_min_distance = st.number_input("Corner min distance (px)", min_value=1, max_value=50, value=3)
    corner_block_size = st.number_input("Corner block size", min_value=3, max_value=31, value=7, step=2)

# Team Assignment
with st.sidebar.expander("👕 Team Assignment"):
    n_clusters = st.number_input("K-Means clusters", min_value=2, max_value=5, value=2)
    init_method = st.selectbox("K-Means init method", ["k-means++", "random"], index=0)

# Pitch / View Transform
with st.sidebar.expander("🏟️ Pitch / View Transform"):
    st.caption("Zone, tilt, and shape stats require a valid pitch transform.")
    st.caption("Full pitch is typically ~105 x 68 m.")

    pitch_length = st.number_input(
        "Pitch length (m)", min_value=1.0, max_value=200.0,
        value=_def_pitch_length, step=0.5, key="pitch_length",
    )
    pitch_width = st.number_input(
        "Pitch width (m)", min_value=1.0, max_value=200.0,
        value=_def_pitch_width, step=0.5, key="pitch_width",
    )

    # Validation warnings
    if pitch_length < 60 or pitch_length > 130:
        st.warning(f"Pitch length {pitch_length} m seems wrong for a full pitch (expected 90-120 m).")
    if pitch_width < 40 or pitch_width > 100:
        st.warning(f"Pitch width {pitch_width} m seems wrong for a full pitch (expected 45-90 m).")

    st.markdown("**Pixel vertices** (4 reference points in video)")
    st.caption("Near Left, Far Left, Far Right, Near Right")

    from ui.calibration import VERTEX_LABELS
    pv_cols = st.columns(2)
    pv1_x = pv_cols[0].number_input(f"{VERTEX_LABELS[0]} X", value=int(_def_pv[0][0]), key="pv1x")
    pv1_y = pv_cols[1].number_input(f"{VERTEX_LABELS[0]} Y", value=int(_def_pv[0][1]), key="pv1y")
    pv2_x = pv_cols[0].number_input(f"{VERTEX_LABELS[1]} X", value=int(_def_pv[1][0]), key="pv2x")
    pv2_y = pv_cols[1].number_input(f"{VERTEX_LABELS[1]} Y", value=int(_def_pv[1][1]), key="pv2y")
    pv3_x = pv_cols[0].number_input(f"{VERTEX_LABELS[2]} X", value=int(_def_pv[2][0]), key="pv3x")
    pv3_y = pv_cols[1].number_input(f"{VERTEX_LABELS[2]} Y", value=int(_def_pv[2][1]), key="pv3y")
    pv4_x = pv_cols[0].number_input(f"{VERTEX_LABELS[3]} X", value=int(_def_pv[3][0]), key="pv4x")
    pv4_y = pv_cols[1].number_input(f"{VERTEX_LABELS[3]} Y", value=int(_def_pv[3][1]), key="pv4y")

    if _saved_calib:
        st.success("Loaded saved calibration from calibration.json")

# Visualization
with st.sidebar.expander("🎨 Visualization"):
    referee_color_hex = st.color_picker("Referee color", "#FFFF00")
    ball_color_hex = st.color_picker("Ball indicator color", "#00FF00")
    has_ball_color_hex = st.color_picker("Has-ball indicator color", "#0000FF")
    camera_overlay_alpha = st.slider("Camera movement overlay alpha", 0.0, 1.0, 0.6, 0.05)
    ball_control_alpha = st.slider("Ball control overlay alpha", 0.0, 1.0, 0.4, 0.05)


# ── Main Area ────────────────────────────────────────────────────────────────

st.title("Football Analysis")

# Video preview
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Video")
    if uploaded_video is not None:
        st.video(uploaded_video)
    elif use_sample_video and os.path.exists("input_videos/08fd33_4.mp4"):
        st.video("input_videos/08fd33_4.mp4")
    else:
        st.info("Upload a video or enable the sample video option.")

with col2:
    st.subheader("Output Video")
    output_path = os.path.join("output_videos", output_filename)
    if 'output_video_path' in st.session_state and os.path.exists(st.session_state['output_video_path']):
        st.video(st.session_state['output_video_path'])
        with open(st.session_state['output_video_path'], 'rb') as f:
            st.download_button("Download output video", f, file_name=output_filename, mime="video/mp4")
    else:
        st.info("Run the pipeline to generate the output video.")


# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION MODE
# ═══════════════════════════════════════════════════════════════════════════════

with st.expander("🎯 Pitch Calibration Tool", expanded=False):
    st.caption(
        "Pick a frame with visible touchlines and penalty box lines, "
        "then click 4 pitch corners in order."
    )

    video_path = _get_video_path()

    if video_path is None:
        st.info("Load a video first (upload or enable sample) to use the calibration tool.")
    else:
        from ui.calibration import (
            get_video_info, read_frame, frame_to_rgb,
            draw_points_on_frame, project_grid_onto_frame,
            build_calibration_config, save_calibration_json,
            VERTEX_LABELS, VERTEX_HELP,
        )
        import cv2

        vinfo = get_video_info(video_path)
        if not vinfo["ok"]:
            st.error("Cannot open video file.")
        else:
            total_frames = vinfo["frame_count"]
            vid_fps = vinfo["fps"] or 24

            # ── frame selector ──
            calib_frame_idx = st.slider(
                "Select frame",
                min_value=0, max_value=max(total_frames - 1, 0), value=0,
                key="calib_frame_idx",
                help=f"Video has {total_frames} frames at {vid_fps:.1f} FPS",
            )
            ts = calib_frame_idx / vid_fps
            st.caption(f"Frame {calib_frame_idx} — {ts:.2f} s ({ts / 60:.1f} min)")

            # ── read + display frame with click ──
            frame_bgr = read_frame(video_path, calib_frame_idx)
            if frame_bgr is None:
                st.error(f"Could not read frame {calib_frame_idx}.")
            else:
                frame_rgb = frame_to_rgb(frame_bgr)

                # Initialize calibration points in session state
                if "calib_points" not in st.session_state:
                    st.session_state["calib_points"] = []

                calib_pts = st.session_state["calib_points"]

                # Show how many points collected and what's next
                n_pts = len(calib_pts)
                if n_pts < 4:
                    next_label = VERTEX_LABELS[n_pts]
                    st.info(f"Click on the frame to place point {n_pts + 1}/4: **{next_label}**")
                else:
                    st.success("All 4 points placed. Preview the overlay or save to config.")

                # Draw existing points on frame
                display_img = draw_points_on_frame(frame_rgb, calib_pts)

                # Resize for display (keep aspect, max width ~800)
                h, w = display_img.shape[:2]
                display_width = min(w, 900)
                scale = display_width / w
                display_h = int(h * scale)
                display_resized = cv2.resize(display_img, (display_width, display_h))

                # Click-to-pick using streamlit-image-coordinates
                from streamlit_image_coordinates import streamlit_image_coordinates

                click_result = streamlit_image_coordinates(
                    display_resized,
                    key=f"calib_click_{calib_frame_idx}_{n_pts}",
                )

                if click_result is not None and n_pts < 4:
                    # Scale click back to original resolution
                    cx = click_result["x"] / scale
                    cy = click_result["y"] / scale
                    st.session_state["calib_points"].append((cx, cy))
                    st.rerun()

                # ── point list + controls ──
                if calib_pts:
                    st.markdown("**Selected points:**")
                    for i, (px, py) in enumerate(calib_pts):
                        label = VERTEX_LABELS[i] if i < 4 else f"P{i + 1}"
                        st.caption(f"  {i + 1}. {label}: ({px:.0f}, {py:.0f})")

                btn_cols = st.columns(3)
                with btn_cols[0]:
                    if st.button("Undo last point", disabled=len(calib_pts) == 0, key="calib_undo"):
                        st.session_state["calib_points"].pop()
                        st.rerun()
                with btn_cols[1]:
                    if st.button("Reset all points", disabled=len(calib_pts) == 0, key="calib_reset"):
                        st.session_state["calib_points"] = []
                        st.rerun()

                with st.expander("Point ordering guide", expanded=False):
                    st.text(VERTEX_HELP)

                # ── preview overlay ──
                has_all_4 = len(calib_pts) >= 4
                if st.button(
                    "Preview overlay",
                    disabled=not has_all_4,
                    key="calib_preview",
                    type="primary",
                ):
                    overlay_img, overlay_warnings = project_grid_onto_frame(
                        frame_rgb, calib_pts[:4], pitch_length, pitch_width,
                    )
                    st.image(overlay_img, caption="Projected pitch grid overlay", use_container_width=True)
                    if overlay_warnings:
                        for w_msg in overlay_warnings:
                            st.warning(w_msg)
                    else:
                        st.success("Grid projected successfully — lines should align with visible pitch markings.")

                # ── save to config ──
                if st.button(
                    "Save calibration to config",
                    disabled=not has_all_4,
                    key="calib_save",
                ):
                    calib_dict = build_calibration_config(
                        pitch_length, pitch_width, calib_pts[:4],
                    )
                    save_calibration_json(CALIB_CONFIG_PATH, calib_dict)
                    st.success(f"Calibration saved to `{CALIB_CONFIG_PATH}`.")
                    st.caption("Sidebar values will update on next page load.")
                    # Update session state so sidebar picks up new values on rerun
                    st.session_state["pv1x"] = int(calib_pts[0][0])
                    st.session_state["pv1y"] = int(calib_pts[0][1])
                    st.session_state["pv2x"] = int(calib_pts[1][0])
                    st.session_state["pv2y"] = int(calib_pts[1][1])
                    st.session_state["pv3x"] = int(calib_pts[2][0])
                    st.session_state["pv3y"] = int(calib_pts[2][1])
                    st.session_state["pv4x"] = int(calib_pts[3][0])
                    st.session_state["pv4y"] = int(calib_pts[3][1])
                    st.session_state["pitch_length"] = pitch_length
                    st.session_state["pitch_width"] = pitch_width


st.markdown("---")

# Run button
if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
    # Determine input video path
    input_video_path = None
    temp_video_file = None

    if uploaded_video is not None:
        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video_file.write(uploaded_video.read())
        temp_video_file.flush()
        temp_video_file.close()
        input_video_path = temp_video_file.name
    elif use_sample_video:
        input_video_path = "input_videos/08fd33_4.mp4"

    if input_video_path is None or not os.path.exists(input_video_path):
        st.error("No input video available. Please upload a video or enable the sample video option.")
    elif not os.path.exists(model_path):
        st.error(f"YOLO model not found at: {model_path}")
    else:
        # Build config
        pixel_vertices = np.array([
            [pv1_x, pv1_y], [pv2_x, pv2_y],
            [pv3_x, pv3_y], [pv4_x, pv4_y]
        ])

        # Stubs are only valid for the sample video — disable when using a different video
        is_sample_video = (uploaded_video is None and use_sample_video)
        effective_use_tracking_stub = use_tracking_stub and is_sample_video
        effective_use_camera_stub = use_camera_stub and is_sample_video

        if not is_sample_video and (use_tracking_stub or use_camera_stub):
            st.warning("Stubs auto-disabled: they were generated from the sample video and don't match your uploaded video.")

        config = {
            'input_video_path': input_video_path,
            'model_path': model_path,
            'batch_size': batch_size,
            'conf_threshold': conf_threshold,
            'read_from_tracking_stub': effective_use_tracking_stub,
            'tracking_stub_path': tracking_stub_path,
            'read_from_camera_stub': effective_use_camera_stub,
            'camera_stub_path': camera_stub_path,
            'max_player_ball_distance': max_player_ball_distance,
            'frame_window': frame_window,
            'frame_rate': frame_rate,
            'minimum_distance': minimum_distance,
            'lk_win_size': lk_win_size,
            'lk_max_level': lk_max_level,
            'corner_max_corners': corner_max_corners,
            'corner_quality_level': corner_quality_level,
            'corner_min_distance': corner_min_distance,
            'corner_block_size': corner_block_size,
            'n_clusters': n_clusters,
            'init_method': init_method,
            # Both old and new keys for backward compatibility
            'court_width': pitch_width,
            'court_length': pitch_length,
            'pitch_width_m': pitch_width,
            'pitch_length_m': pitch_length,
            'pixel_vertices': pixel_vertices,
            'output_video_path': output_path,
            'output_fps': output_fps,
            'output_codec': output_codec,
            'draw_config': {
                'referee_color': hex_to_bgr(referee_color_hex),
                'ball_color': hex_to_bgr(ball_color_hex),
                'has_ball_color': hex_to_bgr(has_ball_color_hex),
                'ball_control_overlay': {'alpha': ball_control_alpha},
            },
            'camera_overlay_config': {'alpha': camera_overlay_alpha},
        }

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def progress_callback(step_name, fraction):
            progress_bar.progress(min(fraction, 1.0))
            status_text.text(step_name)

        try:
            from pipeline import run_pipeline
            _, result_path, run_dir = run_pipeline(config, progress_callback=progress_callback)
            st.session_state['output_video_path'] = result_path
            st.session_state['run_dir'] = run_dir
            st.success(f"Analysis complete! Run artifacts saved to `{run_dir}`")
            st.rerun()
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            if temp_video_file is not None:
                try:
                    os.unlink(temp_video_file.name)
                except PermissionError:
                    pass

# ── Show Stats button (after pipeline has completed) ────────────────────────
if 'run_dir' in st.session_state and st.session_state['run_dir']:
    run_dir = st.session_state['run_dir']
    stats_dir = os.path.join(run_dir, "stats")
    if os.path.isdir(stats_dir):
        st.markdown("---")
        if st.button("📊 Show Stats", type="primary", use_container_width=True):
            st.query_params["run_dir"] = run_dir
            st.switch_page("pages/1_📊_Stats.py")
