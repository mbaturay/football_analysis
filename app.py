import streamlit as st
import tempfile
import os
import numpy as np

st.set_page_config(page_title="Football Analysis", page_icon="⚽", layout="wide")


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


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

# View Transformer / Court
with st.sidebar.expander("🏟️ Court / View Transform"):
    court_width = st.number_input("Court width (m)", min_value=1.0, max_value=200.0, value=68.0, step=0.5)
    court_length = st.number_input("Court length (m)", min_value=1.0, max_value=200.0, value=23.32, step=0.01)

    st.markdown("**Pixel vertices** (4 reference points in video)")
    pv_cols = st.columns(2)
    pv1_x = pv_cols[0].number_input("P1 X", value=110, key="pv1x")
    pv1_y = pv_cols[1].number_input("P1 Y", value=1035, key="pv1y")
    pv2_x = pv_cols[0].number_input("P2 X", value=265, key="pv2x")
    pv2_y = pv_cols[1].number_input("P2 Y", value=275, key="pv2y")
    pv3_x = pv_cols[0].number_input("P3 X", value=910, key="pv3x")
    pv3_y = pv_cols[1].number_input("P3 Y", value=260, key="pv3y")
    pv4_x = pv_cols[0].number_input("P4 X", value=1640, key="pv4x")
    pv4_y = pv_cols[1].number_input("P4 Y", value=915, key="pv4y")

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
            'court_width': court_width,
            'court_length': court_length,
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
