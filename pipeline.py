import json
import cv2
import numpy as np
import pandas as pd
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from analytics_io import create_run_dir, save_run_meta, save_tracks_pickle, write_frames_parquet


def _best_position(track_info):
    """Return the best available (x, y) tuple for a track entry, or (None, None)."""
    for key in ('position_transformed', 'position_adjusted', 'position'):
        pos = track_info.get(key)
        if pos is not None:
            try:
                return float(pos[0]), float(pos[1])
            except (TypeError, IndexError):
                continue
    return None, None


def _build_frames_df(tracks, team_ball_control, fps):
    """Build a DataFrame with one row per frame from completed tracks."""
    num_frames = len(tracks['players'])
    rows = []

    for frame_num in range(num_frames):
        t = frame_num / fps

        # Ball
        ball_info = tracks['ball'][frame_num].get(1, {})
        ball_x, ball_y = _best_position(ball_info)

        # Possession
        team_in_possession = int(team_ball_control[frame_num]) if frame_num < len(team_ball_control) else None

        # Players
        players_list = []
        team_positions = {}  # team_id -> list of (x, y)
        for pid, pinfo in tracks['players'][frame_num].items():
            px, py = _best_position(pinfo)
            team_val = pinfo.get('team')
            player_rec = {
                'id': int(pid),
                'team': int(team_val) if team_val is not None else None,
                'x': float(px) if px is not None else None,
                'y': float(py) if py is not None else None,
                'speed_kmh': float(pinfo['speed']) if pinfo.get('speed') is not None else None,
                'distance_m': float(pinfo['distance']) if pinfo.get('distance') is not None else None,
                'has_ball': bool(pinfo.get('has_ball', False)),
            }
            players_list.append(player_rec)

            if px is not None and py is not None and pinfo.get('team') is not None:
                team_positions.setdefault(pinfo['team'], []).append((px, py))

        # Team centroids
        centroids = {}
        for tid, positions in team_positions.items():
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            centroids[int(tid)] = {'x': float(sum(xs) / len(xs)), 'y': float(sum(ys) / len(ys))}

        rows.append({
            'frame': frame_num,
            't': round(t, 4),
            'team_in_possession': team_in_possession,
            'ball_x': ball_x,
            'ball_y': ball_y,
            'players': json.dumps(players_list),
            'team_centroids': json.dumps(centroids),
        })

    return pd.DataFrame(rows)


def run_pipeline(config, progress_callback=None):
    """
    Run the full football analysis pipeline with configurable parameters.

    config: dict with all pipeline parameters
    progress_callback: callable(step_name: str, progress_fraction: float)
    Returns: (output_video_frames, output_path, run_dir)
    """
    def update_progress(step, frac):
        if progress_callback:
            progress_callback(step, frac)

    # Create run directory
    run_dir = config.get('run_dir') or create_run_dir()

    # Step 1: Read video
    update_progress("Reading video...", 0.0)
    video_frames = read_video(config['input_video_path'])

    # Step 2: Track objects
    update_progress("Running detection & tracking (0%)...", 0.05)
    tracker = Tracker(
        config['model_path'],
        batch_size=config.get('batch_size', 20),
        conf_threshold=config.get('conf_threshold', 0.1),
    )

    def detection_progress(done, total):
        pct = done / total * 100
        # Detection is 5%-40% of total progress
        frac = 0.05 + (done / total) * 0.35
        update_progress(f"Running detection & tracking ({pct:.0f}% — {done}/{total} frames)...", frac)

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=config.get('read_from_tracking_stub', True),
        stub_path=config.get('tracking_stub_path', 'stubs/track_stubs.pkl'),
        progress_callback=detection_progress,
    )
    tracker.add_position_to_tracks(tracks)

    # Clamp tracks to video length (stubs may have more frames than video_frames)
    n_video = len(video_frames)
    for obj_key in tracks:
        if isinstance(tracks[obj_key], list) and len(tracks[obj_key]) > n_video:
            tracks[obj_key] = tracks[obj_key][:n_video]

    # Step 3: Camera movement
    update_progress("Estimating camera movement (0%)...", 0.40)
    camera_movement_estimator = CameraMovementEstimator(
        video_frames[0],
        minimum_distance=config.get('minimum_distance', 5),
        lk_win_size=config.get('lk_win_size', 15),
        lk_max_level=config.get('lk_max_level', 2),
        corner_max_corners=config.get('corner_max_corners', 100),
        corner_quality_level=config.get('corner_quality_level', 0.3),
        corner_min_distance=config.get('corner_min_distance', 3),
        corner_block_size=config.get('corner_block_size', 7),
    )

    def camera_progress(done, total):
        pct = done / total * 100
        # Camera movement is 40%-55% of total progress
        frac = 0.40 + (done / total) * 0.15
        update_progress(f"Estimating camera movement ({pct:.0f}% — {done}/{total} frames)...", frac)

    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=config.get('read_from_camera_stub', True),
        stub_path=config.get('camera_stub_path', 'stubs/camera_movement_stub.pkl'),
        progress_callback=camera_progress,
    )
    # Clamp camera movement to video length (stub may be longer)
    if len(camera_movement_per_frame) > n_video:
        camera_movement_per_frame = camera_movement_per_frame[:n_video]

    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Step 4: View transform
    update_progress("Applying perspective transform...", 0.55)
    pixel_vertices = config.get('pixel_vertices', None)
    target_vertices = config.get('target_vertices', None)
    view_transformer = ViewTransformer(
        court_width=config.get('court_width', 68),
        court_length=config.get('court_length', 23.32),
        pixel_vertices=pixel_vertices,
        target_vertices=target_vertices,
    )
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Step 5: Interpolate ball
    update_progress("Interpolating ball positions...", 0.58)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Step 6: Speed & distance
    update_progress("Calculating speed & distance...", 0.60)
    fps = config.get('frame_rate', 24)
    speed_and_distance_estimator = SpeedAndDistance_Estimator(
        frame_window=config.get('frame_window', 5),
        frame_rate=fps,
    )
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Step 7: Team assignment
    update_progress("Assigning teams...", 0.63)
    team_assigner = TeamAssigner(
        n_clusters=config.get('n_clusters', 2),
        init_method=config.get('init_method', 'k-means++'),
    )
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Step 8: Ball possession
    update_progress("Assigning ball possession...", 0.68)
    player_assigner = PlayerBallAssigner(
        max_player_ball_distance=config.get('max_player_ball_distance', 70),
    )
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Step 9: Save run artifacts
    update_progress("Saving run artifacts...", 0.72)
    num_frames = len(video_frames)

    run_meta = {
        'fps': fps,
        'num_frames': num_frames,
        'court_width': config.get('court_width', 68),
        'court_length': config.get('court_length', 23.32),
        'model_path': config.get('model_path'),
        'conf_threshold': config.get('conf_threshold', 0.1),
        'batch_size': config.get('batch_size', 20),
        'max_player_ball_distance': config.get('max_player_ball_distance', 70),
        'frame_window': config.get('frame_window', 5),
        'input_video_path': config.get('input_video_path'),
        'pixel_vertices': config.get('pixel_vertices'),
    }
    save_run_meta(run_dir, run_meta)
    save_tracks_pickle(run_dir, tracks)

    df_frames = _build_frames_df(tracks, team_ball_control, fps)
    write_frames_parquet(run_dir, df_frames)

    # Step 10: Draw annotations
    update_progress("Drawing annotations...", 0.80)
    draw_config = config.get('draw_config', {})
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control, draw_config=draw_config
    )
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame,
        overlay_config=config.get('camera_overlay_config', {}),
    )
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Step 11: Save video
    update_progress("Saving output video...", 0.90)
    output_path = config.get('output_video_path', 'output_videos/output_video.avi')
    save_video(
        output_video_frames, output_path,
        fps=config.get('output_fps', 24),
        codec=config.get('output_codec', 'XVID'),
    )

    # Step 12: Compute analytics
    update_progress("Computing analytics...", 0.93)
    try:
        from analytics.compute_all import (
            compute_possession, compute_physical,
            compute_shape, compute_ball_movement,
        )
        compute_possession(run_dir)
        update_progress("Computing analytics (physical)...", 0.95)
        compute_physical(run_dir)
        update_progress("Computing analytics (shape)...", 0.97)
        compute_shape(run_dir)
        update_progress("Computing analytics (ball)...", 0.99)
        compute_ball_movement(run_dir)
    except Exception as e:
        # Analytics failure should not block the pipeline
        import warnings
        warnings.warn(f"Analytics computation failed: {e}")

    update_progress("Done!", 1.0)
    return output_video_frames, output_path, run_dir
