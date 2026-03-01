import cv2
import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def run_pipeline(config, progress_callback=None):
    """
    Run the full football analysis pipeline with configurable parameters.

    config: dict with all pipeline parameters
    progress_callback: callable(step_name: str, progress_fraction: float)
    Returns: (output_video_frames, output_path)
    """
    def update_progress(step, frac):
        if progress_callback:
            progress_callback(step, frac)

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
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Step 4: View transform
    update_progress("Applying perspective transform...", 0.45)
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
    update_progress("Interpolating ball positions...", 0.50)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Step 6: Speed & distance
    update_progress("Calculating speed & distance...", 0.55)
    speed_and_distance_estimator = SpeedAndDistance_Estimator(
        frame_window=config.get('frame_window', 5),
        frame_rate=config.get('frame_rate', 24),
    )
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Step 7: Team assignment
    update_progress("Assigning teams...", 0.60)
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
    update_progress("Assigning ball possession...", 0.70)
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

    # Step 9: Draw annotations
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

    # Step 10: Save video
    update_progress("Saving output video...", 0.95)
    output_path = config.get('output_video_path', 'output_videos/output_video.avi')
    save_video(
        output_video_frames, output_path,
        fps=config.get('output_fps', 24),
        codec=config.get('output_codec', 'XVID'),
    )

    update_progress("Done!", 1.0)
    return output_video_frames, output_path
