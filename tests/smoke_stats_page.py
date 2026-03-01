"""Smoke test: create synthetic run data + verify Stats page & helpers compile."""
import json
import os
import sys
import py_compile

import numpy as np
import pandas as pd


def main():
    # 1. Check altair
    try:
        import altair as alt
        print("altair:", alt.__version__)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "altair", "-q"])
        import altair as alt
        print("altair installed:", alt.__version__)

    # 2. Create synthetic run dir with ALL expected stats files
    run_dir = "runs/_smoke_test_stats"
    stats_dir = os.path.join(run_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    meta = {"fps": 24, "num_frames": 2400, "court_length": 23.32, "court_width": 68}
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f)

    # possession.json (with zone_possession + field_tilt nested)
    poss = {
        "possession_overall": {"1": 0.55, "2": 0.45},
        "possession_by_half": {"1": {"1": 0.52, "2": 0.48}, "2": {"1": 0.58, "2": 0.42}},
        "chains_summary": {
            "1": {"avg_possession_duration_s": 4.2, "longest_possession_duration_s": 18.0, "number_of_possessions": 30},
            "2": {"avg_possession_duration_s": 3.8, "longest_possession_duration_s": 15.0, "number_of_possessions": 28},
        },
        "time_to_regain": {
            "1": {"avg_regain_s": 5.1, "median_regain_s": 4.0, "count": 25},
            "2": {"avg_regain_s": 6.2, "median_regain_s": 5.5, "count": 22},
        },
        "zone_possession": {
            "thirds": {"1": {"defensive": 0.3, "middle": 0.4, "attacking": 0.3},
                       "2": {"defensive": 0.35, "middle": 0.35, "attacking": 0.3}},
            "lanes": {"1": {"left": 0.3, "center": 0.4, "right": 0.3},
                      "2": {"left": 0.35, "center": 0.35, "right": 0.3}},
        },
        "field_tilt": {"1": 0.58, "2": 0.42},
    }
    with open(os.path.join(stats_dir, "possession.json"), "w") as f:
        json.dump(poss, f)

    # rolling
    pd.DataFrame({
        "window_start_t": [0, 300, 600],
        "window_end_t": [300, 600, 900],
        "team1_pct": [0.55, 0.48, 0.62],
        "team2_pct": [0.45, 0.52, 0.38],
    }).to_parquet(os.path.join(stats_dir, "possession_rolling_5min.parquet"), index=False)

    # chains
    pd.DataFrame({
        "chain_id": range(10), "team": [1, 2] * 5,
        "start_frame": range(0, 100, 10), "end_frame": range(9, 109, 10),
        "start_t": np.linspace(0, 9, 10), "end_t": np.linspace(0.9, 9.9, 10),
        "duration_s": [0.9] * 10,
    }).to_parquet(os.path.join(stats_dir, "possession_chains.parquet"), index=False)

    # distance
    pd.DataFrame({
        "player_id": range(1, 11), "team": [1] * 5 + [2] * 5,
        "total_distance_m": np.random.uniform(800, 1200, 10).round(2),
        "minutes_played": [10] * 10,
        "distance_per_min_m": np.random.uniform(80, 120, 10).round(2),
    }).to_parquet(os.path.join(stats_dir, "physical_distance.parquet"), index=False)

    # speed
    pd.DataFrame({
        "player_id": range(1, 11), "team": [1] * 5 + [2] * 5,
        "max_speed_kmh": np.random.uniform(25, 35, 10).round(2),
        "avg_active_speed_kmh": np.random.uniform(8, 15, 10).round(2),
    }).to_parquet(os.path.join(stats_dir, "physical_speed.parquet"), index=False)

    # bands
    rows = []
    for pid in range(1, 11):
        for band, lo, hi in [("walk", 0, 7), ("jog", 7, 14), ("run", 14, 20), ("sprint", 20, None)]:
            rows.append({
                "player_id": pid, "team": 1 if pid <= 5 else 2,
                "band": band, "band_lo_kmh": lo, "band_hi_kmh": hi,
                "frames": int(np.random.uniform(50, 300)),
                "time_in_band_s": round(np.random.uniform(2, 12), 2),
            })
    pd.DataFrame(rows).to_parquet(os.path.join(stats_dir, "physical_bands.parquet"), index=False)

    # accel
    pd.DataFrame({
        "player_id": range(1, 11), "team": [1] * 5 + [2] * 5,
        "accel_events": np.random.randint(5, 30, 10),
        "decel_events": np.random.randint(5, 30, 10),
        "max_accel_ms2": np.random.uniform(3, 6, 10).round(3),
        "max_decel_ms2": np.random.uniform(-6, -3, 10).round(3),
    }).to_parquet(os.path.join(stats_dir, "physical_accel.parquet"), index=False)

    # avgpos
    pd.DataFrame({
        "player_id": range(1, 11), "team": [1] * 5 + [2] * 5,
        "avg_x": np.random.uniform(3, 20, 10).round(3),
        "avg_y": np.random.uniform(10, 58, 10).round(3),
        "valid_frames": [200] * 10,
    }).to_parquet(os.path.join(stats_dir, "physical_avgpos.parquet"), index=False)

    # heatmap
    rows = []
    for pid in range(1, 11):
        for bx in range(24):
            for by in range(16):
                if np.random.random() < 0.15:
                    rows.append({
                        "player_id": pid, "team": 1 if pid <= 5 else 2,
                        "bin_x": bx, "bin_y": by,
                        "count_frames": int(np.random.uniform(1, 20)),
                    })
    pd.DataFrame(rows).to_parquet(os.path.join(stats_dir, "physical_heatmap.parquet"), index=False)

    # shape dims
    rows = []
    for f in range(0, 100, 10):
        for team in [1, 2]:
            rows.append({
                "frame": f, "t": f / 24 * 10, "team": team,
                "width_m": round(np.random.uniform(20, 40), 3),
                "length_m": round(np.random.uniform(15, 30), 3),
            })
    pd.DataFrame(rows).to_parquet(os.path.join(stats_dir, "shape_dims.parquet"), index=False)

    # shape area
    rows = []
    for f in range(0, 100, 10):
        for team in [1, 2]:
            rows.append({
                "frame": f, "t": f / 24 * 10, "team": team,
                "area_m2": round(np.random.uniform(100, 400), 3),
            })
    pd.DataFrame(rows).to_parquet(os.path.join(stats_dir, "shape_area.parquet"), index=False)

    # shape def_line
    rows = []
    for f in range(0, 100, 10):
        for team in [1, 2]:
            rows.append({
                "frame": f, "t": f / 24 * 10, "team": team,
                "def_line_x": round(np.random.uniform(3, 20), 3),
            })
    pd.DataFrame(rows).to_parquet(os.path.join(stats_dir, "shape_def_line.parquet"), index=False)

    # shape line_dist
    rows = []
    for f in range(0, 100, 10):
        for team in [1, 2]:
            rows.append({
                "frame": f, "t": f / 24 * 10, "team": team,
                "back_mid_m": round(np.random.uniform(5, 15), 3),
                "mid_front_m": round(np.random.uniform(5, 15), 3),
            })
    pd.DataFrame(rows).to_parquet(os.path.join(stats_dir, "shape_line_dist.parquet"), index=False)

    # centroid
    rows = []
    for f in range(0, 100, 10):
        for team in [1, 2]:
            rows.append({
                "frame": f, "t": f / 24 * 10, "team": team,
                "cx": round(np.random.uniform(5, 18), 3),
                "cy": round(np.random.uniform(20, 48), 3),
            })
    pd.DataFrame(rows).to_parquet(os.path.join(stats_dir, "shape_centroid.parquet"), index=False)

    # ball_frame
    pd.DataFrame({
        "frame": range(100),
        "t": np.linspace(0, 10, 100),
        "ball_x": np.cumsum(np.random.randn(100) * 0.3) + 10,
        "ball_y": np.cumsum(np.random.randn(100) * 0.5) + 34,
        "ball_speed_mps": np.random.uniform(0, 8, 100).round(3),
    }).to_parquet(os.path.join(stats_dir, "ball_frame.parquet"), index=False)

    # territory
    terr = {
        "thirds_overall": {"defensive": 0.35, "middle": 0.40, "attacking": 0.25},
        "danger_zone_frames": {
            "near_x0_goal": 10, "near_xmax_goal": 8,
            "total_valid_frames": 100, "near_x0_pct": 0.10, "near_xmax_pct": 0.08,
        },
        "skipped_frames": 5,
    }
    with open(os.path.join(stats_dir, "ball_territory.json"), "w") as f:
        json.dump(terr, f)

    # ball_possession_metrics
    pd.DataFrame({
        "chain_id": range(10), "team": [1, 2] * 5,
        "start_t": np.linspace(0, 9, 10), "end_t": np.linspace(0.9, 9.9, 10),
        "duration_s": [0.9] * 10,
        "start_ball_x": np.random.uniform(5, 18, 10).round(3),
        "end_ball_x": np.random.uniform(5, 18, 10).round(3),
        "forward_m": np.random.uniform(-3, 5, 10).round(3),
        "progression_rate_mps": np.random.uniform(-1, 3, 10).round(3),
        "total_dist_m": np.random.uniform(2, 10, 10).round(3),
        "directness": np.random.uniform(0, 1, 10).round(4),
    }).to_parquet(os.path.join(stats_dir, "ball_possession_metrics.parquet"), index=False)

    # switches
    pd.DataFrame({
        "frame_from": [10, 50], "t_from": [0.4, 2.1],
        "lane_from": ["left", "right"], "frame_to": [30, 70],
        "t_to": [1.25, 2.9], "lane_to": ["right", "left"],
        "elapsed_s": [0.85, 0.8], "team_in_possession": [1, 2],
    }).to_parquet(os.path.join(stats_dir, "ball_switches.parquet"), index=False)

    print("Synthetic run dir created:", run_dir)
    print("Stats files:", sorted(os.listdir(stats_dir)))

    # 3. Compile checks
    import glob

    py_compile.compile("ui/stats_helpers.py", doraise=True)
    print("\nui/stats_helpers.py compiles OK")

    stats_page = glob.glob("pages/1_*Stats.py")[0]
    py_compile.compile(stats_page, doraise=True)
    print("Stats page compiles OK")

    py_compile.compile("app.py", doraise=True)
    print("app.py compiles OK")

    # 4. Test helpers module directly
    sys.path.insert(0, ".")
    from ui.stats_helpers import (
        resolve_run_path, stats_path, check_files, team_names, team_colors,
        EXPECTED_FILES,
    )
    rp = resolve_run_path(run_dir)
    assert rp.is_dir(), f"resolve_run_path failed: {rp}"
    sp = stats_path(rp)
    assert sp.is_dir(), f"stats_path failed: {sp}"

    found = check_files(sp)
    present = sum(found.values())
    print(f"\nFiles check: {present}/{len(EXPECTED_FILES)} present")
    missing = [f for f, ok in found.items() if not ok]
    if missing:
        print(f"  Missing (expected): {missing}")

    tn = team_names({})
    assert tn == {1: "Team 1", 2: "Team 2"}, f"team_names default failed: {tn}"
    tc = team_colors({})
    assert 1 in tc and 2 in tc, f"team_colors default failed: {tc}"

    print("\nAll helper tests passed")

    # 5. Cleanup
    import shutil
    shutil.rmtree(run_dir)
    print(f"Cleaned up {run_dir}")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
