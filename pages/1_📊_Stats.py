"""
Stats dashboard — read-only view of computed analytics for a run.
"""
from __future__ import annotations

import json
import os

import altair as alt
import pandas as pd
import streamlit as st

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Match Stats", page_icon="📊", layout="wide")

# ── premium CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* KPI card */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3c 100%);
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 12px;
    padding: 18px 20px 14px;
    box-shadow: 0 2px 12px rgba(0,0,0,.25);
}
div[data-testid="stMetric"] label {
    color: #a0a0b8 !important;
    font-size: .78rem !important;
    text-transform: uppercase;
    letter-spacing: .06em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}
/* tabs */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: .88rem !important;
}
/* back button override */
.back-btn button {
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── resolve run_dir ──────────────────────────────────────────────────────────
run_dir: str | None = st.query_params.get("run_dir") or st.session_state.get("run_dir")

if not run_dir or not os.path.isdir(run_dir):
    st.warning("No run directory found.  Run the analysis pipeline first, then click **Show Stats**.")
    if st.button("Back to video"):
        st.switch_page("app.py")
    st.stop()

stats_dir = os.path.join(run_dir, "stats")
if not os.path.isdir(stats_dir):
    st.warning(
        f"Stats folder not found in `{run_dir}`.  "
        "You may need to compute analytics first:\n\n"
        "```python\n"
        "from analytics.compute_all import compute_possession, compute_physical, compute_shape, compute_ball_movement\n"
        f'compute_possession("{run_dir}")\n'
        f'compute_physical("{run_dir}")\n'
        f'compute_shape("{run_dir}")\n'
        f'compute_ball_movement("{run_dir}")\n'
        "```"
    )
    if st.button("Back to video"):
        st.switch_page("app.py")
    st.stop()

# Persist for session
st.session_state["run_dir"] = run_dir


# ── loaders (read-only, cached) ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _try_json(name: str) -> dict | None:
    p = os.path.join(stats_dir, name)
    return _load_json(p) if os.path.isfile(p) else None


def _try_parquet(name: str) -> pd.DataFrame | None:
    p = os.path.join(stats_dir, name)
    return _load_parquet(p) if os.path.isfile(p) else None


# ── load everything ──────────────────────────────────────────────────────────
meta_path = os.path.join(run_dir, "run_meta.json")
meta = _load_json(meta_path) if os.path.isfile(meta_path) else {}

possession_json = _try_json("possession.json")
territory_json = _try_json("ball_territory.json")

df_rolling = _try_parquet("possession_rolling_5min.parquet")
df_chains = _try_parquet("possession_chains.parquet")
df_distance = _try_parquet("physical_distance.parquet")
df_speed = _try_parquet("physical_speed.parquet")
df_bands = _try_parquet("physical_bands.parquet")
df_heatmap = _try_parquet("physical_heatmap.parquet")
df_dims = _try_parquet("shape_dims.parquet")
df_centroid = _try_parquet("shape_centroid.parquet")
df_ball_frame = _try_parquet("ball_frame.parquet")
df_ball_metrics = _try_parquet("ball_possession_metrics.parquet")
df_switches = _try_parquet("ball_switches.parquet")


# ── helpers ──────────────────────────────────────────────────────────────────

def _fmt_pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "—"


def _fmt_m(v: float | None) -> str:
    return f"{v:,.0f} m" if v is not None else "—"


TEAM_COLORS = {1: "#4c9aff", 2: "#ff6b6b"}
TEAM_NAMES = {1: "Team 1", 2: "Team 2"}

alt.themes.enable("dark")


# ── header ───────────────────────────────────────────────────────────────────

col_back, col_title = st.columns([1, 8])
with col_back:
    if st.button("< Back to video"):
        st.switch_page("app.py")
with col_title:
    run_id = os.path.basename(run_dir)
    st.markdown(f"## 📊 Match Stats &nbsp;·&nbsp; `{run_id}`")

fps = meta.get("fps", 24)
num_frames = meta.get("num_frames", "—")
duration_s = num_frames / fps if isinstance(num_frames, (int, float)) else None
pitch = f'{meta.get("court_length", "?")} x {meta.get("court_width", "?")} m'

c1, c2, c3, c4 = st.columns(4)
c1.metric("Duration", f"{duration_s / 60:.1f} min" if duration_s else "—")
c2.metric("FPS", fps)
c3.metric("Frames", f"{num_frames:,}" if isinstance(num_frames, (int, float)) else num_frames)
c4.metric("Pitch", pitch)


# ── KPI row ──────────────────────────────────────────────────────────────────

st.markdown("####")

k1, k2, k3, k4, k5 = st.columns(5)

# Possession
if possession_json:
    overall = possession_json.get("possession_overall", {})
    k1.metric("Possession T1", _fmt_pct(overall.get("1")))
    k2.metric("Possession T2", _fmt_pct(overall.get("2")))
    ft = possession_json.get("field_tilt", {})
    k3.metric("Field Tilt T1", _fmt_pct(ft.get("1")))
else:
    k1.metric("Possession T1", "—")
    k2.metric("Possession T2", "—")
    k3.metric("Field Tilt T1", "—")

# Distance total
if df_distance is not None and not df_distance.empty:
    total_dist = df_distance["total_distance_m"].sum()
    k4.metric("Total Distance", _fmt_m(total_dist))
else:
    k4.metric("Total Distance", "—")

# Max speed
if df_speed is not None and not df_speed.empty:
    max_spd = df_speed["max_speed_kmh"].max()
    k5.metric("Top Speed", f"{max_spd:.1f} km/h" if pd.notna(max_spd) else "—")
else:
    k5.metric("Top Speed", "—")


# ── tabs ─────────────────────────────────────────────────────────────────────

tab_poss, tab_phys, tab_shape, tab_ball, tab_pressure = st.tabs(
    ["Possession", "Physical", "Shape", "Ball", "Pressure"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: POSSESSION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_poss:
    if possession_json is None:
        st.info("Possession stats not computed yet.  Run `compute_possession(run_dir)` first.")
    else:
        col_a, col_b = st.columns(2)

        # Rolling 5-min possession line chart
        with col_a:
            st.markdown("##### Rolling Possession (5-min windows)")
            if df_rolling is not None and not df_rolling.empty:
                team_cols = [c for c in df_rolling.columns if c.startswith("team") and c.endswith("_pct")]
                if team_cols:
                    melted = df_rolling.melt(
                        id_vars=["window_start_t"],
                        value_vars=team_cols,
                        var_name="team",
                        value_name="pct",
                    )
                    melted["team"] = melted["team"].str.replace("team", "Team ").str.replace("_pct", "")
                    melted["min"] = melted["window_start_t"] / 60

                    chart = (
                        alt.Chart(melted)
                        .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=30))
                        .encode(
                            x=alt.X("min:Q", title="Minute"),
                            y=alt.Y("pct:Q", title="Possession %", scale=alt.Scale(domain=[0, 1])),
                            color=alt.Color("team:N", title="Team",
                                            scale=alt.Scale(domain=["Team 1", "Team 2"],
                                                            range=[TEAM_COLORS[1], TEAM_COLORS[2]])),
                            tooltip=["team:N", alt.Tooltip("pct:Q", format=".1%"), "min:Q"],
                        )
                        .properties(height=320)
                        .interactive()
                    )
                    st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("No rolling possession data.")

        # Chain durations histogram
        with col_b:
            st.markdown("##### Possession Chain Duration")
            if df_chains is not None and not df_chains.empty:
                df_chains_disp = df_chains.copy()
                df_chains_disp["team_label"] = df_chains_disp["team"].map(TEAM_NAMES)

                chart = (
                    alt.Chart(df_chains_disp)
                    .mark_bar(opacity=0.8, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(
                        x=alt.X("duration_s:Q", bin=alt.Bin(maxbins=30), title="Duration (s)"),
                        y=alt.Y("count()", title="Chains"),
                        color=alt.Color("team_label:N", title="Team",
                                        scale=alt.Scale(domain=list(TEAM_NAMES.values()),
                                                        range=list(TEAM_COLORS.values()))),
                        tooltip=["team_label:N", "count()"],
                    )
                    .properties(height=320)
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("No chain data.")

        # Summary table
        st.markdown("##### Chains Summary")
        cs = possession_json.get("chains_summary", {})
        if cs:
            rows = []
            for t, v in cs.items():
                rows.append({"Team": TEAM_NAMES.get(int(t), t), **v})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Time to regain
        st.markdown("##### Time to Regain Possession")
        tr = possession_json.get("time_to_regain", {})
        if tr:
            rows = []
            for t, v in tr.items():
                rows.append({"Team": TEAM_NAMES.get(int(t), t),
                             "Avg (s)": v.get("avg_regain_s"), "Median (s)": v.get("median_regain_s"),
                             "Count": v.get("count")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: PHYSICAL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_phys:
    if df_distance is None:
        st.info("Physical stats not computed yet.  Run `compute_physical(run_dir)` first.")
    else:
        col_a, col_b = st.columns(2)

        # Top 10 by distance
        with col_a:
            st.markdown("##### Top 10 — Distance Covered")
            if not df_distance.empty:
                top10 = df_distance.nlargest(10, "total_distance_m").copy()
                top10["label"] = "P" + top10["player_id"].astype(str)
                top10["team_label"] = top10["team"].map(TEAM_NAMES)
                chart = (
                    alt.Chart(top10)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("total_distance_m:Q", title="Distance (m)"),
                        y=alt.Y("label:N", sort="-x", title="Player"),
                        color=alt.Color("team_label:N", title="Team",
                                        scale=alt.Scale(domain=list(TEAM_NAMES.values()),
                                                        range=list(TEAM_COLORS.values()))),
                        tooltip=["label:N", "team_label:N",
                                 alt.Tooltip("total_distance_m:Q", format=",.0f"),
                                 alt.Tooltip("distance_per_min_m:Q", format=".1f")],
                    )
                    .properties(height=360)
                )
                st.altair_chart(chart, use_container_width=True)

        # Speed bands stacked bar (top 10)
        with col_b:
            st.markdown("##### Speed Bands — Top 10 Players")
            if df_bands is not None and not df_bands.empty:
                top_ids = df_distance.nlargest(10, "total_distance_m")["player_id"].tolist() if not df_distance.empty else []
                if top_ids:
                    sub = df_bands[df_bands["player_id"].isin(top_ids)].copy()
                    sub["label"] = "P" + sub["player_id"].astype(str)
                    band_order = ["walk", "jog", "run", "sprint"]
                    band_colors = ["#6c7a89", "#4c9aff", "#ffab40", "#ff4444"]
                    chart = (
                        alt.Chart(sub)
                        .mark_bar()
                        .encode(
                            x=alt.X("time_in_band_s:Q", title="Time (s)", stack="zero"),
                            y=alt.Y("label:N", sort=alt.EncodingSortField(field="time_in_band_s",
                                                                           op="sum", order="descending"),
                                    title="Player"),
                            color=alt.Color("band:N", title="Band",
                                            sort=band_order,
                                            scale=alt.Scale(domain=band_order, range=band_colors)),
                            tooltip=["label:N", "band:N",
                                     alt.Tooltip("time_in_band_s:Q", format=".1f")],
                        )
                        .properties(height=360)
                    )
                    st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("No speed band data.")

        # Heatmap
        st.markdown("##### Player Heatmap")
        if df_heatmap is not None and not df_heatmap.empty:
            all_players = sorted(df_heatmap["player_id"].unique())
            player_labels = {pid: f"P{pid}" for pid in all_players}
            selected_label = st.selectbox("Select player", list(player_labels.values()), index=0)
            selected_pid = [k for k, v in player_labels.items() if v == selected_label][0]

            sub = df_heatmap[df_heatmap["player_id"] == selected_pid]
            if not sub.empty:
                chart = (
                    alt.Chart(sub)
                    .mark_rect(cornerRadius=2)
                    .encode(
                        x=alt.X("bin_x:O", title="X bin"),
                        y=alt.Y("bin_y:O", title="Y bin", sort="descending"),
                        color=alt.Color("count_frames:Q", title="Frames",
                                        scale=alt.Scale(scheme="inferno")),
                        tooltip=["bin_x:O", "bin_y:O", "count_frames:Q"],
                    )
                    .properties(height=340)
                )
                st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("No heatmap data.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: SHAPE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_shape:
    if df_dims is None:
        st.info("Shape stats not computed yet.  Run `compute_shape(run_dir)` first.")
    else:
        st.markdown("##### Team Width & Length Over Time")
        if not df_dims.empty:
            melted = df_dims.melt(
                id_vars=["frame", "t", "team"],
                value_vars=["width_m", "length_m"],
                var_name="dimension",
                value_name="meters",
            )
            melted["team_label"] = melted["team"].map(TEAM_NAMES)
            melted["min"] = melted["t"] / 60
            melted["dim_team"] = melted["dimension"] + " — " + melted["team_label"]

            chart = (
                alt.Chart(melted)
                .mark_line(strokeWidth=1.5, opacity=0.85)
                .encode(
                    x=alt.X("min:Q", title="Minute"),
                    y=alt.Y("meters:Q", title="Meters"),
                    color=alt.Color("dim_team:N", title="Metric"),
                    strokeDash=alt.StrokeDash("dimension:N",
                                               scale=alt.Scale(domain=["width_m", "length_m"],
                                                               range=[[1, 0], [4, 4]])),
                    tooltip=["dim_team:N", alt.Tooltip("meters:Q", format=".1f"), alt.Tooltip("min:Q", format=".1f")],
                )
                .properties(height=360)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("No dimension data.")

        # Centroid table (summary)
        if df_centroid is not None and not df_centroid.empty:
            st.markdown("##### Average Centroid per Team")
            avg_c = df_centroid.groupby("team").agg(avg_cx=("cx", "mean"), avg_cy=("cy", "mean")).reset_index()
            avg_c["team"] = avg_c["team"].map(TEAM_NAMES)
            avg_c = avg_c.round(2)
            st.dataframe(avg_c, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: BALL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ball:
    if df_ball_frame is None:
        st.info("Ball stats not computed yet.  Run `compute_ball_movement(run_dir)` first.")
    else:
        col_a, col_b = st.columns(2)

        # Territory stacked bar
        with col_a:
            st.markdown("##### Ball Territory (thirds)")
            if territory_json:
                thirds = territory_json.get("thirds_overall", {})
                if thirds:
                    tdf = pd.DataFrame([{"third": k, "pct": v} for k, v in thirds.items()])
                    third_order = ["defensive", "middle", "attacking"]
                    third_colors = ["#4c9aff", "#a0a0b8", "#ff6b6b"]
                    chart = (
                        alt.Chart(tdf)
                        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                        .encode(
                            x=alt.X("third:N", sort=third_order, title="Third"),
                            y=alt.Y("pct:Q", title="Fraction", scale=alt.Scale(domain=[0, 1])),
                            color=alt.Color("third:N", sort=third_order, title="Third",
                                            scale=alt.Scale(domain=third_order, range=third_colors)),
                            tooltip=["third:N", alt.Tooltip("pct:Q", format=".1%")],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(chart, use_container_width=True)

                # Danger zones
                dz = territory_json.get("danger_zone_frames", {})
                if dz:
                    st.caption(
                        f"Danger zone (near x=0): **{_fmt_pct(dz.get('near_x0_pct'))}** · "
                        f"Danger zone (near x=max): **{_fmt_pct(dz.get('near_xmax_pct'))}**"
                    )
            else:
                st.caption("No territory data.")

        # Switches of play
        with col_b:
            st.markdown("##### Switches of Play")
            if df_switches is not None and not df_switches.empty:
                st.metric("Total Switches", len(df_switches))
                st.dataframe(
                    df_switches[["frame_from", "t_from", "lane_from", "frame_to", "t_to", "lane_to", "elapsed_s"]].head(20),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No switches detected.")

        # Progression / directness summary
        st.markdown("##### Possession Progression & Directness")
        if df_ball_metrics is not None and not df_ball_metrics.empty:
            summary = (
                df_ball_metrics.dropna(subset=["forward_m"])
                .groupby("team")
                .agg(
                    avg_forward_m=("forward_m", "mean"),
                    avg_rate_mps=("progression_rate_mps", "mean"),
                    avg_directness=("directness", "mean"),
                    chains=("chain_id", "count"),
                )
                .reset_index()
                .round(3)
            )
            summary["team"] = summary["team"].map(TEAM_NAMES)
            st.dataframe(summary, use_container_width=True, hide_index=True)
        else:
            st.caption("No progression data.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: PRESSURE (placeholder)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pressure:
    st.info("Pressure analytics coming soon.")
    st.caption("This tab will include pressing intensity, PPDA, and high-press triggers.")
