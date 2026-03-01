"""
Stats dashboard — read-only view of computed analytics for a run.

Reads all data from runs/<run_dir>/stats/.  Never recomputes analytics.
Files are lazy-loaded per tab via cached helpers in ui.stats_helpers.
"""
from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from ui.stats_helpers import (
    resolve_run_path,
    stats_path,
    load_json,
    try_json,
    try_parquet,
    df_to_csv_bytes,
    json_to_bytes,
    check_files,
    team_names,
    team_colors,
    _mtime,
)

# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Match Stats", page_icon="📊", layout="wide")

# ── inject minimal premium CSS ──────────────────────────────────────────────

st.markdown("""
<style>
/* KPI cards */
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
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}
/* tabs */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: .88rem !important;
}
/* muted helper text */
.muted { color: #888; font-size: .82rem; }
/* download row */
.dl-row { margin-top: 8px; }
</style>
""", unsafe_allow_html=True)


# ── resolve run_dir ──────────────────────────────────────────────────────────

_raw = st.query_params.get("run_dir") or st.session_state.get("run_dir")

if not _raw:
    st.warning("No run directory found.  Run the analysis pipeline first, then click **Show Stats**.")
    if st.button("Back to video"):
        st.switch_page("app.py")
    st.stop()

run_path = resolve_run_path(_raw)

if not run_path.is_dir():
    st.warning(f"Run directory `{_raw}` does not exist.")
    if st.button("Back to video"):
        st.switch_page("app.py")
    st.stop()

sd = stats_path(run_path)
if not sd.is_dir():
    st.warning(
        f"Stats folder not found in `{run_path}`.  "
        "Compute analytics first:\n\n"
        "```python\n"
        "from analytics.compute_all import compute_possession, compute_physical, compute_shape, compute_ball_movement\n"
        f'compute_possession("{run_path}")\n'
        f'compute_physical("{run_path}")\n'
        f'compute_shape("{run_path}")\n'
        f'compute_ball_movement("{run_path}")\n'
        "```"
    )
    if st.button("Back to video"):
        st.switch_page("app.py")
    st.stop()

st.session_state["run_dir"] = str(run_path)
run_id = run_path.name


# ── load meta (always needed) ───────────────────────────────────────────────

meta_file = run_path / "run_meta.json"
meta: dict = load_json(str(meta_file), _mtime=_mtime(str(meta_file))) if meta_file.is_file() else {}

fps = meta.get("fps", 24)
num_frames = meta.get("num_frames")
duration_s = num_frames / fps if isinstance(num_frames, (int, float)) else None
court_l = meta.get("court_length", "?")
court_w = meta.get("court_width", "?")

TNAMES = team_names(meta)
TCOLORS = team_colors(meta)
_tname_list = list(TNAMES.values())
_tcolor_list = [TCOLORS.get(tid, "#888") for tid in TNAMES]

alt.themes.enable("dark")


# ── formatting helpers ───────────────────────────────────────────────────────

def _pct(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _val(v, fmt=".1f", suffix="") -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    try:
        return f"{float(v):{fmt}}{suffix}"
    except (TypeError, ValueError):
        return "—"


def _download(label, data, filename, mime="text/csv", key=None):
    st.download_button(label, data, file_name=filename, mime=mime, key=key)


# ═══════════════════════════════════════════════════════════════════════════════
# A) HEADER
# ═══════════════════════════════════════════════════════════════════════════════

hdr_left, hdr_right = st.columns([7, 2])

with hdr_left:
    st.markdown(f"## Run `{run_id}`")
    dur_str = f"{duration_s / 60:.1f} min" if duration_s else "—"
    frames_str = f"{num_frames:,}" if isinstance(num_frames, (int, float)) else "—"
    st.markdown(
        f'<span class="muted">{dur_str} &nbsp;·&nbsp; {fps} FPS &nbsp;·&nbsp; '
        f'{frames_str} frames &nbsp;·&nbsp; {court_l} x {court_w} m</span>',
        unsafe_allow_html=True,
    )

with hdr_right:
    if st.button("Back to video"):
        st.switch_page("app.py")
    st.code(str(run_path), language=None)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# B) KPI CARDS ROW
# ═══════════════════════════════════════════════════════════════════════════════

# Pre-load lightweight JSONs needed for KPIs
possession_json = try_json(sd, "possession.json")

# Possession & field tilt from JSON
overall = (possession_json or {}).get("possession_overall", {})
ft_data = (possession_json or {}).get("field_tilt", {})
# If field_tilt is nested (from compute_all), extract inner dict
if isinstance(ft_data, dict) and "field_tilt" in ft_data:
    ft_data = ft_data["field_tilt"]
regain_data = (possession_json or {}).get("time_to_regain", {})

# Distance & speed from parquet (small summary tables — fine to load once)
df_distance = try_parquet(sd, "physical_distance.parquet")
df_speed = try_parquet(sd, "physical_speed.parquet")
df_dims_kpi = try_parquet(sd, "shape_dims.parquet")

k1, k2, k3, k4, k5, k6 = st.columns(6)

# KPI 1 — Possession
poss_str = f'{_pct(overall.get("1"))} / {_pct(overall.get("2"))}'
k1.metric("Possession", poss_str if overall else "—")

# KPI 2 — Field tilt
ft_str = f'{_pct(ft_data.get("1"))} / {_pct(ft_data.get("2"))}'
k2.metric("Field Tilt", ft_str if ft_data else "—")

# KPI 3 — Regain time
r1 = (regain_data.get("1") or {}).get("avg_regain_s")
r2 = (regain_data.get("2") or {}).get("avg_regain_s")
regain_str = f'{_val(r1, ".1f", "s")} / {_val(r2, ".1f", "s")}'
k3.metric("Avg Regain", regain_str if regain_data else "—")

# KPI 4 — Total distance
if df_distance is not None and not df_distance.empty:
    t1_dist = df_distance.loc[df_distance["team"] == 1, "total_distance_m"].sum() / 1000
    t2_dist = df_distance.loc[df_distance["team"] == 2, "total_distance_m"].sum() / 1000
    k4.metric("Distance (km)", f"{t1_dist:.1f} / {t2_dist:.1f}")
else:
    k4.metric("Distance (km)", "—")

# KPI 5 — Top speed
if df_speed is not None and not df_speed.empty:
    idx = df_speed["max_speed_kmh"].idxmax()
    row = df_speed.loc[idx]
    k5.metric("Top Speed", f'{row["max_speed_kmh"]:.1f} km/h',
              delta=f'P{int(row["player_id"])}', delta_color="off")
else:
    k5.metric("Top Speed", "—")

# KPI 6 — Compactness (avg team width x length or hull area)
if df_dims_kpi is not None and not df_dims_kpi.empty:
    avg_w = df_dims_kpi["width_m"].mean()
    avg_l = df_dims_kpi["length_m"].mean()
    k6.metric("Avg Shape", f"{avg_w:.0f}w x {avg_l:.0f}l m")
else:
    k6.metric("Compactness", "—")


# ═══════════════════════════════════════════════════════════════════════════════
# D) ASSUMPTIONS & DATA-QUALITY PANEL
# ═══════════════════════════════════════════════════════════════════════════════

with st.expander("Assumptions & data quality", expanded=False):
    aq1, aq2 = st.columns(2)
    with aq1:
        st.markdown("**Attack direction**")
        st.caption("Team 1 attacks +x in H1, -x in H2; Team 2 is opposite.  Halves split at time midpoint.")
        st.markdown("**Sampling**")
        st.caption(f"Heavy shape stats sampled at 1 Hz (every {fps} frames).")
        st.markdown("**Totals**")
        st.caption(f"Frames: {frames_str}  ·  FPS: {fps}")
        # Missing ball frames
        territory_json = try_json(sd, "ball_territory.json")
        if territory_json:
            skipped = territory_json.get("skipped_frames", "?")
            st.caption(f"Skipped ball frames (missing position): **{skipped}**")
    with aq2:
        st.markdown("**Files found**")
        found = check_files(sd)
        for fname, exists in found.items():
            icon = "yes" if exists else "no"
            st.caption(f"{'[x]' if exists else '[ ]'} `{fname}`")


# ═══════════════════════════════════════════════════════════════════════════════
# C) TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_poss, tab_phys, tab_shape, tab_ball, tab_pressure = st.tabs(
    ["Possession", "Physical", "Shape", "Ball", "Pressure"]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: POSSESSION
# ─────────────────────────────────────────────────────────────────────────────
with tab_poss:
    if possession_json is None:
        st.info("Possession stats not available.  Run `compute_possession(run_dir)` to generate them.")
    else:
        # Lazy-load tab-specific parquets
        df_rolling = try_parquet(sd, "possession_rolling_5min.parquet")
        df_chains = try_parquet(sd, "possession_chains.parquet")

        # ── row 1: rolling + chain histogram ──
        p_c1, p_c2 = st.columns(2)

        with p_c1:
            st.markdown("##### Rolling Possession (5-min)")
            if df_rolling is not None and not df_rolling.empty:
                team_cols = [c for c in df_rolling.columns if c.startswith("team") and c.endswith("_pct")]
                melted = df_rolling.melt(
                    id_vars=["window_start_t"], value_vars=team_cols,
                    var_name="team", value_name="pct",
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
                                        scale=alt.Scale(domain=_tname_list, range=_tcolor_list)),
                        tooltip=["team:N", alt.Tooltip("pct:Q", format=".1%"), alt.Tooltip("min:Q", format=".1f")],
                    )
                    .properties(height=320)
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)
                _download("CSV", df_to_csv_bytes(df_rolling), "possession_rolling.csv", key="dl_rolling")
            else:
                st.caption("Not available.")

        with p_c2:
            st.markdown("##### Chain Duration Distribution")
            if df_chains is not None and not df_chains.empty:
                cdf = df_chains.copy()
                cdf["team_label"] = cdf["team"].map(TNAMES)
                chart = (
                    alt.Chart(cdf)
                    .mark_bar(opacity=0.8, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(
                        x=alt.X("duration_s:Q", bin=alt.Bin(maxbins=30), title="Duration (s)"),
                        y=alt.Y("count()", title="Chains"),
                        color=alt.Color("team_label:N", title="Team",
                                        scale=alt.Scale(domain=_tname_list, range=_tcolor_list)),
                        tooltip=["team_label:N", "count()"],
                    )
                    .properties(height=320)
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)
                _download("CSV", df_to_csv_bytes(df_chains), "possession_chains.csv", key="dl_chains")
            else:
                st.caption("Not available.")

        # ── row 2: callout cards ──
        cs = possession_json.get("chains_summary", {})
        if cs:
            cols = st.columns(len(cs) * 3)
            idx = 0
            for tid, v in cs.items():
                tname = TNAMES.get(int(tid), f"Team {tid}")
                cols[idx].metric(f"{tname} — Avg Duration", _val(v.get("avg_possession_duration_s"), ".2f", "s"))
                cols[idx + 1].metric(f"{tname} — Longest", _val(v.get("longest_possession_duration_s"), ".1f", "s"))
                cols[idx + 2].metric(f"{tname} — # Possessions", v.get("number_of_possessions", "—"))
                idx += 3

        # ── row 3: possession by half ──
        by_half = possession_json.get("possession_by_half", {})
        if by_half:
            st.markdown("##### Possession by Half")
            rows = []
            for half_id, teams in by_half.items():
                for tid, pct in teams.items():
                    rows.append({"half": f"H{half_id}", "team": TNAMES.get(int(tid), f"Team {tid}"), "pct": pct})
            hdf = pd.DataFrame(rows)
            chart = (
                alt.Chart(hdf)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("half:N", title="Half"),
                    y=alt.Y("pct:Q", title="Possession %", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("team:N", title="Team",
                                    scale=alt.Scale(domain=_tname_list, range=_tcolor_list)),
                    xOffset="team:N",
                    tooltip=["half:N", "team:N", alt.Tooltip("pct:Q", format=".1%")],
                )
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)

        # ── row 4: zone shares ──
        zone_poss = possession_json.get("zone_possession", {})
        if zone_poss:
            zp1, zp2 = st.columns(2)
            with zp1:
                thirds = zone_poss.get("thirds", {})
                if thirds:
                    st.markdown("##### Possession by Third")
                    rows = []
                    for tid, zones in thirds.items():
                        for zone, pct in zones.items():
                            rows.append({"team": TNAMES.get(int(tid), f"T{tid}"), "third": zone, "pct": pct})
                    zdf = pd.DataFrame(rows)
                    third_order = ["defensive", "middle", "attacking"]
                    chart = (
                        alt.Chart(zdf)
                        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                        .encode(
                            x=alt.X("third:N", sort=third_order, title="Third"),
                            y=alt.Y("pct:Q", title="Share", scale=alt.Scale(domain=[0, 1])),
                            color=alt.Color("team:N", scale=alt.Scale(domain=_tname_list, range=_tcolor_list)),
                            xOffset="team:N",
                            tooltip=["team:N", "third:N", alt.Tooltip("pct:Q", format=".1%")],
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(chart, use_container_width=True)

            with zp2:
                lanes = zone_poss.get("lanes", {})
                if lanes:
                    st.markdown("##### Possession by Lane")
                    rows = []
                    for tid, lz in lanes.items():
                        for lane, pct in lz.items():
                            rows.append({"team": TNAMES.get(int(tid), f"T{tid}"), "lane": lane, "pct": pct})
                    ldf = pd.DataFrame(rows)
                    lane_order = ["left", "center", "right"]
                    chart = (
                        alt.Chart(ldf)
                        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                        .encode(
                            x=alt.X("lane:N", sort=lane_order, title="Lane"),
                            y=alt.Y("pct:Q", title="Share", scale=alt.Scale(domain=[0, 1])),
                            color=alt.Color("team:N", scale=alt.Scale(domain=_tname_list, range=_tcolor_list)),
                            xOffset="team:N",
                            tooltip=["team:N", "lane:N", alt.Tooltip("pct:Q", format=".1%")],
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(chart, use_container_width=True)

        # ── row 5: field tilt chart ──
        if ft_data:
            st.markdown("##### Field Tilt")
            ftdf = pd.DataFrame([{"team": TNAMES.get(int(k), f"T{k}"), "tilt": v} for k, v in ft_data.items()])
            chart = (
                alt.Chart(ftdf)
                .mark_arc(innerRadius=50)
                .encode(
                    theta=alt.Theta("tilt:Q"),
                    color=alt.Color("team:N", scale=alt.Scale(domain=_tname_list, range=_tcolor_list)),
                    tooltip=["team:N", alt.Tooltip("tilt:Q", format=".1%")],
                )
                .properties(height=220, width=220)
            )
            st.altair_chart(chart)

        # ── row 6: top 10 longest possessions ──
        if df_chains is not None and not df_chains.empty:
            st.markdown("##### Top 10 Longest Possessions")
            top10 = df_chains.nlargest(10, "duration_s").copy()
            top10["team"] = top10["team"].map(TNAMES)
            st.dataframe(
                top10[["chain_id", "team", "start_t", "end_t", "duration_s"]],
                use_container_width=True, hide_index=True,
            )

        # ── row 7: regain time table ──
        tr = possession_json.get("time_to_regain", {})
        if tr:
            st.markdown("##### Time to Regain Possession")
            rows = []
            for tid, v in tr.items():
                rows.append({
                    "Team": TNAMES.get(int(tid), f"T{tid}"),
                    "Avg (s)": v.get("avg_regain_s"),
                    "Median (s)": v.get("median_regain_s"),
                    "Count": v.get("count"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── downloads ──
        st.markdown("---")
        dc1, dc2, dc3 = st.columns(3)
        if df_rolling is not None:
            dc1.download_button("Rolling CSV", df_to_csv_bytes(df_rolling), "rolling_5min.csv", key="dl_r2")
        if df_chains is not None:
            dc2.download_button("Chains CSV", df_to_csv_bytes(df_chains), "chains.csv", key="dl_c2")
        dc3.download_button("possession.json", json_to_bytes(possession_json),
                            "possession.json", mime="application/json", key="dl_pj")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: PHYSICAL
# ─────────────────────────────────────────────────────────────────────────────
with tab_phys:
    # Lazy-load
    _df_dist = try_parquet(sd, "physical_distance.parquet")
    _df_speed = try_parquet(sd, "physical_speed.parquet")

    if _df_dist is None and _df_speed is None:
        st.info("Physical stats not available.  Run `compute_physical(run_dir)` to generate them.")
    else:
        # Controls
        ctrl1, ctrl2 = st.columns([1, 2])
        with ctrl1:
            team_filter = st.selectbox("Team filter", ["All", *_tname_list], key="phys_team")

        def _filter_team(df: pd.DataFrame) -> pd.DataFrame:
            if team_filter == "All" or df is None or df.empty:
                return df
            tid = {v: k for k, v in TNAMES.items()}.get(team_filter)
            return df[df["team"] == tid] if tid is not None else df

        ph1, ph2 = st.columns(2)

        # ── distance leaderboard ──
        with ph1:
            st.markdown("##### Distance Leaderboard")
            if _df_dist is not None and not _df_dist.empty:
                dff = _filter_team(_df_dist).nlargest(10, "total_distance_m").copy()
                dff["label"] = "P" + dff["player_id"].astype(str)
                dff["team_label"] = dff["team"].map(TNAMES)
                chart = (
                    alt.Chart(dff)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("total_distance_m:Q", title="Distance (m)"),
                        y=alt.Y("label:N", sort="-x", title="Player"),
                        color=alt.Color("team_label:N", title="Team",
                                        scale=alt.Scale(domain=_tname_list, range=_tcolor_list)),
                        tooltip=["label:N", "team_label:N",
                                 alt.Tooltip("total_distance_m:Q", format=",.0f"),
                                 alt.Tooltip("distance_per_min_m:Q", format=".1f")],
                    )
                    .properties(height=360)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("Not available.")

        # ── speed bands ──
        with ph2:
            st.markdown("##### Speed Bands — Top 10")
            _df_bands = try_parquet(sd, "physical_bands.parquet")
            if _df_bands is not None and not _df_bands.empty and _df_dist is not None and not _df_dist.empty:
                top_ids = _filter_team(_df_dist).nlargest(10, "total_distance_m")["player_id"].tolist()
                sub = _df_bands[_df_bands["player_id"].isin(top_ids)].copy()
                sub["label"] = "P" + sub["player_id"].astype(str)
                band_order = ["walk", "jog", "run", "sprint"]
                band_colors = ["#6c7a89", "#4c9aff", "#ffab40", "#ff4444"]
                chart = (
                    alt.Chart(sub)
                    .mark_bar()
                    .encode(
                        x=alt.X("time_in_band_s:Q", title="Time (s)", stack="zero"),
                        y=alt.Y("label:N",
                                sort=alt.EncodingSortField(field="time_in_band_s", op="sum", order="descending"),
                                title="Player"),
                        color=alt.Color("band:N", title="Band", sort=band_order,
                                        scale=alt.Scale(domain=band_order, range=band_colors)),
                        tooltip=["label:N", "band:N", alt.Tooltip("time_in_band_s:Q", format=".1f")],
                    )
                    .properties(height=360)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("Not available.")

        # ── accel / decel ──
        _df_accel = try_parquet(sd, "physical_accel.parquet")
        if _df_accel is not None and not _df_accel.empty:
            st.markdown("##### Acceleration & Deceleration Events")
            acc = _filter_team(_df_accel).nlargest(10, "accel_events").copy()
            acc["label"] = "P" + acc["player_id"].astype(str)
            acc["team_label"] = acc["team"].map(TNAMES)
            melted = acc.melt(
                id_vars=["label", "team_label"],
                value_vars=["accel_events", "decel_events"],
                var_name="type", value_name="count",
            )
            melted["type"] = melted["type"].str.replace("_events", "").str.title()
            chart = (
                alt.Chart(melted)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("count:Q", title="Events"),
                    y=alt.Y("label:N", sort="-x", title="Player"),
                    color=alt.Color("type:N", title="Type",
                                    scale=alt.Scale(domain=["Accel", "Decel"], range=["#4caf50", "#f44336"])),
                    xOffset="type:N",
                    tooltip=["label:N", "type:N", "count:Q"],
                )
                .properties(height=360)
            )
            st.altair_chart(chart, use_container_width=True)

        # ── heatmap ──
        _df_heatmap = try_parquet(sd, "physical_heatmap.parquet")
        if _df_heatmap is not None and not _df_heatmap.empty:
            st.markdown("##### Player Heatmap")
            heat_df = _filter_team(_df_heatmap)
            all_players = sorted(heat_df["player_id"].unique())
            if all_players:
                labels = [f"P{pid}" for pid in all_players]
                sel = st.selectbox("Select player", labels, index=0, key="heat_player")
                sel_pid = all_players[labels.index(sel)]
                sub = heat_df[heat_df["player_id"] == sel_pid]
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

        # ── avg position scatter ──
        _df_avgpos = try_parquet(sd, "physical_avgpos.parquet")
        if _df_avgpos is not None and not _df_avgpos.empty:
            st.markdown("##### Average Positions")
            ap = _filter_team(_df_avgpos).dropna(subset=["avg_x", "avg_y"]).copy()
            if not ap.empty:
                ap["label"] = "P" + ap["player_id"].astype(str)
                ap["team_label"] = ap["team"].map(TNAMES)
                chart = (
                    alt.Chart(ap)
                    .mark_circle(size=120, opacity=0.85)
                    .encode(
                        x=alt.X("avg_x:Q", title="X (m)"),
                        y=alt.Y("avg_y:Q", title="Y (m)", scale=alt.Scale(reverse=True)),
                        color=alt.Color("team_label:N", scale=alt.Scale(domain=_tname_list, range=_tcolor_list)),
                        tooltip=["label:N", "team_label:N",
                                 alt.Tooltip("avg_x:Q", format=".1f"),
                                 alt.Tooltip("avg_y:Q", format=".1f")],
                    )
                    .properties(height=340)
                )
                st.altair_chart(chart, use_container_width=True)

        # ── merged player summary table ──
        st.markdown("##### Player Summary")
        parts = []
        if _df_dist is not None and not _df_dist.empty:
            parts.append(_filter_team(_df_dist)[["player_id", "team", "total_distance_m", "distance_per_min_m"]])
        if _df_speed is not None and not _df_speed.empty:
            parts.append(_filter_team(_df_speed)[["player_id", "max_speed_kmh", "avg_active_speed_kmh"]])
        if _df_accel is not None and not _df_accel.empty:
            parts.append(_filter_team(_df_accel)[["player_id", "accel_events", "decel_events"]])

        if parts:
            merged = parts[0]
            for p in parts[1:]:
                merged = merged.merge(p, on="player_id", how="outer")
            merged = merged.sort_values("total_distance_m", ascending=False, na_position="last").reset_index(drop=True)
            merged["team"] = merged["team"].map(TNAMES)
            st.dataframe(merged, use_container_width=True, hide_index=True)
            _download("Player summary CSV", df_to_csv_bytes(merged), "player_summary.csv", key="dl_psummary")

        # ── downloads ──
        st.markdown("---")
        dl_cols = st.columns(4)
        if _df_dist is not None:
            dl_cols[0].download_button("Distance CSV", df_to_csv_bytes(_df_dist), "distance.csv", key="dl_dist")
        if _df_speed is not None:
            dl_cols[1].download_button("Speed CSV", df_to_csv_bytes(_df_speed), "speed.csv", key="dl_spd")
        if _df_bands is not None:
            dl_cols[2].download_button("Bands CSV", df_to_csv_bytes(_df_bands), "speed_bands.csv", key="dl_bands")
        if _df_accel is not None:
            dl_cols[3].download_button("Accel CSV", df_to_csv_bytes(_df_accel), "accel.csv", key="dl_acc")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: SHAPE
# ─────────────────────────────────────────────────────────────────────────────
with tab_shape:
    _df_dims = try_parquet(sd, "shape_dims.parquet")
    if _df_dims is None:
        st.info("Shape stats not available.  Run `compute_shape(run_dir)` to generate them.")
    else:
        # ── width / length over time ──
        st.markdown("##### Team Width & Length Over Time")
        if not _df_dims.empty:
            melted = _df_dims.melt(
                id_vars=["frame", "t", "team"],
                value_vars=["width_m", "length_m"],
                var_name="dimension", value_name="meters",
            )
            melted["team_label"] = melted["team"].map(TNAMES)
            melted["min"] = melted["t"] / 60
            melted["series"] = melted["dimension"] + " — " + melted["team_label"]

            chart = (
                alt.Chart(melted)
                .mark_line(strokeWidth=1.5, opacity=0.85)
                .encode(
                    x=alt.X("min:Q", title="Minute"),
                    y=alt.Y("meters:Q", title="Meters"),
                    color=alt.Color("series:N", title="Metric"),
                    strokeDash=alt.StrokeDash("dimension:N",
                                               scale=alt.Scale(domain=["width_m", "length_m"],
                                                               range=[[1, 0], [4, 4]])),
                    tooltip=["series:N", alt.Tooltip("meters:Q", format=".1f"), alt.Tooltip("min:Q", format=".1f")],
                )
                .properties(height=360)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

        # ── hull area over time ──
        _df_area = try_parquet(sd, "shape_area.parquet")
        if _df_area is not None and not _df_area.empty:
            st.markdown("##### Convex Hull Area Over Time")
            adf = _df_area.dropna(subset=["area_m2"]).copy()
            adf["team_label"] = adf["team"].map(TNAMES)
            adf["min"] = adf["t"] / 60
            chart = (
                alt.Chart(adf)
                .mark_line(strokeWidth=1.5)
                .encode(
                    x=alt.X("min:Q", title="Minute"),
                    y=alt.Y("area_m2:Q", title="Area (m\u00b2)"),
                    color=alt.Color("team_label:N", scale=alt.Scale(domain=_tname_list, range=_tcolor_list)),
                    tooltip=["team_label:N", alt.Tooltip("area_m2:Q", format=".1f"), alt.Tooltip("min:Q", format=".1f")],
                )
                .properties(height=320)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

        # ── defensive line height ──
        _df_defline = try_parquet(sd, "shape_def_line.parquet")
        if _df_defline is not None and not _df_defline.empty:
            st.markdown("##### Defensive Line Height Over Time")
            dldf = _df_defline.copy()
            dldf["team_label"] = dldf["team"].map(TNAMES)
            dldf["min"] = dldf["t"] / 60
            chart = (
                alt.Chart(dldf)
                .mark_line(strokeWidth=1.5)
                .encode(
                    x=alt.X("min:Q", title="Minute"),
                    y=alt.Y("def_line_x:Q", title="Def Line X (m)"),
                    color=alt.Color("team_label:N", scale=alt.Scale(domain=_tname_list, range=_tcolor_list)),
                    tooltip=["team_label:N", alt.Tooltip("def_line_x:Q", format=".2f"), alt.Tooltip("min:Q", format=".1f")],
                )
                .properties(height=320)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

        # ── inter-line distances ──
        _df_linedist = try_parquet(sd, "shape_line_dist.parquet")
        if _df_linedist is not None and not _df_linedist.empty:
            st.markdown("##### Inter-Line Distances Over Time")
            lddf = _df_linedist.dropna(subset=["back_mid_m", "mid_front_m"]).copy()
            lddf["team_label"] = lddf["team"].map(TNAMES)
            lddf["min"] = lddf["t"] / 60
            melted = lddf.melt(
                id_vars=["min", "team_label"],
                value_vars=["back_mid_m", "mid_front_m"],
                var_name="gap", value_name="meters",
            )
            melted["series"] = melted["gap"] + " — " + melted["team_label"]
            chart = (
                alt.Chart(melted)
                .mark_line(strokeWidth=1.5, opacity=0.8)
                .encode(
                    x=alt.X("min:Q", title="Minute"),
                    y=alt.Y("meters:Q", title="Gap (m)"),
                    color=alt.Color("series:N", title="Gap"),
                    tooltip=["series:N", alt.Tooltip("meters:Q", format=".1f"), alt.Tooltip("min:Q", format=".1f")],
                )
                .properties(height=320)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

        # ── most-stretched moments ──
        st.markdown("##### Most Stretched Moments")
        if _df_area is not None and not _df_area.empty:
            stretched = _df_area.dropna(subset=["area_m2"]).nlargest(10, "area_m2").copy()
            stretched["team"] = stretched["team"].map(TNAMES)
            st.dataframe(stretched[["frame", "t", "team", "area_m2"]], use_container_width=True, hide_index=True)
        elif not _df_dims.empty:
            combo = _df_dims.copy()
            combo["spread"] = combo["width_m"] * combo["length_m"]
            stretched = combo.nlargest(10, "spread").copy()
            stretched["team"] = stretched["team"].map(TNAMES)
            st.dataframe(stretched[["frame", "t", "team", "width_m", "length_m", "spread"]],
                         use_container_width=True, hide_index=True)

        # ── centroid summary ──
        _df_centroid = try_parquet(sd, "shape_centroid.parquet")
        if _df_centroid is not None and not _df_centroid.empty:
            st.markdown("##### Average Centroid per Team")
            avg_c = _df_centroid.groupby("team").agg(avg_cx=("cx", "mean"), avg_cy=("cy", "mean")).reset_index()
            avg_c["team"] = avg_c["team"].map(TNAMES)
            st.dataframe(avg_c.round(2), use_container_width=True, hide_index=True)

        # ── downloads ──
        st.markdown("---")
        dl_cols = st.columns(4)
        if _df_dims is not None:
            dl_cols[0].download_button("Dims CSV", df_to_csv_bytes(_df_dims), "shape_dims.csv", key="dl_dims")
        if _df_area is not None:
            dl_cols[1].download_button("Area CSV", df_to_csv_bytes(_df_area), "shape_area.csv", key="dl_area")
        if _df_defline is not None:
            dl_cols[2].download_button("Def Line CSV", df_to_csv_bytes(_df_defline), "def_line.csv", key="dl_def")
        if _df_linedist is not None:
            dl_cols[3].download_button("Line Dist CSV", df_to_csv_bytes(_df_linedist), "line_dist.csv", key="dl_ldist")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: BALL
# ─────────────────────────────────────────────────────────────────────────────
with tab_ball:
    _df_ball = try_parquet(sd, "ball_frame.parquet")
    if _df_ball is None:
        st.info("Ball stats not available.  Run `compute_ball_movement(run_dir)` to generate them.")
    else:
        # ── ball speed over time ──
        st.markdown("##### Ball Speed Over Time")
        if not _df_ball.empty:
            bsdf = _df_ball.dropna(subset=["ball_speed_mps"]).copy()
            bsdf["min"] = bsdf["t"] / 60
            chart = (
                alt.Chart(bsdf)
                .mark_line(strokeWidth=1, opacity=0.7, color="#ffa726")
                .encode(
                    x=alt.X("min:Q", title="Minute"),
                    y=alt.Y("ball_speed_mps:Q", title="Speed (m/s)"),
                    tooltip=[alt.Tooltip("min:Q", format=".2f"), alt.Tooltip("ball_speed_mps:Q", format=".2f")],
                )
                .properties(height=280)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

        bc1, bc2 = st.columns(2)

        # ── territory stacked bar ──
        with bc1:
            st.markdown("##### Ball Territory (thirds)")
            _territory_json = try_json(sd, "ball_territory.json")
            if _territory_json:
                thirds = _territory_json.get("thirds_overall", {})
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
                        .properties(height=300)
                    )
                    st.altair_chart(chart, use_container_width=True)

                dz = _territory_json.get("danger_zone_frames", {})
                if dz:
                    st.caption(
                        f"Danger zone (x=0 goal): **{_pct(dz.get('near_x0_pct'))}** · "
                        f"Danger zone (x=max goal): **{_pct(dz.get('near_xmax_pct'))}**"
                    )
            else:
                st.caption("Territory data not available.")

        # ── switches bar ──
        with bc2:
            st.markdown("##### Switches of Play")
            _df_sw = try_parquet(sd, "ball_switches.parquet")
            if _df_sw is not None and not _df_sw.empty:
                st.metric("Total Switches", len(_df_sw))
                # direction breakdown
                dir_counts = _df_sw.groupby(["lane_from", "lane_to"]).size().reset_index(name="count")
                dir_counts["direction"] = dir_counts["lane_from"] + " -> " + dir_counts["lane_to"]
                chart = (
                    alt.Chart(dir_counts)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#ab47bc")
                    .encode(
                        x=alt.X("direction:N", title="Direction"),
                        y=alt.Y("count:Q", title="Switches"),
                        tooltip=["direction:N", "count:Q"],
                    )
                    .properties(height=240)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("No switches detected.")

        # ── directness histogram ──
        _df_bm = try_parquet(sd, "ball_possession_metrics.parquet")
        if _df_bm is not None and not _df_bm.empty:
            st.markdown("##### Directness Distribution")
            ddf = _df_bm.dropna(subset=["directness"]).copy()
            ddf["team_label"] = ddf["team"].map(TNAMES)
            chart = (
                alt.Chart(ddf)
                .mark_bar(opacity=0.75, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("directness:Q", bin=alt.Bin(maxbins=25), title="Directness Index"),
                    y=alt.Y("count()", title="Chains"),
                    color=alt.Color("team_label:N", scale=alt.Scale(domain=_tname_list, range=_tcolor_list)),
                    tooltip=["team_label:N", "count()"],
                )
                .properties(height=300)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

            # ── top 10 most direct possessions ──
            st.markdown("##### Top 10 Most Direct Possessions")
            top_direct = ddf.nlargest(10, "directness").copy()
            top_direct["team"] = top_direct["team"].map(TNAMES)
            display_cols = ["chain_id", "team", "duration_s", "forward_m", "total_dist_m", "directness"]
            display_cols = [c for c in display_cols if c in top_direct.columns]
            st.dataframe(top_direct[display_cols], use_container_width=True, hide_index=True)

        # ── downloads ──
        st.markdown("---")
        dl_cols = st.columns(4)
        if _df_ball is not None:
            dl_cols[0].download_button("Ball frame CSV", df_to_csv_bytes(_df_ball), "ball_frame.csv", key="dl_bf")
        if _df_bm is not None:
            dl_cols[1].download_button("Metrics CSV", df_to_csv_bytes(_df_bm), "ball_metrics.csv", key="dl_bm")
        if _df_sw is not None:
            dl_cols[2].download_button("Switches CSV", df_to_csv_bytes(_df_sw), "switches.csv", key="dl_sw")
        _tj = try_json(sd, "ball_territory.json")
        if _tj:
            dl_cols[3].download_button("Territory JSON", json_to_bytes(_tj),
                                       "ball_territory.json", mime="application/json", key="dl_tj")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: PRESSURE (placeholder)
# ─────────────────────────────────────────────────────────────────────────────
with tab_pressure:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1e1e2e, #2a2a3c);
        border: 1px solid rgba(255,255,255,.06);
        border-radius: 16px;
        padding: 40px 32px;
        text-align: center;
        margin: 24px 0;
    ">
        <h3 style="margin:0 0 12px">Pressure Metrics</h3>
        <p style="color:#a0a0b8; margin:0 0 20px">Not generated for this run.</p>
        <div style="text-align:left; max-width:400px; margin:auto; color:#a0a0b8; font-size:.9rem">
            <p>Future metrics will include:</p>
            <ul>
                <li>PPDA (passes allowed per defensive action)</li>
                <li>Pressing intensity per 5-min window</li>
                <li>High-press trigger detection</li>
                <li>Counter-press success rate</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST (manual steps)
# ─────────────────────────────────────────────────────────────────────────────
# 1. streamlit run app.py
# 2. Run the pipeline on any video (sample is fine with stubs).
# 3. After completion, click "Show Stats" button.
# 4. Verify: header shows run_id, duration, FPS, frames, pitch.
# 5. Verify: 6 KPI cards render (some may show "—" if stats not computed).
# 6. Open "Assumptions & data quality" expander — check file list.
# 7. Click each tab; charts and tables should render or show "Not available".
# 8. Click download buttons; CSVs and JSONs should download.
# 9. Click "Back to video" — should return to main page.
