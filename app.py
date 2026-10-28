import os
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.ml_engine import ScoutBrain
from src.club_analytics import ClubAnalytics

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Garuda Scout AI v6.0",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS: COMPACT & MODERN DASHBOARD UI ---
st.markdown(
    """
    <style>
    /* Kurangi padding atas/bawah halaman utama agar lebih muat banyak di layar */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    /* Ukuran judul utama & subjudul lebih proporsional */
    h1 {
        font-size: 1.65rem !important;
        margin-bottom: 0.1rem !important;
        padding-bottom: 0rem !important;
    }
    h2 {
        font-size: 1.25rem !important;
        margin-top: 0.4rem !important;
        margin-bottom: 0.3rem !important;
    }
    h3 {
        font-size: 1.05rem !important;
        margin-top: 0.3rem !important;
        margin-bottom: 0.2rem !important;
    }
    /* Compact KPI Metric Cards - cegah teks terpotong (...) */
    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 8px;
        padding: 8px 12px !important;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        margin-bottom: 2px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.72rem !important;
    }
    /* Buat Tab Menu lebih rapat agar seluruh 7 tab muat tanpa scroll horizontal */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 36px !important;
        padding: 3px 10px !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }
    /* Rapatkan jarak antar widget di Sidebar */
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
    }
    hr {
        margin-top: 0.75rem !important;
        margin-bottom: 0.75rem !important;
    }
    /* Form Badges Styling */
    .badge-win {
        background-color: #28a745; color: white; padding: 2px 7px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; margin-right: 3px;
    }
    .badge-draw {
        background-color: #ffc107; color: black; padding: 2px 7px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; margin-right: 3px;
    }
    .badge-loss {
        background-color: #dc3545; color: white; padding: 2px 7px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; margin-right: 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- HEADER COMPACT ---
st.title("🇮🇩 Garuda Scout AI v6.0")
st.caption(
    "Pro Football Scouting & Intelligence Suite — Liga 1, Liga 2, Liga 3 & ASEAN | "
    "Multi-Attribute ML, Moneyball Valuation, Match Analytics & Regulasi PSSI/LIB"
)


# --- LOAD DATA & ML ENGINE (CACHED) ---
@st.cache_resource(show_spinner="Memuat database 3.000+ pemain & statistik 104 klub...")
def load_scout_brain():
    return ScoutBrain()


try:
    brain = load_scout_brain()
    df = brain.df
    club_engine = brain.club_analytics
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()


def format_idr(val):
    """Format nilai Rupiah secara ringkas agar tidak terpotong di kartu metrik."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "-"
    if v <= 0:
        return "-"
    if v >= 1e9:
        return f"Rp {v / 1e9:,.2f} M"
    if v >= 1e6:
        return f"Rp {v / 1e6:,.0f} Jt"
    return f"Rp {v:,.0f}"


def generate_ai_scouting_report(player_row, radar_info, similar_df, api_key=None):
    """
    Menghasilkan Executive Scouting Report untuk pemain individu.
    """
    p_name = player_row["player_name"]
    p_team = player_row["team"]
    p_league = player_row["league"]
    p_pos = player_row["position"]
    p_age = int(player_row["age"]) if player_row["age"] > 0 else int(player_row["age_clean"])
    p_height = int(player_row["height_cm"]) if player_row["height_cm"] > 0 else int(player_row["height_clean"])
    p_foot = str(player_row.get("foot", "-")).capitalize()
    p_nat = player_row.get("nationality", "Indonesia")
    p_quota = player_row.get("quota_status", "Lokal")
    p_mv_raw = player_row.get("market_value_raw", "-")
    p_mv_est = float(player_row.get("market_value_est", 0))
    p_exp_mv = float(player_row.get("expected_mv_est", 0))
    p_underval = float(player_row.get("undervalued_ratio", 1.0))
    p_contract = player_row.get("contract_expiry", "-")
    p_months = int(player_row.get("contract_months_left", -1))
    p_archetype = player_row.get("player_archetype", "⚙️ Squad Rotation")
    p_trend = player_row.get("mv_trend", "STABLE")
    p_growth = float(player_row.get("mv_growth_pct", 0.0))

    metrics = radar_info.get("metrics", {}) if radar_info else {}
    sim_names = []
    if similar_df is not None and not similar_df.empty:
        for _, srow in similar_df.head(3).iterrows():
            sim_names.append(
                f"{srow['player_name']} ({srow['team']} - {srow['market_value_raw']}, "
                f"Similarity {srow['similarity_score']}%)"
            )

    if api_key and api_key.strip():
        prompt = f"""Anda adalah Chief Football Scout & Direktur Teknik profesional untuk klub Liga 1 Indonesia.
Buatlah Executive Scouting Report yang tajam, terstruktur, dan berbasis data dalam Bahasa Indonesia untuk pemain berikut:

- Nama Pemain: {p_name}
- Klub Saat Ini: {p_team} ({p_league})
- Posisi: {p_pos}
- Usia: {p_age} tahun ({'Memenuhi Regulasi U-22' if p_age <= 22 else 'Senior'})
- Postur & Kaki Dominan: {p_height} cm | Kaki {p_foot}
- Kewarganegaraan & Status Kuota Liga Indonesia: {p_nat} ({p_quota})
- Arketipe ML: {p_archetype}
- Nilai Pasar Saat Ini: {p_mv_raw} ({format_idr(p_mv_est)})
- Expected Market Value (Model RandomForest): {format_idr(p_exp_mv)} (Undervalued Ratio: {p_underval:.2f}x)
- Tren Valuasi: {p_trend} ({p_growth:+.1f}%)
- Kontrak Berakhir: {p_contract} ({f'{p_months} bulan tersisa' if p_months >= 0 else 'Belum dipublikasikan'})
- Skor Persentil 6D Radar (0-100): {json.dumps(metrics)}
- 3 Pemain Pembanding Terdekat (Replacement/Comparable): {'; '.join(sim_names) if sim_names else '-'}

Susun laporan dengan sub-judul Markdown:
1. Ringkasan Eksekutif & Profil Taktikal
2. Analisis Valuasi Moneyball & Status Kontrak
3. Kecocokan Regulasi Liga Indonesia (Kuota {p_quota} / U-22)
4. Rekomendasi Keputusan Transfer (Beli / Pinjam / Pantau / Negosiasi Bosman)"""

        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-3.8-flash:generateContent?key={api_key.strip()}"
            )
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            resp = requests.post(url, json=payload, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"], "Gemini 3.8 Flash AI"
        except Exception:
            pass

    val_verdict = (
        f"**Undervalued ({p_underval:.2f}x)** — Model ML memprediksi nilai wajar mencapai "
        f"**{format_idr(p_exp_mv)}**, di atas harga pasar saat ini (**{p_mv_raw}**)."
        if p_underval >= 1.15
        else (
            f"**Fair Value ({p_underval:.2f}x)** — Valuasi saat ini (**{p_mv_raw}**) sejalan dengan "
            f"ekspektasi model (**{format_idr(p_exp_mv)}**)."
            if p_underval >= 0.85
            else f"**Premium / Marquee ({p_underval:.2f}x)** — Valuasi (**{p_mv_raw}**) "
            f"di atas rata-rata profil usianya (**{format_idr(p_exp_mv)}**), mencerminkan status bintang utama."
        )
    )

    contract_verdict = (
        f"⚠️ **Peluang Bosman / Kontrak Segera Habis ({p_contract} — ~{p_months} bln lagi)**. "
        "Posisi tawar tinggi untuk negosiasi biaya transfer rendah atau pra-kontrak."
        if 0 <= p_months <= 12
        else (
            f"Kontrak tercatat hingga **{p_contract}** (~{p_months} bln tersisa). Membutuhkan skema transfer/pinjaman."
            if p_months > 12
            else "Tanggal akhir kontrak belum dipublikasikan; disarankan verifikasi langsung ke agensi."
        )
    )

    u22_badge = (
        "✅ **Memenuhi Regulasi U-22** (Cocok untuk slot wajib U-22 Liga 1/Liga 2)."
        if p_age <= 22
        else f"Kategori usia **{p_age} tahun** (Usia matang untuk kontribusi langsung)."
    )

    strengths = [k for k, v in metrics.items() if v >= 70]
    strength_str = ", ".join(strengths) if strengths else "Keseimbangan atribut lintas dimensi"

    report_md = f"""#### 📋 Executive Scouting Dossier: **{p_name}**
**Klub:** {p_team} ({p_league}) | **Posisi:** {p_pos} | **Arketipe ML:** {p_archetype}

* **Profil Fisik & Skuad:** Tinggi **{p_height} cm**, kaki **{p_foot}** (Persentil Postur: **{metrics.get('Physical Stature', 50):.1f}%** | *Squad Standing*: **{metrics.get('Squad Standing', 50):.1f}%**). Keunggulan 6D: *{strength_str}*.
* **Valuasi Moneyball & Momentum:** {val_verdict} Tren **{p_trend}** ({p_growth:+.1f}%).
* **Status Kontrak:** {contract_verdict}
* **Regulasi Liga (PSSI/LIB):** Kuota **{p_quota}** ({p_nat}) — {u22_badge}
* **Pemain Pembanding Terdekat:** {' | '.join(sim_names) if sim_names else '-'}
"""
    return report_md, "ScoutBrain v6.0 ML Engine"


# ==========================================
# 🛠️ SIDEBAR GLOBAL FILTER (COMPACT)
# ==========================================
st.sidebar.markdown("### 🛠️ Filter Scouting")

all_leagues = sorted(df["league"].unique())
sel_leagues = st.sidebar.multiselect(
    "1. Kompetisi / Liga",
    options=all_leagues,
    default=all_leagues,
)

df_l1 = df[df["league"].isin(sel_leagues)] if sel_leagues else df.iloc[0:0]

available_teams = sorted(df_l1["team"].unique())
sel_teams = st.sidebar.multiselect("2. Klub (Opsional)", options=available_teams)

df_l2 = df_l1[df_l1["team"].isin(sel_teams)] if sel_teams else df_l1

available_positions = sorted(df_l2["position"].unique())
sel_positions = st.sidebar.multiselect("3. Posisi (Opsional)", options=available_positions)

c_sb_q, c_sb_u = st.sidebar.columns([1.3, 1])
with c_sb_q:
    quota_options = ["Semua", "Lokal", "ASEAN", "Asia (AFC)", "Asing Non-Asia"]
    sel_quota = st.selectbox("4. Kuota", options=quota_options, index=0)
with c_sb_u:
    st.write("")
    st.write("")
    only_u22_sidebar = st.checkbox("👶 U-22", value=False, help="Hanya pemain berusia ≤ 22 tahun")

min_age_db = int(df["age_clean"].min()) if not df.empty else 15
max_age_db = int(df["age_clean"].max()) if not df.empty else 45
age_range = st.sidebar.slider(
    "5. Rentang Usia",
    min_value=min_age_db,
    max_value=max_age_db,
    value=(min_age_db, 22 if only_u22_sidebar else max_age_db),
)

# Terapkan filter ke main_df
main_df = df_l2.copy()
if sel_positions:
    main_df = main_df[main_df["position"].isin(sel_positions)]
if sel_quota != "Semua":
    main_df = main_df[main_df["quota_status"] == sel_quota]
if only_u22_sidebar:
    main_df = main_df[main_df["is_u22"] == True]
main_df = main_df[
    (main_df["age_clean"] >= age_range[0]) & (main_df["age_clean"] <= age_range[1])
]

col_sb1, col_sb2 = st.sidebar.columns(2)
col_sb1.metric("Pemain", f"{len(main_df):,}")
col_sb2.metric("Klub", f"{main_df['team'].nunique()}")

with st.sidebar.expander("🔑 Gemini AI Key (Opsional)"):
    env_key = os.environ.get("GEMINI_API_KEY", "")
    gemini_api_key = st.text_input(
        "GEMINI_API_KEY",
        value=env_key,
        type="password",
        help="Opsional: Isi untuk memakai gemini-3.8-flash pada laporan scouting.",
    )

# ==========================================
# 📑 TABS MENU (7 TABS LENGKAP & COMPACT)
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Pasar & Tren",
    "🏟️ Analisis Klub & Taktik",
    "🤖 Squad Planner",
    "🔄 Cari Pengganti",
    "🕸️ Radar & AI Scout",
    "💎 Gems & Kontrak",
    "📱 Konten & Data",
])

# ==========================================
# TAB 1: MARKET EXPLORER & HISTORICAL TRENDS
# ==========================================
with tab1:
    if not main_df.empty:
        # Compact KPI Row — angka singkat + delta informatif agar tidak pernah terpotong (...)
        k1, k2, k3, k4, k5 = st.columns(5)
        valued_main = main_df[main_df["market_value_est"] > 0]
        avg_val = valued_main["market_value_est"].mean() if not valued_main.empty else 0
        max_val = main_df["market_value_est"].max()
        u22_count = int(main_df["is_u22"].sum())
        u22_pct = (u22_count / len(main_df) * 100) if len(main_df) > 0 else 0
        exp_12m = int(
            ((main_df["contract_months_left"] >= 0) & (main_df["contract_months_left"] <= 12)).sum()
        )

        k1.metric("Pemain Terfilter", f"{len(main_df):,}", f"{main_df['team'].nunique()} Klub")
        k2.metric("Rata-rata Valuasi", format_idr(avg_val))
        k3.metric("Valuasi Tertinggi", format_idr(max_val))
        k4.metric("Talenta U-22", f"{u22_count:,}", f"{u22_pct:.1f}% Skuad")
        k5.metric("Kontrak ≤12 Bln", f"{exp_12m:,}", "Target Bosman")

        # Baris kontrol grafik yang ringkas (1 baris)
        c_title, c_x, c_col = st.columns([1.6, 1, 1.4])
        with c_title:
            st.markdown("#### 🎯 Peta Sebaran Valuasi Pemain")
        with c_x:
            x_axis_mode = st.radio(
                "Sumbu X:",
                ["Posisi", "Usia"],
                horizontal=True,
                key="t1_xaxis",
            )
        with c_col:
            color_mode = st.radio(
                "Warna:",
                ["Liga", "Arketipe ML", "Kuota"],
                horizontal=True,
                key="t1_color",
            )

        x_col = "position" if x_axis_mode == "Posisi" else "age_clean"
        color_col = (
            "league"
            if color_mode == "Liga"
            else ("player_archetype" if color_mode == "Arketipe ML" else "quota_status")
        )

        plot_df = main_df.copy()
        plot_df["size_viz"] = plot_df["market_value_est"].clip(lower=150_000_000)

        fig_scatter = px.scatter(
            plot_df,
            x=x_col,
            y="market_value_est",
            color=color_col,
            size="size_viz",
            hover_name="player_name",
            hover_data={
                "team": True,
                "league": True,
                "position": True,
                "age": True,
                "height_cm": True,
                "foot": True,
                "quota_status": True,
                "market_value_raw": True,
                "player_archetype": True,
                "size_viz": False,
                "market_value_est": False,
            },
            labels={
                "market_value_est": "Valuasi (Rp)",
                "position": "Posisi",
                "age_clean": "Usia (Thn)",
                "league": "Liga",
                "player_archetype": "Arketipe ML",
                "quota_status": "Kuota",
            },
            height=400,
        )
        fig_scatter.update_layout(
            margin=dict(l=10, r=10, t=15, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")
        # --- SUB-BAGIAN: TREN HISTORIS & MARKET MOVERS ---
        col_hist, col_movers = st.columns([1.15, 1.35])

        with col_hist:
            st.markdown("#### 📈 Tren Historis Pemain (25 Snapshot)")
            player_list_t1 = sorted(main_df["player_name"].unique())
            default_idx = 0
            for candidate_default in ["Rizky Ridho", "Thom Haye", "Beckham Putra", "Ramadhan Sananta"]:
                if candidate_default in player_list_t1:
                    default_idx = player_list_t1.index(candidate_default)
                    break

            sel_hist_player = st.selectbox(
                "Pilih Pemain:",
                options=player_list_t1,
                index=default_idx,
                key="t1_hist_player",
                label_visibility="collapsed",
            )
            hist_df = brain.get_player_value_history(sel_hist_player)
            if not hist_df.empty:
                fig_hist = px.line(
                    hist_df,
                    x="date",
                    y="market_value_est",
                    markers=True,
                    hover_data=["team", "market_value_raw"],
                    labels={"date": "Snapshot", "market_value_est": "Valuasi (Rp)"},
                    height=280,
                )
                fig_hist.update_traces(line=dict(width=2.5, color="#00CC96"), marker=dict(size=7))
                fig_hist.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("Data snapshot historis belum tersedia untuk pemain ini.")

        with col_movers:
            c_mv_hdr, c_mv_rad = st.columns([1.2, 1])
            with c_mv_hdr:
                st.markdown("#### 🚀 Top Market Movers")
            with c_mv_rad:
                mover_dir = st.radio(
                    "Arah:",
                    ["📈 Top Gainers", "📉 Top Drops"],
                    horizontal=True,
                    key="t1_mover_dir",
                    label_visibility="collapsed",
                )
            dir_code = "gainers" if "Gainers" in mover_dir else "drops"
            movers_df = brain.get_market_movers(
                direction=dir_code,
                league_filter=sel_leagues if sel_leagues else None,
                top_n=10,
            )
            if not movers_df.empty:
                st.dataframe(
                    movers_df.rename(
                        columns={
                            "player_name": "Pemain",
                            "team": "Klub",
                            "league": "Liga",
                            "position": "Posisi",
                            "age": "Usia",
                            "market_value_raw": "Valuasi",
                            "change_formatted": "Selisih",
                            "mv_growth_pct": "%",
                        }
                    ).drop(columns=["mv_trend"], errors="ignore"),
                    hide_index=True,
                    use_container_width=True,
                    height=280,
                )
            else:
                st.info("Tidak ada data pergerakan nilai pasar pada filter ini.")
    else:
        st.warning("Data kosong. Silakan sesuaikan filter di Sidebar.")

# ==========================================
# TAB 2: ANALISIS PERTANDINGAN & DIAGNOSIS KLUB
# ==========================================
with tab2:
    st.markdown("#### 🏟️ Analisis Statistik Pertandingan & Diagnosis Sektor Klub")
    st.caption(
        "Evaluasi mendalam rekam laga, rasio gol, ketahanan bertahan, dan diagnosis 5 sektor kunci klub "
        "disertai rekomendasi perbaikan taktis dan target rekrutmen konkret dari database 3.060 pemain."
    )

    # 1. Pilihan Liga & Klub Target
    col_sel_lg, col_sel_tm = st.columns([1.2, 2.0])
    with col_sel_lg:
        avail_standings_leagues = sorted(club_engine.df_clubs["league"].unique()) if not club_engine.df_clubs.empty else all_leagues
        sel_club_league = st.selectbox(
            "1. Pilih Kompetisi",
            options=avail_standings_leagues,
            index=0,
            key="t2_club_league",
        )
    with col_sel_tm:
        clubs_in_lg = sorted(club_engine.df_clubs[club_engine.df_clubs["league"] == sel_club_league]["team"].unique()) if not club_engine.df_clubs.empty else sorted(df[df["league"] == sel_club_league]["team"].unique())
        sel_target_club = st.selectbox(
            "2. Pilih Klub yang Akan Dianalisis",
            options=clubs_in_lg,
            index=0,
            key="t2_target_club",
        )

    # Ambil data pertandingan & skuad
    m_info = club_engine.get_club_match_stats(sel_target_club)
    s_info = club_engine.get_club_squad_profile(sel_target_club)
    diag_info = club_engine.evaluate_sectors(sel_target_club)

    if m_info and s_info and diag_info:
        # Header Klub: Logo, Peringkat, Rekor, Form
        col_c_crest, col_c_meta, col_c_form = st.columns([0.6, 2.4, 2.0])
        with col_c_crest:
            crest_url = m_info.get("crest_url")
            if crest_url and str(crest_url).startswith("http"):
                st.image(crest_url, width=65)
            else:
                st.markdown("### 🛡️")
        with col_c_meta:
            st.markdown(f"### {sel_target_club}")
            st.caption(
                f"**Kompetisi:** {m_info['league']} | **Peringkat:** #{m_info['rank']} ({m_info.get('group', 'Utama')}) | "
                f"**Nilai Skuad:** {format_idr(s_info['total_mv'])} ({s_info['squad_size']} Pemain)"
            )
        with col_c_form:
            st.write("**Tren Form Terakhir:**")
            form_str = str(m_info.get("form", "-"))
            badges_html = ""
            for ch in form_str:
                if ch == "W":
                    badges_html += '<span class="badge-win">W</span>'
                elif ch == "D":
                    badges_html += '<span class="badge-draw">D</span>'
                elif ch == "L":
                    badges_html += '<span class="badge-loss">L</span>'
                else:
                    badges_html += f"<span>{ch}</span>"
            if badges_html:
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.write("-")

        # Row 1: KPI Pertandingan Klub
        cm1, cm2, cm3, cm4, cm5 = st.columns(5)
        p_val = m_info["played"]
        w_val, d_val, l_val = m_info["won"], m_info["drawn"], m_info["lost"]
        pts_val = m_info["points"]
        ppg_val = m_info["ppg"]
        gf_val, ga_val, gd_val = m_info["gf"], m_info["ga"], m_info["gd"]
        gf_m = m_info["gf_per_match"]
        ga_m = m_info["ga_per_match"]
        cs_count = m_info.get("clean_sheets", 0)

        cm1.metric("Poin & Ranking", f"{pts_val} Pts", f"Rank #{m_info['rank']} (PPG {ppg_val})")
        cm2.metric("Rekam Laga", f"{p_val} Main", f"{w_val}M - {d_val}S - {l_val}K")
        cm3.metric("Produktivitas Gol", f"{gf_val} Gol", f"{gf_m:.2f} Gol/Laga")
        cm4.metric("Kemasukan & Nirbobol", f"{ga_val} Gol", f"{ga_m:.2f}/laga ({cs_count} Clean Sheets)")
        cm5.metric("Selisih Gol (GD)", f"{gd_val:+d}", f"Win Rate {m_info.get('win_rate', 0)}%")

        # Expandable Klasemen Lengkap Liga
        with st.expander(f"📋 Lihat Klasemen Lengkap {sel_club_league}"):
            df_standings_display = club_engine.get_standings(league=sel_club_league)
            if not df_standings_display.empty:
                st.dataframe(
                    df_standings_display[[
                        "rank", "team", "group", "played", "won", "drawn", "lost",
                        "gf", "ga", "gd", "points", "ppg", "win_rate", "form",
                    ]].rename(
                        columns={
                            "rank": "#", "team": "Klub", "group": "Grup", "played": "Main",
                            "won": "M", "drawn": "S", "lost": "K", "gf": "GM", "ga": "GK",
                            "gd": "SG", "points": "Poin", "ppg": "PPG", "win_rate": "Win%", "form": "Form",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

        st.markdown("---")
        # Row 2: Evaluasi 5 Sektor Lapangan (Radar Chart & Kartu Status Sektor)
        col_sec_rad, col_sec_cards = st.columns([1.15, 1.35])

        with col_sec_rad:
            st.markdown("##### 🕸️ Radar Keseimbangan 5 Sektor Klub")
            sec_dict = diag_info["sectors"]
            radar_cats = list(sec_dict.keys())
            club_scores = [sec_dict[k]["score"] for k in radar_cats]

            # Tutup poligon radar
            radar_cats_closed = radar_cats + [radar_cats[0]]
            club_scores_closed = club_scores + [club_scores[0]]
            benchmark_closed = [75.0] * (len(radar_cats) + 1)

            fig_sec_radar = go.Figure()
            fig_sec_radar.add_trace(
                go.Scatterpolar(
                    r=club_scores_closed,
                    theta=radar_cats_closed,
                    fill="toself",
                    name=sel_target_club,
                    line=dict(color="#00CC96", width=2.5),
                    opacity=0.75,
                )
            )
            fig_sec_radar.add_trace(
                go.Scatterpolar(
                    r=benchmark_closed,
                    theta=radar_cats_closed,
                    name="Standar Solid (75%)",
                    line=dict(color="rgba(255, 165, 0, 0.7)", width=1.5, dash="dash"),
                )
            )
            fig_sec_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[20, 100], ticksuffix="%")),
                height=340,
                margin=dict(l=25, r=25, t=15, b=25),
                legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_sec_radar, use_container_width=True)
            st.caption(f"⚡ **Overall Sector Health Index:** **{diag_info['overall_health_score']}/100**")

        with col_sec_cards:
            st.markdown("##### 🩺 Rapor Kesehatan 5 Sektor Kunci")
            for sec_k, sec_v in sec_dict.items():
                col_c1, col_c2 = st.columns([1.8, 1])
                with col_c1:
                    st.write(f"**{sec_v['name']}**: {sec_v['status']} (Skor: **{sec_v['score']:.1f}**)")
                    st.caption(f"💬 *{sec_v['diagnosis']}*")
                with col_c2:
                    # Ringkasan metrik kunci
                    km = sec_v.get("key_metrics", {})
                    km_txt = " | ".join([f"{k}: {v}" for k, v in list(km.items())[:2]])
                    st.caption(km_txt)
                st.markdown("<hr style='margin: 4px 0 !important;'>", unsafe_allow_html=True)

        st.markdown("---")
        # Row 3: Rekomendasi Solusi & Rekrutmen Spesifik untuk Sektor yang Perlu Diperbaiki
        st.markdown("#### 🚨 Diagnosis Sektor Prioritas & Rekomendasi Target Rekrutmen")
        st.write(
            f"Berdasarkan evaluasi statistik pertandingan dan kedalaman skuad {sel_target_club}, "
            "berikut adalah sektor yang paling mendesak untuk dibenahi serta rekomendasi pemain rekrutan terbaik:"
        )

        recs_list = club_engine.get_actionable_recommendations(sel_target_club)
        for idx, rec in enumerate(recs_list, 1):
            with st.container():
                st.markdown(
                    f"##### Sektor Prioritas #{idx}: {rec['sector_name']} "
                    f"*(Skor: {rec['score']} | Status: {rec['status']})*"
                )
                st.info(f"**Problem Utama:** {rec['diagnosis']}\n\n**Solusi Taktikal:** {rec['tactical_profile']}")

                cand_df = rec.get("candidates")
                if cand_df is not None and not cand_df.empty:
                    st.write("**🎯 Rekomendasi Target Pemain yang Cocok untuk Ditransfer/Dipinjam:**")
                    st.dataframe(
                        cand_df.rename(
                            columns={
                                "player_name": "Nama Pemain",
                                "team": "Klub Asal",
                                "league": "Liga",
                                "position": "Posisi",
                                "age": "Usia",
                                "height_cm": "Tinggi",
                                "foot": "Kaki",
                                "quota_status": "Kuota",
                                "market_value_raw": "Valuasi",
                                "contract_expiry": "Kontrak Habis",
                            }
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.write("Tidak ada kandidat pemain yang cocok dalam batasan budget normal.")

        st.markdown("---")
        # Row 4: AI Executive Tactical Report Generator
        st.markdown("##### 📋 AI Executive Tactical & Club Diagnostic Report")
        c_diag_btn, c_diag_note = st.columns([1.5, 2.5])
        with c_diag_btn:
            gen_club_report = st.button(
                "📝 Buat Laporan Diagnostik Taktikal Klub",
                type="primary",
                use_container_width=True,
                key="btn_club_report",
            )
        with c_diag_note:
            st.caption("Menghasilkan laporan dewan direksi/kepala pelatih mencakup analisis taktik, titik lemah, dan rencana aksi transfer.")

        if gen_club_report:
            with st.spinner(f"Menyusun laporan diagnostik taktikal untuk {sel_target_club}..."):
                rep_md, engine_name = club_engine.generate_club_diagnostic_report(
                    sel_target_club, api_key=gemini_api_key
                )
            st.caption(f"⚡ Dihasilkan oleh: **{engine_name}**")
            st.markdown(rep_md)

    else:
        st.warning("Data pertandingan klub ini sedang dipersiapkan atau belum tersedia.")

# ==========================================
# TAB 3: SMART SQUAD PLANNER & REGULASI
# ==========================================
with tab3:
    st.markdown("#### 🤖 Smart Squad Planner & Filter Regulasi PSSI/LIB")

    c_p1, c_p2, c_p3, c_p4, c_p5, c_p6 = st.columns([1.3, 1.2, 1.4, 1.0, 0.9, 1.0])
    with c_p1:
        target_team_planner = st.selectbox(
            "Klub Target",
            options=sorted(df["team"].unique()),
            key="t3_team",
        )
    with c_p2:
        target_pos_planner = st.selectbox(
            "Posisi Dibutuhkan",
            options=sorted(df["position"].unique()),
            key="t3_pos",
        )
    with c_p3:
        strategy_map = {
            "⚖️ Balanced": "balanced",
            "💎 Wonderkid U-23": "wonderkid",
            "🚀 Moneyball": "moneyball",
            "🛡️ Proven (24-30)": "proven",
        }
        sel_strategy_label = st.selectbox(
            "Strategi Transfer",
            options=list(strategy_map.keys()),
            key="t3_strategy",
        )
        sel_strategy = strategy_map[sel_strategy_label]
    with c_p4:
        sel_quota_planner = st.selectbox(
            "Kuota Regulasi",
            options=["Semua", "Lokal", "ASEAN", "Asia (AFC)", "Asing Non-Asia"],
            key="t3_quota",
        )
    with c_p5:
        sel_foot_planner = st.selectbox(
            "Kaki Dominan",
            options=["Semua", "kanan", "kiri", "keduanya"],
            key="t3_foot",
        )
    with c_p6:
        custom_budget_m = st.number_input(
            "Maks Budget (M Rp)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.5,
            key="t3_budget",
            help="0 = Otomatis menyesuaikan rata-rata skuad klub",
        )

    c_act1, c_act2, c_act3 = st.columns([1.2, 3.0, 1.2])
    with c_act1:
        u22_only_planner = st.checkbox(
            "👶 Wajib U-22 (≤ 22 Thn)",
            value=False,
            key="t3_u22",
        )
    with c_act2:
        club_df = df[df["team"] == target_team_planner]
        if not club_df.empty:
            c_avg = club_df[club_df["market_value_est"] > 0]["market_value_est"].mean()
            c_foreign = int((club_df["quota_status"] != "Lokal").sum())
            c_u22 = int(club_df["is_u22"].sum())
            st.caption(
                f"📌 **Skuad {target_team_planner}** ({club_df['league'].iloc[0]}): "
                f"**{len(club_df)} Pemain** | Rerata: **{format_idr(c_avg)}** | "
                f"Asing/ASEAN: **{c_foreign}** | U-22: **{c_u22}**"
            )
    with c_act3:
        run_planner = st.button(
            "🔍 Cari Kandidat",
            type="primary",
            use_container_width=True,
            key="btn_planner",
        )

    if run_planner:
        custom_budget_val = custom_budget_m * 1e9 if custom_budget_m > 0 else None
        with st.spinner("Menghitung kesesuaian taktik, regulasi, dan budget..."):
            recs_planner = brain.recommend_for_team_needs(
                team_name=target_team_planner,
                target_position=target_pos_planner,
                top_n=10,
                quota_filter=sel_quota_planner,
                u22_only=u22_only_planner,
                preferred_foot=sel_foot_planner,
                strategy=sel_strategy,
                custom_max_budget=custom_budget_val,
            )

        if recs_planner is not None and not recs_planner.empty:
            st.dataframe(
                recs_planner.rename(
                    columns={
                        "player_name": "Nama Pemain",
                        "team": "Klub Asal",
                        "league": "Liga",
                        "age": "Usia",
                        "height_cm": "Tinggi",
                        "foot": "Kaki",
                        "nationality": "Negara",
                        "quota_status": "Kuota",
                        "contract_expiry": "Kontrak s/d",
                        "market_value_raw": "Valuasi",
                        "player_archetype": "Arketipe ML",
                        "scout_score": "Scout Score",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.warning("Tidak ditemukan kandidat yang cocok. Coba longgarkan filter Kuota/Kaki/Budget.")

# ==========================================
# TAB 4: MULTI-ATTRIBUTE REPLACEMENT FINDER
# ==========================================
with tab4:
    st.markdown("#### 🔄 Multi-Attribute Replacement Finder (8D Similarity)")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        leagues_list = sorted(df["league"].unique())
        sel_league_t4 = st.selectbox("1. Liga Target", options=leagues_list, key="t4_league")
    with col_f2:
        teams_in_league = sorted(df[df["league"] == sel_league_t4]["team"].unique())
        sel_team_t4 = st.selectbox("2. Klub", options=teams_in_league, key="t4_team")
    with col_f3:
        pos_in_team = sorted(
            df[(df["league"] == sel_league_t4) & (df["team"] == sel_team_t4)]["position"].unique()
        )
        sel_pos_t4 = st.selectbox("3. Posisi", options=pos_in_team, key="t4_pos")
    with col_f4:
        players_final = sorted(
            df[
                (df["league"] == sel_league_t4)
                & (df["team"] == sel_team_t4)
                & (df["position"] == sel_pos_t4)
            ]["player_name"].unique()
        )
        sel_player_t4 = st.selectbox("4. Pemain Target", options=players_final, key="t4_player")

    cf1, cf2, cf3, cf4, cf5 = st.columns([1.1, 1.1, 1.1, 1.8, 1.1])
    with cf1:
        same_foot_t4 = st.checkbox("🦶 Kaki Sama", value=False, key="t4_same_foot")
    with cf2:
        quota_t4 = st.selectbox(
            "Filter Kuota",
            options=["Semua", "Lokal", "ASEAN", "Asia (AFC)", "Asing Non-Asia"],
            key="t4_quota",
            label_visibility="collapsed",
        )
    with cf3:
        max_budget_t4_m = st.number_input(
            "Maks Budget (M)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.5,
            key="t4_max_budget",
            label_visibility="collapsed",
            help="Maksimal Valuasi dalam Miliar Rp (0 = Tanpa Batas)",
        )
    with cf4:
        target_leagues_t4 = st.multiselect(
            "Filter Liga Pencarian",
            options=leagues_list,
            default=leagues_list,
            key="t4_target_leagues",
            label_visibility="collapsed",
        )
    with cf5:
        run_replace = st.button(
            "🔍 Cari Pengganti",
            type="primary",
            use_container_width=True,
            key="btn_replace",
        )

    target_info = df[df["player_name"] == sel_player_t4].iloc[0]
    st.caption(
        f"🎯 **Profil {target_info['player_name']}**: "
        f"Usia **{int(target_info['age'])} thn** | Tinggi **{int(target_info['height_cm'])} cm** | "
        f"Kaki **{target_info['foot']}** | Kuota **{target_info['quota_status']}** ({target_info['nationality']}) | "
        f"Valuasi **{target_info['market_value_raw']}** | **{target_info['player_archetype']}**"
    )

    if run_replace:
        budget_limit = max_budget_t4_m * 1e9 if max_budget_t4_m > 0 else None
        similar_players = brain.get_similar_players(
            player_name=sel_player_t4,
            top_n=12,
            same_foot_only=same_foot_t4,
            quota_filter=quota_t4,
            max_budget=budget_limit,
            target_leagues=target_leagues_t4 if target_leagues_t4 else None,
        )
        if similar_players is not None and not similar_players.empty:
            st.dataframe(
                similar_players.rename(
                    columns={
                        "player_name": "Nama Pemain",
                        "team": "Klub",
                        "league": "Liga",
                        "position": "Posisi",
                        "age": "Usia",
                        "height_cm": "Tinggi",
                        "foot": "Kaki",
                        "nationality": "Negara",
                        "quota_status": "Kuota",
                        "market_value_raw": "Valuasi",
                        "similarity_score": "Similarity (%)",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.warning("Tidak ditemukan pemain pengganti yang sesuai dengan filter di atas.")

# ==========================================
# TAB 5: HEAD-TO-HEAD RADAR & AI SCOUT REPORT
# ==========================================
with tab5:
    st.markdown("#### 🕸️ Head-to-Head 6D Radar & AI Executive Scouting Report")

    all_player_names = sorted(df["player_name"].unique())
    default_h2h = [
        p for p in ["Rizky Ridho", "Thom Haye", "Beckham Putra", "Ramadhan Sananta"]
        if p in all_player_names
    ][:2]
    if len(default_h2h) < 2 and len(all_player_names) >= 2:
        default_h2h = all_player_names[:2]

    selected_h2h = st.multiselect(
        "Pilih 2–4 Pemain untuk Dibandingkan:",
        options=all_player_names,
        default=default_h2h,
        max_selections=4,
        key="t5_players",
    )

    if selected_h2h:
        radar_data = brain.compare_players(selected_h2h)

        col_rad, col_tbl = st.columns([1.05, 1.35])
        with col_rad:
            categories = [
                "Market Valuation",
                "Age & Upside",
                "Physical Stature",
                "Squad Standing",
                "Market Momentum",
                "Moneyball Value",
            ]
            fig_radar = go.Figure()
            palette = ["#00CC96", "#EF553B", "#636EFA", "#AB63FA"]

            for idx, item in enumerate(radar_data):
                vals = [item["metrics"].get(c, 50.0) for c in categories]
                vals_closed = vals + [vals[0]]
                cats_closed = categories + [categories[0]]
                color_hex = palette[idx % len(palette)]

                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=vals_closed,
                        theta=cats_closed,
                        fill="toself",
                        name=f"{item['player_name']} ({item['team']})",
                        line=dict(color=color_hex, width=2.2),
                        opacity=0.75,
                    )
                )

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
                height=360,
                margin=dict(l=30, r=30, t=25, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_tbl:
            comp_rows = []
            for item in radar_data:
                row_dict = {
                    "Pemain": item["player_name"],
                    "Klub": item["team"],
                    "Posisi": item["position"],
                    "Usia": item["age"],
                    "Tinggi": item["height_cm"],
                    "Kaki": item["foot"],
                    "Kuota": item["quota_status"],
                    "Valuasi": item["market_value_raw"],
                    "Arketipe": item["archetype"],
                }
                for k_m, v_m in item["metrics"].items():
                    row_dict[k_m] = f"{v_m:.0f}%"
                comp_rows.append(row_dict)

            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

            # Tombol AI Scouting Report langsung di bawah tabel agar hemat ruang vertikal
            c_rep_sel, c_rep_btn = st.columns([1.6, 1.2])
            with c_rep_sel:
                report_target_player = st.selectbox(
                    "Target Laporan AI:",
                    options=selected_h2h,
                    key="t5_report_target",
                    label_visibility="collapsed",
                )
            with c_rep_btn:
                gen_report_clicked = st.button(
                    "📝 Buat AI Scout Report",
                    type="primary",
                    use_container_width=True,
                    key="btn_ai_report",
                )

        if gen_report_clicked:
            with st.spinner(f"Menyusun Executive Scouting Report untuk {report_target_player}..."):
                p_row = df[df["player_name"] == report_target_player].iloc[0]
                r_info = next(
                    (x for x in radar_data if x["player_name"] == report_target_player),
                    None,
                )
                sim_df = brain.get_similar_players(report_target_player, top_n=3)
                report_md, engine_used = generate_ai_scouting_report(
                    p_row, r_info, sim_df, api_key=gemini_api_key
                )
            st.caption(f"⚡ Dihasilkan oleh: **{engine_used}**")
            st.markdown(report_md)
    else:
        st.info("Pilih minimal 1 pemain di atas untuk menampilkan Radar Chart & AI Scouting Report.")

# ==========================================
# TAB 6: UNDERVALUED GEMS & CONTRACT RADAR
# ==========================================
with tab6:
    sub_gem, sub_bosman = st.tabs([
        "🚀 Undervalued & Liga 2/3 Promotion Gems",
        "⏳ Contract Expiry / Bosman Radar",
    ])

    with sub_gem:
        cg1, cg2, cg3, cg4, cg5 = st.columns([1.8, 1.2, 1.1, 0.8, 0.8])
        with cg1:
            default_gem_leagues = [
                lg for lg in ["Indonesia (Liga 2)", "Indonesia (Liga 3)", "Indonesia (Liga 1)"]
                if lg in all_leagues
            ]
            gem_leagues = st.multiselect(
                "Kompetisi",
                options=all_leagues,
                default=default_gem_leagues if default_gem_leagues else all_leagues,
                key="t6_gem_leagues",
            )
        with cg2:
            gem_pos = st.selectbox(
                "Posisi",
                options=["Semua"] + sorted(df["position"].unique()),
                key="t6_gem_pos",
            )
        with cg3:
            gem_max_age = st.slider(
                "Usia Maksimal",
                min_value=17,
                max_value=32,
                value=24,
                key="t6_gem_age",
            )
        with cg4:
            gem_top_n = st.number_input("Top N", min_value=5, max_value=50, value=20, key="t6_gem_n")
        with cg5:
            st.write("")
            st.write("")
            gem_u22 = st.checkbox("👶 U-22", value=False, key="t6_gem_u22")

        gems_df = brain.find_undervalued_gems(
            league_filter=gem_leagues if gem_leagues else None,
            position_filter=gem_pos,
            max_age=gem_max_age,
            u22_only=gem_u22,
            top_n=int(gem_top_n),
        )

        if not gems_df.empty:
            st.dataframe(
                gems_df.rename(
                    columns={
                        "player_name": "Nama Pemain",
                        "team": "Klub",
                        "league": "Kompetisi",
                        "position": "Posisi",
                        "age": "Usia",
                        "nationality": "Negara",
                        "quota_status": "Kuota",
                        "market_value_raw": "Harga Aktual",
                        "expected_mv_formatted": "Expected ML",
                        "undervalued_ratio": "Rasio (x)",
                        "gem_score": "Gem Score",
                    }
                ),
                hide_index=True,
                use_container_width=True,
                height=380,
            )
        else:
            st.warning("Tidak ditemukan kandidat Undervalued pada kombinasi filter ini.")

    with sub_bosman:
        cb1, cb2, cb3, cb4 = st.columns(4)
        with cb1:
            months_limit = st.slider(
                "Kontrak Habis ≤ (Bulan):",
                min_value=3,
                max_value=24,
                value=12,
                step=3,
                key="t6_bos_months",
            )
        with cb2:
            bos_league = st.selectbox(
                "Kompetisi",
                options=["Semua"] + all_leagues,
                key="t6_bos_league",
            )
        with cb3:
            bos_pos = st.selectbox(
                "Posisi",
                options=["Semua"] + sorted(df["position"].unique()),
                key="t6_bos_pos",
            )
        with cb4:
            bos_quota = st.selectbox(
                "Kuota Regulasi",
                options=["Semua", "Lokal", "ASEAN", "Asia (AFC)", "Asing Non-Asia"],
                key="t6_bos_quota",
            )

        exp_df = brain.get_expiring_contracts(
            months_ahead=months_limit,
            league_filter=bos_league,
            position_filter=bos_pos,
            quota_filter=bos_quota,
            top_n=30,
        )

        if not exp_df.empty:
            display_exp = exp_df.drop(columns=["market_value_est"], errors="ignore").rename(
                columns={
                    "player_name": "Nama Pemain",
                    "team": "Klub Saat Ini",
                    "league": "Kompetisi",
                    "position": "Posisi",
                    "age": "Usia",
                    "nationality": "Negara",
                    "quota_status": "Kuota",
                    "contract_expiry": "Kontrak Berakhir",
                    "contract_months_left": "Sisa (Bln)",
                    "market_value_raw": "Valuasi",
                }
            )
            st.dataframe(display_exp, hide_index=True, use_container_width=True, height=380)
        else:
            st.info("Tidak ditemukan pemain dengan kontrak yang segera habis pada filter ini.")

# ==========================================
# TAB 7: CONTENT CREATOR & DATABASE
# ==========================================
with tab7:
    sub_db, sub_content = st.tabs([
        f"📝 Master Database ({len(main_df):,} Pemain)",
        "📱 Content Creator Studio",
    ])

    with sub_db:
        c_db_info, c_db_dl = st.columns([3, 1])
        with c_db_info:
            st.caption("Menampilkan data pemain sesuai filter aktif di Sidebar.")
        with c_db_dl:
            csv_bytes = main_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download CSV",
                data=csv_bytes,
                file_name="garuda_scout_filtered_players.csv",
                mime="text/csv",
                use_container_width=True,
            )

        display_cols = [
            "player_name", "team", "league", "position", "age", "is_u22",
            "height_cm", "foot", "nationality", "quota_status", "contract_expiry",
            "market_value_raw", "mv_trend", "player_archetype",
        ]
        avail_display_cols = [c for c in display_cols if c in main_df.columns]
        st.dataframe(main_df[avail_display_cols], hide_index=True, use_container_width=True, height=440)

    with sub_content:
        col_fact1, col_fact2 = st.columns(2)
        with col_fact1:
            team_values = (
                df.groupby(["team", "league"])["market_value_est"]
                .sum()
                .reset_index()
                .sort_values("market_value_est", ascending=False)
                .head(8)
            )
            team_values["val_miliar"] = (team_values["market_value_est"] / 1e9).round(2)
            fig_sultan = px.bar(
                team_values,
                x="val_miliar",
                y="team",
                color="league",
                orientation="h",
                title="💰 Top 8 Skuad Termahal (Miliar Rp)",
                labels={"val_miliar": "Miliar Rp", "team": "Klub", "league": "Liga"},
                height=300,
            )
            fig_sultan.update_layout(
                yaxis={"categoryorder": "total ascending"},
                margin=dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(fig_sultan, use_container_width=True)

        with col_fact2:
            valid_age_df = df[df["age"] > 0]
            team_counts = valid_age_df.groupby("team")["player_name"].count()
            eligible_teams = team_counts[team_counts >= 15].index
            team_age = (
                valid_age_df[valid_age_df["team"].isin(eligible_teams)]
                .groupby(["team", "league"])["age"]
                .mean()
                .reset_index()
                .sort_values("age", ascending=True)
                .head(8)
            )
            team_age["age"] = team_age["age"].round(2)
            fig_age = px.bar(
                team_age,
                x="age",
                y="team",
                color="league",
                orientation="h",
                title="👶 Top 8 Skuad Termuda (Rata-rata Usia)",
                labels={"age": "Usia (Thn)", "team": "Klub", "league": "Liga"},
                height=300,
            )
            fig_age.update_layout(
                yaxis={"categoryorder": "total descending"},
                margin=dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(fig_age, use_container_width=True)

        st.markdown("---")
        c_gen1, c_gen2 = st.columns([1, 2])
        with c_gen1:
            target_league_content = st.selectbox(
                "Liga Target Konten",
                options=sorted(df["league"].unique()),
                key="t7_league",
            )
            criteria = st.radio(
                "Tipe Konten",
                ["💎 Wonderkid (≤ 22 Thn)", "🚀 Moneyball Gem", "🛡️ Senior Leader (≥ 30 Thn)"],
                key="t7_crit",
            )
            generate_btn = st.button(
                "🎲 Generate Player to Watch",
                type="primary",
                use_container_width=True,
                key="btn_content",
            )

        if generate_btn:
            with c_gen2:
                subset = df[(df["league"] == target_league_content) & (df["market_value_est"] > 0)]
                avg_val = subset["market_value_est"].mean() if not subset.empty else 0

                if "Wonderkid" in criteria:
                    candidates = subset[
                        (subset["age"] > 0)
                        & (subset["age"] <= 22)
                        & (subset["market_value_est"] >= avg_val * 0.8)
                    ]
                    tag = "#Wonderkid #GarudaMuda"
                elif "Moneyball" in criteria:
                    candidates = subset[subset["undervalued_ratio"] >= 1.25]
                    tag = "#MoneyballGem #Undervalued"
                else:
                    candidates = subset[(subset["age"] >= 30) & (subset["market_value_est"] >= avg_val)]
                    tag = "#VeteranLeader #Pengalaman"

                if not candidates.empty:
                    player = candidates.sample(1).iloc[0]
                    caption = (
                        f"🔥 PLAYER TO WATCH: {player['player_name']} 🔥\n"
                        f"🏟️ Klub: {player['team']} ({player['league']}) | 🎯 Posisi: {player['position']}\n"
                        f"🎂 Usia: {int(player['age'])} Thn | Tinggi: {int(player['height_cm'])} cm | Kaki: {player['foot']}\n"
                        f"🌏 Kuota: {player['quota_status']} ({player['nationality']})\n"
                        f"💰 Market Value: {player['market_value_raw']} (Expected ML: {format_idr(player['expected_mv_est'])})\n"
                        f"#GarudaScoutAI {tag} #{str(player['team']).replace(' ', '')}"
                    )
                    st.text_area("Caption Siap Copy:", value=caption, height=135)
                else:
                    st.warning("Belum ada kandidat yang memenuhi kriteria di liga ini.")
