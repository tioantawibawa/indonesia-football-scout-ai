import os
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.ml_engine import ScoutBrain

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Garuda Scout AI v6.0",
    page_icon="🇮🇩",
    layout="wide",
)

st.title("🇮🇩 Garuda Scout AI v6.0: Pro Scouting & Intelligence Suite")
st.caption(
    "Platform Analitik Scouting Sepak Bola Indonesia (Liga 1, Liga 2, Liga 3) & ASEAN "
    "berbasis Multi-Attribute ML, Moneyball Valuation & Regulasi PSSI/LIB"
)


# --- LOAD DATA & ML ENGINE (CACHED) ---
@st.cache_resource(show_spinner="Memuat database 3.000+ pemain & melatih model ML...")
def load_scout_brain():
    return ScoutBrain()


try:
    brain = load_scout_brain()
    df = brain.df
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()


def format_idr(val):
    """Format nilai Rupiah secara ringkas dan mudah dibaca."""
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
    Menghasilkan Executive Scouting Report.
    - Jika GEMINI_API_KEY tersedia, memanggil model gemini-3.8-flash.
    - Jika tidak tersedia, menghasilkan laporan analitik terstruktur berbasis metrik ML ScoutBrain.
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

    # Fallback: Data-Driven ML Executive Scouting Report (tanpa API eksternal)
    val_verdict = (
        f"**Undervalued ({p_underval:.2f}x)** — Model ML memprediksi nilai wajar pemain ini mencapai "
        f"**{format_idr(p_exp_mv)}**, lebih tinggi dibanding harga pasarnya saat ini (**{p_mv_raw}**)."
        if p_underval >= 1.15
        else (
            f"**Fair Value ({p_underval:.2f}x)** — Valuasi pasar saat ini (**{p_mv_raw}**) sejalan dengan "
            f"ekspektasi model (**{format_idr(p_exp_mv)}**)."
            if p_underval >= 0.85
            else f"**Premium / Marquee Valuation ({p_underval:.2f}x)** — Valuasi saat ini (**{p_mv_raw}**) "
            f"berada di atas rata-rata profil usianya (**{format_idr(p_exp_mv)}**), mencerminkan status pemain kunci."
        )
    )

    contract_verdict = (
        f"⚠️ **Peluang Bosman / Kontrak Segera Habis ({p_contract} — ~{p_months} bulan lagi)**. "
        "Klub peminat memiliki daya tawar tinggi untuk negosiasi biaya transfer rendah atau pra-kontrak."
        if 0 <= p_months <= 12
        else (
            f"Kontrak tercatat hingga **{p_contract}** (~{p_months} bulan tersisa). Membutuhkan skema tebus kontrak atau peminjaman."
            if p_months > 12
            else "Tanggal akhir kontrak belum dipublikasikan secara terbuka; disarankan verifikasi langsung ke agensi pemain."
        )
    )

    u22_badge = (
        "✅ **Memenuhi Regulasi U-22** (Sangat strategis untuk slot starter wajib U-22 Liga 1/Liga 2)."
        if p_age <= 22
        else f"Kategori usia **{p_age} tahun** (Usia matang/emas untuk kontribusi instan di tim utama)."
    )

    strengths = [k for k, v in metrics.items() if v >= 70]
    strength_str = ", ".join(strengths) if strengths else "Keseimbangan atribut lintas dimensi"

    report_md = f"""### 📋 Executive Scouting Dossier: **{p_name}**
**Klub:** {p_team} ({p_league}) | **Posisi:** {p_pos} | **Arketipe ML:** {p_archetype}

---

#### 1. Profil Fisik & Peran Skuad
* **Postur & Kaki Dominan:** {p_height} cm dengan kaki dominan **{p_foot}** (Persentil Postur Posisi: **{metrics.get('Physical Stature', 50):.1f}%**).
* **Peran di Klub Saat Ini:** Skor *Squad Standing* berada di persentil **{metrics.get('Squad Standing', 50):.1f}%**, menunjukkan posisi pentingnya dalam struktur nilai skuad **{p_team}**.
* **Keunggulan Utama (6D Radar):** {strength_str}.

#### 2. Analisis Valuasi Moneyball & Momentum Pasar
* **Status Valuasi:** {val_verdict}
* **Momentum Pasar:** Tren **{p_trend}** ({p_growth:+.1f}% dibanding periode sebelumnya), dengan skor momentum **{metrics.get('Market Momentum', 50):.1f}/100**.
* **Situasi Kontrak:** {contract_verdict}

#### 3. Kepatuhan Regulasi Liga Indonesia (PSSI / LIB)
* **Status Kuota:** **{p_quota}** (Kewarganegaraan: {p_nat}).
* **Status Usia:** {u22_badge}

#### 4. Pemain Pembanding (Apple-to-Apple Replacements)
{chr(10).join([f'- {s}' for s in sim_names]) if sim_names else '- Tidak ada pembanding terdekat.'}
"""
    return report_md, "ScoutBrain v6.0 ML Engine"


# ==========================================
# 🛠️ SIDEBAR GLOBAL FILTER
# ==========================================
st.sidebar.header("🛠️ Filter Global Scouting")
st.sidebar.caption("Filter ini diterapkan pada Tab Market Explorer & Database")

# 1. Filter Negara/Liga
all_leagues = sorted(df["league"].unique())
sel_leagues = st.sidebar.multiselect(
    "1. Pilih Kompetisi / Liga",
    options=all_leagues,
    default=all_leagues,
)

df_l1 = df[df["league"].isin(sel_leagues)] if sel_leagues else df.iloc[0:0]

# 2. Filter Klub (Dinamis berdasarkan Liga yg dipilih)
available_teams = sorted(df_l1["team"].unique())
sel_teams = st.sidebar.multiselect("2. Pilih Klub (Opsional)", options=available_teams)

df_l2 = df_l1[df_l1["team"].isin(sel_teams)] if sel_teams else df_l1

# 3. Filter Posisi
available_positions = sorted(df_l2["position"].unique())
sel_positions = st.sidebar.multiselect("3. Pilih Posisi (Opsional)", options=available_positions)

# 4. Filter Regulasi Kuota & U-22
quota_options = ["Semua", "Lokal", "ASEAN", "Asia (AFC)", "Asing Non-Asia"]
sel_quota = st.sidebar.selectbox("4. Filter Kuota Regulasi", options=quota_options, index=0)

only_u22_sidebar = st.sidebar.checkbox("👶 Hanya Pemain U-22 (≤ 22 Tahun)", value=False)

# 5. Filter Rentang Usia
min_age_db = int(df["age_clean"].min()) if not df.empty else 15
max_age_db = int(df["age_clean"].max()) if not df.empty else 45
age_range = st.sidebar.slider(
    "5. Rentang Usia",
    min_value=min_age_db,
    max_value=max_age_db,
    value=(min_age_db, 22 if only_u22_sidebar else max_age_db),
)

# Terapkan semua filter ke main_df
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

st.sidebar.markdown("---")
col_sb1, col_sb2 = st.sidebar.columns(2)
col_sb1.metric("Data Terpantau", f"{len(main_df):,}")
col_sb2.metric("Total Klub", f"{main_df['team'].nunique()}")

with st.sidebar.expander("🔑 Opsional: Gemini AI Key"):
    env_key = os.environ.get("GEMINI_API_KEY", "")
    gemini_api_key = st.text_input(
        "GEMINI_API_KEY (gemini-3.8-flash)",
        value=env_key,
        type="password",
        help="Kosongkan untuk menggunakan mesin laporan analitik bawaan ScoutBrain ML.",
    )

# ==========================================
# 📑 TABS MENU (6 TABS)
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Market Explorer & Trends",
    "🤖 Smart Squad Planner",
    "🔄 Replacement Finder",
    "🕸️ Head-to-Head & AI Report",
    "💎 Undervalued & Contract Radar",
    "📱 Content & 📝 Database",
])

# ==========================================
# TAB 1: MARKET EXPLORER & HISTORICAL TRENDS
# ==========================================
with tab1:
    st.subheader("📊 Analisis Pasar & Tren Historis Valuasi")

    if not main_df.empty:
        # KPI Metrics Row
        k1, k2, k3, k4, k5 = st.columns(5)
        valued_main = main_df[main_df["market_value_est"] > 0]
        avg_val = valued_main["market_value_est"].mean() if not valued_main.empty else 0
        max_val = main_df["market_value_est"].max()
        u22_count = int(main_df["is_u22"].sum())
        exp_12m = int(
            ((main_df["contract_months_left"] >= 0) & (main_df["contract_months_left"] <= 12)).sum()
        )

        k1.metric("Pemain Terfilter", f"{len(main_df):,} Pemain")
        k2.metric("Rata-rata Valuasi", format_idr(avg_val))
        k3.metric("Valuasi Tertinggi", format_idr(max_val))
        k4.metric("Talenta U-22", f"{u22_count:,} ({u22_count / len(main_df) * 100:.1f}%)")
        k5.metric("Kontrak Habis ≤12 Bln", f"{exp_12m:,} Pemain")

        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            x_axis_mode = st.radio(
                "Sumbu X Grafik Sebaran:",
                ["Posisi", "Usia (Age Curve)"],
                horizontal=True,
                key="t1_xaxis",
            )
        with col_ctrl2:
            color_mode = st.radio(
                "Warna Berdasarkan:",
                ["Kompetisi / Liga", "Arketipe Pemain (ML)", "Status Kuota Regulasi"],
                horizontal=True,
                key="t1_color",
            )

        x_col = "position" if x_axis_mode == "Posisi" else "age_clean"
        color_col = (
            "league"
            if color_mode == "Kompetisi / Liga"
            else ("player_archetype" if color_mode == "Arketipe Pemain (ML)" else "quota_status")
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
            title=f"Sebaran Valuasi Pemain ({len(main_df):,} Pemain)",
            labels={
                "market_value_est": "Valuasi Pasar (Rp)",
                "position": "Posisi",
                "age_clean": "Usia (Tahun)",
                "league": "Liga",
                "player_archetype": "Arketipe ML",
                "quota_status": "Kuota",
            },
            height=520,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")
        # --- SUB-BAGIAN: TREN HISTORIS & MARKET MOVERS ---
        col_hist, col_movers = st.columns([1.2, 1.3])

        with col_hist:
            st.markdown("#### 📈 Grafik Tren Historis Nilai Pasar (Time-Series)")
            st.caption("Melacak riwayat perkembangan harga pemain dari 25 snapshot historis.")
            player_list_t1 = sorted(main_df["player_name"].unique())
            default_idx = 0
            for candidate_default in ["Rizky Ridho", "Thom Haye", "Beckham Putra", "Ramadhan Sananta"]:
                if candidate_default in player_list_t1:
                    default_idx = player_list_t1.index(candidate_default)
                    break

            sel_hist_player = st.selectbox(
                "Pilih Pemain untuk Dilihat Riwayat Valuasinya:",
                options=player_list_t1,
                index=default_idx,
                key="t1_hist_player",
            )
            hist_df = brain.get_player_value_history(sel_hist_player)
            if not hist_df.empty:
                fig_hist = px.line(
                    hist_df,
                    x="date",
                    y="market_value_est",
                    markers=True,
                    hover_data=["team", "market_value_raw"],
                    title=f"Riwayat Nilai Pasar: {sel_hist_player}",
                    labels={"date": "Tanggal Snapshot", "market_value_est": "Valuasi (Rp)"},
                    height=340,
                )
                fig_hist.update_traces(line=dict(width=3, color="#00CC96"), marker=dict(size=8))
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("Data snapshot historis belum tersedia untuk pemain ini.")

        with col_movers:
            st.markdown("#### 🚀 Top Market Movers (Gainers vs Drops)")
            mover_dir = st.radio(
                "Kategori Pergerakan:",
                ["📈 Top Gainers (Kenaikan Tertinggi)", "📉 Top Drops (Penurunan Terdalam)"],
                horizontal=True,
                key="t1_mover_dir",
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
                            "change_formatted": "Perubahan",
                            "mv_growth_pct": "Pertumbuhan (%)",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                    height=340,
                )
            else:
                st.info("Tidak ada data pergerakan nilai pasar pada filter ini.")
    else:
        st.warning("Data kosong. Silakan sesuaikan filter di Sidebar.")

# ==========================================
# TAB 2: SMART SQUAD PLANNER & REGULASI
# ==========================================
with tab2:
    st.header("🤖 Smart Squad Planner & Filter Regulasi Liga")
    st.write(
        "Rekomendasi rekrutan pemain baru berdasarkan **kapasitas finansial klub**, "
        "**kuota regulasi PSSI/LIB (Lokal, ASEAN, Asia, Asing, U-22)**, dan **strategi transfer**."
    )

    c_p1, c_p2, c_p3 = st.columns(3)
    with c_p1:
        target_team_planner = st.selectbox(
            "1. Pilih Klub Target",
            options=sorted(df["team"].unique()),
            key="t2_team",
        )
        target_pos_planner = st.selectbox(
            "2. Posisi Dibutuhkan",
            options=sorted(df["position"].unique()),
            key="t2_pos",
        )
    with c_p2:
        strategy_map = {
            "⚖️ Balanced (Keseimbangan Kualitas & Usia)": "balanced",
            "💎 Wonderkid U-23 (Investasi Jangka Panjang)": "wonderkid",
            "🚀 Moneyball (Undervalued & Efisiensi Tinggi)": "moneyball",
            "🛡️ Proven Star (Usia Emas 24-30 Tahun)": "proven",
        }
        sel_strategy_label = st.selectbox(
            "3. Strategi Transfer Klub",
            options=list(strategy_map.keys()),
            key="t2_strategy",
        )
        sel_strategy = strategy_map[sel_strategy_label]

        sel_quota_planner = st.selectbox(
            "4. Filter Kuota Regulasi",
            options=["Semua", "Lokal", "ASEAN", "Asia (AFC)", "Asing Non-Asia"],
            key="t2_quota",
        )
    with c_p3:
        sel_foot_planner = st.selectbox(
            "5. Preferensi Kaki Dominan",
            options=["Semua", "kanan", "kiri", "keduanya"],
            key="t2_foot",
        )
        u22_only_planner = st.checkbox(
            "👶 Wajib Memenuhi Regulasi U-22 (≤ 22 Tahun)",
            value=False,
            key="t2_u22",
        )
        custom_budget_m = st.number_input(
            "6. Batas Budget Maksimal (Miliar Rp, 0 = Otomatis)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.5,
            key="t2_budget",
        )

    # Ringkasan profil skuad klub target
    club_df = df[df["team"] == target_team_planner]
    if not club_df.empty:
        c_avg = club_df[club_df["market_value_est"] > 0]["market_value_est"].mean()
        c_foreign = int((club_df["quota_status"] != "Lokal").sum())
        c_u22 = int(club_df["is_u22"].sum())
        st.info(
            f"📌 **Profil Skuad {target_team_planner}** ({club_df['league'].iloc[0]}): "
            f"**{len(club_df)} Pemain** | Rata-rata Valuasi: **{format_idr(c_avg)}** | "
            f"Pemain Asing/ASEAN/Asia: **{c_foreign}** | Pemain U-22: **{c_u22}**"
        )

    if st.button("🔍 Cari Kandidat Rekrutan Terbaik", type="primary", key="btn_planner"):
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
            st.success(
                f"Ditemukan {len(recs_planner)} kandidat **{target_pos_planner}** terbaik untuk **{target_team_planner}**:"
            )
            st.dataframe(
                recs_planner.rename(
                    columns={
                        "player_name": "Nama Pemain",
                        "team": "Klub Asal",
                        "league": "Liga",
                        "age": "Usia",
                        "height_cm": "Tinggi (cm)",
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
            st.warning(
                "Tidak ditemukan kandidat yang memenuhi seluruh filter ketat tersebut. "
                "Coba longgarkan filter Kuota, Kaki Dominan, atau Batas Budget."
            )

# ==========================================
# TAB 3: MULTI-ATTRIBUTE REPLACEMENT FINDER
# ==========================================
with tab3:
    st.header("🔄 Multi-Attribute Replacement Finder (8D Similarity)")
    st.write(
        "Mencari pemain pengganti yang **identik secara profil multi-atribut** "
        "(Usia, Log-Valuasi, Tinggi Badan, Kaki Dominan, Kasta Liga, dan Peran Skuad) "
        "menggunakan algoritma *Hybrid Weighted Cosine + Gaussian RBF Similarity*."
    )

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        leagues_list = sorted(df["league"].unique())
        sel_league_t3 = st.selectbox("1. Pilih Liga Pemain Target", options=leagues_list, key="t3_league")
    with col_f2:
        teams_in_league = sorted(df[df["league"] == sel_league_t3]["team"].unique())
        sel_team_t3 = st.selectbox("2. Pilih Klub", options=teams_in_league, key="t3_team")
    with col_f3:
        pos_in_team = sorted(
            df[(df["league"] == sel_league_t3) & (df["team"] == sel_team_t3)]["position"].unique()
        )
        sel_pos_t3 = st.selectbox("3. Pilih Posisi", options=pos_in_team, key="t3_pos")
    with col_f4:
        players_final = sorted(
            df[
                (df["league"] == sel_league_t3)
                & (df["team"] == sel_team_t3)
                & (df["position"] == sel_pos_t3)
            ]["player_name"].unique()
        )
        sel_player_t3 = st.selectbox("4. Pilih Pemain Target", options=players_final, key="t3_player")

    # Tampilkan kartu ringkas pemain target
    target_info = df[df["player_name"] == sel_player_t3].iloc[0]
    st.markdown(
        f"**🎯 Profil Target — {target_info['player_name']}**: "
        f"Usia **{int(target_info['age'])} thn** | Tinggi **{int(target_info['height_cm'])} cm** | "
        f"Kaki **{target_info['foot']}** | Kuota **{target_info['quota_status']}** ({target_info['nationality']}) | "
        f"Valuasi **{target_info['market_value_raw']}** | Arketipe: **{target_info['player_archetype']}**"
    )

    # Filter Opsional Pencarian Pengganti
    cf1, cf2, cf3, cf4 = st.columns(4)
    with cf1:
        same_foot_t3 = st.checkbox("🦶 Wajib Kaki Dominan Sama", value=False, key="t3_same_foot")
    with cf2:
        quota_t3 = st.selectbox(
            "Filter Kuota Pengganti",
            options=["Semua", "Lokal", "ASEAN", "Asia (AFC)", "Asing Non-Asia"],
            key="t3_quota",
        )
    with cf3:
        max_budget_t3_m = st.number_input(
            "Maks. Valuasi (Miliar Rp, 0 = Bebas)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.5,
            key="t3_max_budget",
        )
    with cf4:
        target_leagues_t3 = st.multiselect(
            "Filter Liga Pencarian",
            options=leagues_list,
            default=leagues_list,
            key="t3_target_leagues",
        )

    if st.button(f"🔍 Cari Pengganti Identik untuk {sel_player_t3}", type="primary", key="btn_replace"):
        budget_limit = max_budget_t3_m * 1e9 if max_budget_t3_m > 0 else None
        similar_players = brain.get_similar_players(
            player_name=sel_player_t3,
            top_n=12,
            same_foot_only=same_foot_t3,
            quota_filter=quota_t3,
            max_budget=budget_limit,
            target_leagues=target_leagues_t3 if target_leagues_t3 else None,
        )
        if similar_players is not None and not similar_players.empty:
            st.success(f"Top {len(similar_players)} Rekomendasi Pengganti untuk **{sel_player_t3}**:")
            st.dataframe(
                similar_players.rename(
                    columns={
                        "player_name": "Nama Pemain",
                        "team": "Klub",
                        "league": "Liga",
                        "position": "Posisi",
                        "age": "Usia",
                        "height_cm": "Tinggi (cm)",
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
            st.warning("Tidak ditemukan pemain pengganti yang sesuai dengan batasan filter di atas.")

# ==========================================
# TAB 4: HEAD-TO-HEAD RADAR & AI SCOUT REPORT
# ==========================================
with tab4:
    st.header("🕸️ Head-to-Head Radar Comparison & AI Scouting Report")
    st.write(
        "Bandingkan hingga 4 pemain secara *apple-to-apple* menggunakan **6D Percentile Radar Chart** "
        "dan hasilkan **Executive Scouting Report** otomatis."
    )

    all_player_names = sorted(df["player_name"].unique())
    default_h2h = [
        p for p in ["Rizky Ridho", "Jay Idzes", "Jordi Amat", "Mezut Ozil", "Thom Haye", "Beckham Putra"]
        if p in all_player_names
    ][:2]
    if len(default_h2h) < 2 and len(all_player_names) >= 2:
        default_h2h = all_player_names[:2]

    selected_h2h = st.multiselect(
        "Pilih 2 hingga 4 Pemain untuk Dibandingkan:",
        options=all_player_names,
        default=default_h2h,
        max_selections=4,
        key="t4_players",
    )

    if selected_h2h:
        radar_data = brain.compare_players(selected_h2h)

        col_rad, col_tbl = st.columns([1.15, 1.25])
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
                        line=dict(color=color_hex, width=2.5),
                        opacity=0.75,
                    )
                )

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%"),
                ),
                title="6D Scouting Percentile Radar (0 - 100%)",
                height=460,
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_tbl:
            st.markdown("#### 📋 Tabel Perbandingan Atribut & Persentil")
            comp_rows = []
            for item in radar_data:
                row_dict = {
                    "Pemain": item["player_name"],
                    "Klub": item["team"],
                    "Liga": item["league"],
                    "Posisi": item["position"],
                    "Usia": item["age"],
                    "Tinggi (cm)": item["height_cm"],
                    "Kaki": item["foot"],
                    "Kuota": item["quota_status"],
                    "Valuasi": item["market_value_raw"],
                    "Arketipe ML": item["archetype"],
                }
                for k_m, v_m in item["metrics"].items():
                    row_dict[f"Radar: {k_m}"] = f"{v_m:.1f}%"
                comp_rows.append(row_dict)

            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("🧠 AI Executive Scouting Report Generator")
        col_rep_sel, col_rep_btn = st.columns([2, 1])
        with col_rep_sel:
            report_target_player = st.selectbox(
                "Pilih Pemain untuk Dianalisis dalam Laporan Lengkap:",
                options=selected_h2h,
                key="t4_report_target",
            )
        with col_rep_btn:
            st.write("")
            st.write("")
            gen_report_clicked = st.button(
                "📝 Generate AI Scouting Report",
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
        st.info("Silakan pilih minimal 1 pemain di atas untuk menampilkan Radar Chart & AI Scouting Report.")

# ==========================================
# TAB 5: UNDERVALUED GEMS & CONTRACT RADAR
# ==========================================
with tab5:
    st.header("💎 Undervalued Gems (Liga 2/3 Promotion Radar) & Contract Expiry")

    sub_gem, sub_bosman = st.tabs([
        "🚀 Undervalued & Liga 2/3 Promotion Gems",
        "⏳ Contract Expiry / Bosman Radar",
    ])

    with sub_gem:
        st.subheader("🚀 Deteksi Pemain Undervalued & Talenta Promosi (RandomForest Valuation)")
        st.write(
            "Menemukan pemain muda yang memiliki **Expected Market Value** di atas harga pasarnya saat ini "
            "— sangat ideal untuk mencari bintang **Liga 2 & Liga 3** yang siap promosi ke **Liga 1**."
        )

        cg1, cg2, cg3, cg4 = st.columns(4)
        with cg1:
            default_gem_leagues = [
                lg for lg in ["Indonesia (Liga 2)", "Indonesia (Liga 3)", "Indonesia (Liga 1)"]
                if lg in all_leagues
            ]
            gem_leagues = st.multiselect(
                "1. Filter Kompetisi",
                options=all_leagues,
                default=default_gem_leagues if default_gem_leagues else all_leagues,
                key="t5_gem_leagues",
            )
        with cg2:
            gem_pos = st.selectbox(
                "2. Filter Posisi",
                options=["Semua"] + sorted(df["position"].unique()),
                key="t5_gem_pos",
            )
        with cg3:
            gem_max_age = st.slider(
                "3. Usia Maksimal",
                min_value=17,
                max_value=32,
                value=24,
                key="t5_gem_age",
            )
        with cg4:
            gem_u22 = st.checkbox("👶 Hanya U-22 (≤ 22 Tahun)", value=False, key="t5_gem_u22")
            gem_top_n = st.number_input("Jumlah Kandidat", min_value=5, max_value=50, value=20, key="t5_gem_n")

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
                        "market_value_raw": "Harga Saat Ini",
                        "expected_mv_formatted": "Expected Value (ML)",
                        "undervalued_ratio": "Rasio Undervalued (x)",
                        "gem_score": "Gem Score",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.warning("Tidak ditemukan kandidat Undervalued pada kombinasi filter ini.")

    with sub_bosman:
        st.subheader("⏳ Contract Expiry & Bosman Free-Agent Radar")
        st.write(
            "Daftar pemain bernilai tinggi yang kontraknya akan habis dalam waktu dekat — "
            "target strategis untuk negosiasi *free transfer* (Bosman) atau biaya tebus diskon."
        )

        cb1, cb2, cb3, cb4 = st.columns(4)
        with cb1:
            months_limit = st.slider(
                "1. Kontrak Habis Dalam ≤ (Bulan):",
                min_value=3,
                max_value=24,
                value=12,
                step=3,
                key="t5_bos_months",
            )
        with cb2:
            bos_league = st.selectbox(
                "2. Filter Kompetisi",
                options=["Semua"] + all_leagues,
                key="t5_bos_league",
            )
        with cb3:
            bos_pos = st.selectbox(
                "3. Filter Posisi",
                options=["Semua"] + sorted(df["position"].unique()),
                key="t5_bos_pos",
            )
        with cb4:
            bos_quota = st.selectbox(
                "4. Filter Kuota Regulasi",
                options=["Semua", "Lokal", "ASEAN", "Asia (AFC)", "Asing Non-Asia"],
                key="t5_bos_quota",
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
                    "contract_expiry": "Tanggal Kontrak Habis",
                    "contract_months_left": "Estimasi Sisa (Bulan)",
                    "market_value_raw": "Valuasi Pasar",
                }
            )
            st.dataframe(display_exp, hide_index=True, use_container_width=True)
        else:
            st.info("Tidak ditemukan pemain dengan kontrak yang segera habis pada filter ini.")

# ==========================================
# TAB 6: CONTENT CREATOR & DATABASE
# ==========================================
with tab6:
    st.header("📱 Content Creator Studio & 📝 Master Database")

    col_fact1, col_fact2 = st.columns(2)
    with col_fact1:
        st.markdown("#### 💰 Top 10 The Sultan Clubs (Skuad Termahal)")
        team_values = (
            df.groupby(["team", "league"])["market_value_est"]
            .sum()
            .reset_index()
            .sort_values("market_value_est", ascending=False)
            .head(10)
        )
        team_values["val_miliar"] = (team_values["market_value_est"] / 1e9).round(2)
        fig_sultan = px.bar(
            team_values,
            x="val_miliar",
            y="team",
            color="league",
            orientation="h",
            title="Top 10 Skuad Termahal (Miliar Rp)",
            labels={"val_miliar": "Total Valuasi (Miliar Rp)", "team": "Klub", "league": "Liga"},
            height=400,
        )
        fig_sultan.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_sultan, use_container_width=True)

    with col_fact2:
        st.markdown("#### 👶 Top 10 The Young Guns (Skuad Termuda)")
        valid_age_df = df[df["age"] > 0]
        team_counts = valid_age_df.groupby("team")["player_name"].count()
        eligible_teams = team_counts[team_counts >= 15].index
        team_age = (
            valid_age_df[valid_age_df["team"].isin(eligible_teams)]
            .groupby(["team", "league"])["age"]
            .mean()
            .reset_index()
            .sort_values("age", ascending=True)
            .head(10)
        )
        team_age["age"] = team_age["age"].round(2)
        fig_age = px.bar(
            team_age,
            x="age",
            y="team",
            color="league",
            orientation="h",
            title="Top 10 Skuad Termuda (Rata-rata Usia Valid)",
            labels={"age": "Rata-rata Usia (Tahun)", "team": "Klub", "league": "Liga"},
            height=400,
        )
        fig_age.update_layout(yaxis={"categoryorder": "total descending"})
        st.plotly_chart(fig_age, use_container_width=True)

    st.markdown("---")
    st.subheader("📸 Player to Watch Social Media Generator")
    c_gen1, c_gen2 = st.columns([1, 2])
    with c_gen1:
        target_league_content = st.selectbox(
            "Pilih Liga Target",
            options=sorted(df["league"].unique()),
            key="t6_league",
        )
        criteria = st.radio(
            "Tipe Konten",
            ["💎 Wonderkid (≤ 22 Tahun)", "🚀 Moneyball Undervalued", "🛡️ Senior Leader (≥ 30 Tahun)"],
            key="t6_crit",
        )
        generate_btn = st.button("🎲 Generate Konten Visual!", type="primary", key="btn_content")

    if generate_btn:
        with c_gen2:
            subset = df[(df["league"] == target_league_content) & (df["market_value_est"] > 0)]
            avg_val = subset["market_value_est"].mean() if not subset.empty else 0

            if "Wonderkid" in criteria:
                candidates = subset[(subset["age"] > 0) & (subset["age"] <= 22) & (subset["market_value_est"] >= avg_val * 0.8)]
                tag = "#Wonderkid #GarudaMuda"
            elif "Moneyball" in criteria:
                candidates = subset[subset["undervalued_ratio"] >= 1.25]
                tag = "#MoneyballGem #Undervalued"
            else:
                candidates = subset[(subset["age"] >= 30) & (subset["market_value_est"] >= avg_val)]
                tag = "#VeteranLeader #Pengalaman"

            if not candidates.empty:
                player = candidates.sample(1).iloc[0]
                p_radar = brain.compare_players([player["player_name"]])
                if p_radar:
                    m_dict = p_radar[0]["metrics"]
                    fig_ig = go.Figure(
                        go.Bar(
                            x=list(m_dict.values()),
                            y=list(m_dict.keys()),
                            orientation="h",
                            marker_color="#00CC96",
                        )
                    )
                    fig_ig.update_layout(
                        title=f"🔥 PLAYER TO WATCH: {player['player_name']} ({player['team']})",
                        xaxis=dict(range=[0, 100], title="Skor Persentil (0-100%)"),
                        template="plotly_dark",
                        height=350,
                    )
                    st.plotly_chart(fig_ig, use_container_width=True)

                caption = (
                    f"🔥 PLAYER TO WATCH: {player['player_name']} 🔥\n\n"
                    f"🏟️ Klub: {player['team']} ({player['league']})\n"
                    f"🎯 Posisi: {player['position']} ({player['player_archetype']})\n"
                    f"🎂 Usia: {int(player['age'])} Tahun | Tinggi: {int(player['height_cm'])} cm | Kaki: {player['foot']}\n"
                    f"🌏 Kuota: {player['quota_status']} ({player['nationality']})\n"
                    f"💰 Market Value: {player['market_value_raw']} (Expected ML: {format_idr(player['expected_mv_est'])})\n\n"
                    f"#GarudaScoutAI {tag} #{str(player['team']).replace(' ', '')}"
                )
                st.text_area("Caption Siap Copy (Instagram / X / TikTok):", value=caption, height=180)
            else:
                st.warning("Belum ada kandidat yang memenuhi kriteria di liga ini.")

    st.markdown("---")
    st.subheader(f"📝 Master Database Explorer ({len(main_df):,} Pemain Terfilter)")
    st.caption("Tabel di bawah ini merespon seluruh filter pada Sidebar di sebelah kiri.")

    display_cols = [
        "player_name", "team", "league", "position", "age", "is_u22",
        "height_cm", "foot", "nationality", "quota_status", "contract_expiry",
        "market_value_raw", "mv_trend", "player_archetype",
    ]
    avail_display_cols = [c for c in display_cols if c in main_df.columns]
    st.dataframe(main_df[avail_display_cols], hide_index=True, use_container_width=True)

    csv_bytes = main_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Data Terfilter (CSV)",
        data=csv_bytes,
        file_name="garuda_scout_filtered_players.csv",
        mime="text/csv",
    )
