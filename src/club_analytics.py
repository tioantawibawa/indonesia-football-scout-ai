import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests


class ClubAnalytics:
    def __init__(
        self,
        match_stats_path="data/processed/club_match_stats.csv",
        player_db_path="data/processed/master_player_db.csv",
    ):
        self.match_stats_path = match_stats_path
        self.player_db_path = player_db_path

        # 1. Load Match Standings & Performance Stats
        if os.path.exists(match_stats_path):
            self.df_clubs = pd.read_csv(match_stats_path)
        else:
            self.df_clubs = pd.DataFrame()

        # 2. Load Master Player Database
        if os.path.exists(player_db_path):
            self.df_players = pd.read_csv(player_db_path)
            if "league_country" in self.df_players.columns and "league" not in self.df_players.columns:
                self.df_players.rename(columns={"league_country": "league"}, inplace=True)
        else:
            self.df_players = pd.DataFrame()

        self._calculate_league_benchmarks()

    def _calculate_league_benchmarks(self):
        """Menghitung metrik rata-rata liga sebagai tolak ukur (benchmarks) evaluasi performa."""
        self.league_benchmarks = {}
        if self.df_clubs.empty:
            return

        for league_name, grp in self.df_clubs.groupby("league"):
            played_sum = grp["played"].sum()
            gf_sum = grp["gf"].sum()
            ga_sum = grp["ga"].sum()
            pts_sum = grp["points"].sum()

            self.league_benchmarks[league_name] = {
                "avg_gf_per_match": round(gf_sum / played_sum, 2) if played_sum > 0 else 1.25,
                "avg_ga_per_match": round(ga_sum / played_sum, 2) if played_sum > 0 else 1.25,
                "avg_ppg": round(pts_sum / played_sum, 2) if played_sum > 0 else 1.35,
                "avg_win_rate": round(grp["win_rate"].mean(), 1) if not grp.empty else 33.3,
            }

    # =========================================================================
    # 1. KLASEMEN & METRIK PERTANDINGAN
    # =========================================================================
    def get_standings(self, league=None, group=None):
        """Mengambil tabel klasemen resmi berdasarkan liga dan grup kompetisi."""
        if self.df_clubs.empty:
            return pd.DataFrame()

        df_sub = self.df_clubs.copy()
        if league and league != "Semua":
            df_sub = df_sub[df_sub["league"] == league]
        if group and group != "Semua":
            df_sub = df_sub[df_sub["group"] == group]

        # Urutkan berdasarkan Poin, Selisih Gol, Gol Memasukkan
        df_sub = df_sub.sort_values(
            by=["points", "gd", "gf"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        return df_sub

    def get_club_match_stats(self, team_name):
        """Mengambil data pertandingan spesifik satu klub."""
        if self.df_clubs.empty:
            return None
        match = self.df_clubs[self.df_clubs["team"] == team_name]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    # =========================================================================
    # 2. PROFIL SKUAD & STRUKTUR POSISI INTERNAL
    # =========================================================================
    def get_club_squad_profile(self, team_name):
        """
        Menganalisis profil internal skuad klub dari database 3.060 pemain:
        - Distribusi umur & status U-22
        - Komposisi kuota pemain asing (Lokal, ASEAN, Asia, Asing Non-Asia)
        - Kedalaman per posisi (Kiper, Bek Tengah, Bek Sayap, Gelandang, Penyerang)
        - Rerata postur tinggi badan lini belakang & kiper
        """
        if self.df_players.empty:
            return {}

        players = self.df_players[self.df_players["team"] == team_name].copy()
        if players.empty:
            return {}

        squad_size = len(players)
        valid_ages = players[players["age"] > 0]["age"]
        avg_age = round(float(valid_ages.mean()), 1) if not valid_ages.empty else 25.5

        # Distribusi Umur
        u22_players = players[(players["age"] > 0) & (players["age"] <= 22)]
        prime_players = players[(players["age"] >= 23) & (players["age"] <= 29)]
        veteran_players = players[players["age"] >= 30]

        # Valuasi
        valued_players = players[players["market_value_est"] > 0]
        total_mv = float(players["market_value_est"].sum())
        avg_mv = float(valued_players["market_value_est"].mean()) if not valued_players.empty else 0.0

        # Kuota Pemain
        quota_counts = players["quota_status"].value_counts().to_dict()
        lokal_count = quota_counts.get("Lokal", 0)
        asean_count = quota_counts.get("ASEAN", 0)
        asia_count = quota_counts.get("Asia (AFC)", 0)
        non_asia_count = quota_counts.get("Asing Non-Asia", 0)
        foreign_total = asean_count + asia_count + non_asia_count

        # Distribusi Grup Posisi
        pos_grp = players["position_group"].value_counts().to_dict()
        gk_count = pos_grp.get("Goalkeeper", 0)
        def_count = pos_grp.get("Defender", 0)
        mid_count = pos_grp.get("Midfielder", 0)
        att_count = pos_grp.get("Attack", 0)

        # Spesifik Posisi Kunci
        def has_pos(p_str):
            return players["position"].str.contains(p_str, case=False, na=False)

        cb_players = players[has_pos("Bek-Tengah")]
        lb_players = players[has_pos("Bek-Kiri")]
        rb_players = players[has_pos("Bek-Kanan")]
        dm_players = players[has_pos("Gel. Bertahan")]
        am_cm_players = players[has_pos("Gel. Tengah") | has_pos("Gel. Serang")]
        winger_players = players[has_pos("Sayap")]
        cf_players = players[has_pos("Depan-Tengah") | has_pos("Depan-Kedua")]

        # Postur Duel Udara
        cb_heights = cb_players[cb_players["height_cm"] > 140]["height_cm"]
        cb_height_avg = round(float(cb_heights.mean()), 1) if not cb_heights.empty else 180.0

        gk_heights = players[players["position_group"] == "Goalkeeper"]["height_cm"]
        gk_heights_valid = gk_heights[gk_heights > 140]
        gk_height_avg = round(float(gk_heights_valid.mean()), 1) if not gk_heights_valid.empty else 182.0

        # Pemain Kunci / Bintang
        top_stars = players.nlargest(3, "market_value_est")
        star_names = [f"{r['player_name']} ({r['position']} - {r['market_value_raw']})" for _, r in top_stars.iterrows()]

        return {
            "team": team_name,
            "squad_size": squad_size,
            "avg_age": avg_age,
            "u22_count": len(u22_players),
            "u22_pct": round((len(u22_players) / squad_size * 100), 1) if squad_size > 0 else 0.0,
            "u22_names": u22_players["player_name"].tolist(),
            "prime_count": len(prime_players),
            "veteran_count": len(veteran_players),
            "total_mv": total_mv,
            "avg_mv": avg_mv,
            "lokal_count": lokal_count,
            "asean_count": asean_count,
            "asia_count": asia_count,
            "non_asia_count": non_asia_count,
            "foreign_total": foreign_total,
            "gk_count": gk_count,
            "def_count": def_count,
            "mid_count": mid_count,
            "att_count": att_count,
            "cb_count": len(cb_players),
            "cb_height_avg": cb_height_avg,
            "lb_count": len(lb_players),
            "rb_count": len(rb_players),
            "dm_count": len(dm_players),
            "am_cm_count": len(am_cm_players),
            "winger_count": len(winger_players),
            "cf_count": len(cf_players),
            "gk_height_avg": gk_height_avg,
            "star_players": star_names,
        }

    # =========================================================================
    # 3. EVALUASI 5 SEKTOR & SECTOR HEALTH INDEX (0 - 100)
    # =========================================================================
    def evaluate_sectors(self, team_name):
        """
        Menghitung skor kesehatan (Health Index 0-100) untuk 5 sektor utama:
        1. 🛡️ Sektor Kiper (Goalkeeping Security)
        2. 🏰 Sektor Pertahanan (Defensive Solidity)
        3. ⚙️ Sektor Lini Tengah (Midfield Balance & Transition)
        4. ⚡ Sektor Lini Serang (Attacking Sharpness)
        5. 👶 Sektor Kedalaman & Regulasi U-22 (Youth & Squad Depth)
        """
        m_stats = self.get_club_match_stats(team_name)
        s_prof = self.get_club_squad_profile(team_name)

        if not m_stats or not s_prof:
            return None

        league = m_stats.get("league", "Indonesia (Liga 1)")
        bench = self.league_benchmarks.get(
            league,
            {"avg_gf_per_match": 1.25, "avg_ga_per_match": 1.25, "avg_ppg": 1.35},
        )

        ga_match = float(m_stats.get("ga_per_match", 1.2))
        gf_match = float(m_stats.get("gf_per_match", 1.2))
        bench_ga = float(bench.get("avg_ga_per_match", 1.25))
        bench_gf = float(bench.get("avg_gf_per_match", 1.25))
        clean_sheet_rate = float(m_stats.get("clean_sheet_rate", 20.0))

        sectors = {}

        # -------------------------------------------------------------
        # 1. 🛡️ SEKTOR PENJAGA GAWANG (GOALKEEPING SECURITY)
        # -------------------------------------------------------------
        gk_count = s_prof["gk_count"]
        gk_height = s_prof["gk_height_avg"]
        # Rasio kebobolan: semakin sedikit kebobolan, skor semakin tinggi
        ga_ratio = max(0.2, min(2.5, ga_match / max(bench_ga, 0.5)))
        gk_perf_score = (2.0 - ga_ratio) * 45.0  # ~45-90 poin
        gk_depth_score = min(20.0, gk_count * 6.5)
        gk_height_bonus = 10.0 if gk_height >= 184 else (5.0 if gk_height >= 180 else 0.0)
        gk_cs_bonus = (clean_sheet_rate / 100.0) * 20.0

        raw_gk_score = float(np.clip(gk_perf_score + gk_depth_score + gk_height_bonus + gk_cs_bonus, 25.0, 98.0))

        gk_status = "Solid 🟢" if raw_gk_score >= 78 else ("Waspada 🟡" if raw_gk_score >= 60 else "Kritis 🔴")
        gk_diag = []
        if ga_match > bench_ga * 1.15:
            gk_diag.append(f"Rasio kebobolan tinggi ({ga_match:.2f}/laga vs rerata liga {bench_ga:.2f}).")
        if gk_count < 3:
            gk_diag.append(f"Kedalaman kiper tipis (hanya ada {gk_count} penjaga gawang terdaftar).")
        if gk_height < 182:
            gk_diag.append(f"Rerata postur kiper ({gk_height:.0f} cm) rentan duel udara/crossing silang.")
        if not gk_diag:
            gk_diag.append(f"Keamanan gawang stabil dengan {m_stats.get('clean_sheets', 0)} laga nirbobol (Clean Sheet).")

        sectors["Penjaga Gawang"] = {
            "name": "🛡️ Sektor Penjaga Gawang",
            "score": round(raw_gk_score, 1),
            "status": gk_status,
            "benchmark_diff": round(bench_ga - ga_match, 2),  # Positif artinya kebobolan lebih sedikit dari liga
            "diagnosis": " ".join(gk_diag),
            "key_metrics": {
                "Kebobolan/Laga": f"{ga_match:.2f}",
                "Clean Sheet Rate": f"{clean_sheet_rate:.1f}%",
                "Jumlah Kiper": f"{gk_count} Pemain",
                "Rerata Tinggi Kiper": f"{gk_height:.0f} cm",
            },
        }

        # -------------------------------------------------------------
        # 2. 🏰 SEKTOR PERTAHANAN (DEFENSIVE SOLIDITY)
        # -------------------------------------------------------------
        cb_count = s_prof["cb_count"]
        lb_count = s_prof["lb_count"]
        rb_count = s_prof["rb_count"]
        cb_height = s_prof["cb_height_avg"]

        def_ga_score = (2.0 - ga_ratio) * 45.0
        cb_aerial_score = 15.0 if cb_height >= 185 else (10.0 if cb_height >= 182 else 4.0)
        fullback_balance = 15.0 if (lb_count >= 2 and rb_count >= 2) else (8.0 if (lb_count >= 1 and rb_count >= 1) else 2.0)
        cb_depth = 15.0 if cb_count >= 4 else (9.0 if cb_count == 3 else 3.0)

        raw_def_score = float(np.clip(def_ga_score + cb_aerial_score + fullback_balance + cb_depth, 25.0, 98.0))
        def_status = "Solid 🟢" if raw_def_score >= 78 else ("Waspada 🟡" if raw_def_score >= 60 else "Kritis 🔴")

        def_diag = []
        if ga_match > bench_ga:
            def_diag.append(f"Lini belakang kerap bocor ({ga_match:.2f} gol kemasukan per laga).")
        if cb_height < 183:
            def_diag.append(f"Rerata postur bek tengah ({cb_height:.0f} cm) kurang dominan untuk set-piece udara.")
        if lb_count < 2 or rb_count < 2:
            def_diag.append(f"Defisit bek sayap murni (Bek Kiri: {lb_count}, Bek Kanan: {rb_count}) membatasi fleksibilitas rotasi.")
        if not def_diag:
            def_diag.append("Struktur lini pertahanan sangat kokoh dan memiliki postur duel udara yang dominan.")

        sectors["Lini Pertahanan"] = {
            "name": "🏰 Sektor Lini Pertahanan",
            "score": round(raw_def_score, 1),
            "status": def_status,
            "benchmark_diff": round(bench_ga - ga_match, 2),
            "diagnosis": " ".join(def_diag),
            "key_metrics": {
                "Kebobolan/Laga": f"{ga_match:.2f}",
                "Bek Tengah (CB)": f"{cb_count} Pemain",
                "Postur Bek Tengah": f"{cb_height:.0f} cm",
                "Bek Sayap (LB/RB)": f"{lb_count} LB / {rb_count} RB",
            },
        }

        # -------------------------------------------------------------
        # 3. ⚙️ LINI TENGAH & TRANSISI (MIDFIELD CONTROL & BALANCE)
        # -------------------------------------------------------------
        dm_count = s_prof["dm_count"]
        am_cm_count = s_prof["am_cm_count"]
        mid_count = s_prof["mid_count"]

        # Keseimbangan jangkar pertahanan vs motor kreativitas
        dm_anchor_score = 25.0 if dm_count >= 2 else (14.0 if dm_count == 1 else 4.0)
        creative_score = 25.0 if am_cm_count >= 3 else (15.0 if am_cm_count >= 2 else 6.0)
        overall_mid_score = min(25.0, mid_count * 3.5)
        gd_contrib = float(np.clip(15.0 + float(m_stats.get("gd", 0)) * 1.5, 5.0, 25.0))

        raw_mid_score = float(np.clip(dm_anchor_score + creative_score + overall_mid_score + gd_contrib, 25.0, 98.0))
        mid_status = "Solid 🟢" if raw_mid_score >= 78 else ("Waspada 🟡" if raw_mid_score >= 60 else "Kritis 🔴")

        mid_diag = []
        if dm_count == 0:
            mid_diag.append("Tidak memiliki Gelandang Bertahan (DM) murni sebagai pemutus serangan lawan.")
        elif dm_count == 1:
            mid_diag.append("Hanya memiliki 1 gelandang bertahan murni, risiko fatal jika cedera atau terkena akumulasi kartu.")
        if am_cm_count < 3:
            mid_diag.append("Pilihan kreator serangan (AM/CM) minim, aliran bola ke lini depan rentan mandek.")
        if not mid_diag:
            mid_diag.append("Lini tengah seimbang antara fungsi filter pertahanan dan transisi serangan dinamis.")

        sectors["Lini Tengah"] = {
            "name": "⚙️ Sektor Lini Tengah",
            "score": round(raw_mid_score, 1),
            "status": mid_status,
            "benchmark_diff": round(float(m_stats.get("ppg", 1.0)) - float(bench.get("avg_ppg", 1.35)), 2),
            "diagnosis": " ".join(mid_diag),
            "key_metrics": {
                "Total Gelandang": f"{mid_count} Pemain",
                "Gel. Bertahan (DM)": f"{dm_count} Pemain",
                "Gel. Kreatif (CM/AM)": f"{am_cm_count} Pemain",
                "Points Per Game (PPG)": f"{m_stats.get('ppg', 0.0)}",
            },
        }

        # -------------------------------------------------------------
        # 4. ⚡ LINI SERANG & KETAJAMAN (ATTACKING SHARPNESS)
        # -------------------------------------------------------------
        cf_count = s_prof["cf_count"]
        winger_count = s_prof["winger_count"]
        gf_ratio = max(0.2, min(2.5, gf_match / max(bench_gf, 0.5)))
        gf_perf_score = (gf_ratio / 1.5) * 45.0  # produktivitas gol
        cf_depth_score = 25.0 if cf_count >= 3 else (16.0 if cf_count == 2 else 6.0)
        wing_threat_score = 20.0 if winger_count >= 4 else (12.0 if winger_count >= 2 else 4.0)
        win_rate_bonus = (float(m_stats.get("win_rate", 30.0)) / 100.0) * 10.0

        raw_att_score = float(np.clip(gf_perf_score + cf_depth_score + wing_threat_score + win_rate_bonus, 25.0, 98.0))
        att_status = "Solid 🟢" if raw_att_score >= 78 else ("Waspada 🟡" if raw_att_score >= 60 else "Kritis 🔴")

        att_diag = []
        if gf_match < bench_gf:
            att_diag.append(f"Produktivitas gol rendah ({gf_match:.2f}/laga vs rerata liga {bench_gf:.2f}).")
        if cf_count <= 1:
            att_diag.append("Krisis striker murni (hanya ada 1 target man), sangat bergantung pada opsi tunggal.")
        if winger_count < 3:
            att_diag.append("Minim penyerang sayap murni untuk membongkar pertahanan blok rendah (low-block).")
        if not att_diag:
            att_diag.append(f"Serangan agresif dan tajam dengan rata-rata {gf_match:.2f} gol per pertandingan.")

        sectors["Lini Serang"] = {
            "name": "⚡ Sektor Lini Serang",
            "score": round(raw_att_score, 1),
            "status": att_status,
            "benchmark_diff": round(gf_match - bench_gf, 2),  # Positif artinya mencetak lebih banyak dari rata-rata liga
            "diagnosis": " ".join(att_diag),
            "key_metrics": {
                "Gol Dicetak/Laga": f"{gf_match:.2f}",
                "Striker Murni (CF)": f"{cf_count} Pemain",
                "Penyerang Sayap": f"{winger_count} Pemain",
                "Win Rate Klub": f"{m_stats.get('win_rate', 0.0)}%",
            },
        }

        # -------------------------------------------------------------
        # 5. 👶 KEDALAMAN SKUAD & REGULASI U-22 (YOUTH & DEPTH)
        # -------------------------------------------------------------
        u22_count = s_prof["u22_count"]
        avg_age = s_prof["avg_age"]
        foreign_tot = s_prof["foreign_total"]

        # Skor U-22: Butuh minimal 5-8 pemain muda untuk rotasi starter wajib U-22 LIB
        u22_score = 35.0 if u22_count >= 8 else (25.0 if u22_count >= 5 else (12.0 if u22_count >= 3 else 4.0))
        # Usia optimal: 24 - 27 tahun
        age_balance_score = 30.0 if avg_age <= 26.5 else (20.0 if avg_age <= 28.5 else 8.0)
        depth_size_score = 25.0 if s_prof["squad_size"] >= 26 else (15.0 if s_prof["squad_size"] >= 22 else 5.0)
        foreign_slot_score = 10.0 if 3 <= foreign_tot <= 8 else 5.0

        raw_youth_score = float(np.clip(u22_score + age_balance_score + depth_size_score + foreign_slot_score, 25.0, 98.0))
        youth_status = "Solid 🟢" if raw_youth_score >= 78 else ("Waspada 🟡" if raw_youth_score >= 60 else "Kritis 🔴")

        youth_diag = []
        if u22_count < 5:
            youth_diag.append(f"Kekurangan opsi U-22 (hanya {u22_count} pemain), berisiko melanggar regulasi wajib starter U-22 PSSI/LIB.")
        if avg_age >= 28.5:
            youth_diag.append(f"Skuad menua (rerata usia {avg_age:.1f} tahun), rentan penurunan fisik di fase akhir liga.")
        if s_prof["squad_size"] < 24:
            youth_diag.append(f"Ukuran skuad tipis ({s_prof['squad_size']} pemain) untuk kompetisi berjadwal padat.")
        if not youth_diag:
            youth_diag.append(f"Regenerasi skuad sehat ({u22_count} talenta muda U-22) dan kedalaman bangku cadangan sangat siap.")

        sectors["Regenerasi & U-22"] = {
            "name": "👶 Sektor Regenerasi & U-22",
            "score": round(raw_youth_score, 1),
            "status": youth_status,
            "benchmark_diff": round(25.5 - avg_age, 1),
            "diagnosis": " ".join(youth_diag),
            "key_metrics": {
                "Pemain U-22": f"{u22_count} Pemain ({s_prof['u22_pct']}%)",
                "Rerata Usia": f"{avg_age:.1f} Tahun",
                "Total Skuad": f"{s_prof['squad_size']} Pemain",
                "Pemain Asing": f"{foreign_tot} Pemain",
            },
        }

        # Urutkan sektor berdasarkan skor terendah (Sektor paling butuh perbaikan)
        sorted_priorities = sorted(sectors.items(), key=lambda kv: kv[1]["score"])

        return {
            "team": team_name,
            "overall_health_score": round(float(np.mean([s["score"] for s in sectors.values()])), 1),
            "sectors": sectors,
            "priorities": [k for k, _ in sorted_priorities],
        }

    # =========================================================================
    # 4. REKOMENDASI SOLUSI TAKTIKAL & TARGET TRANSFER KONKRET
    # =========================================================================
    def get_actionable_recommendations(self, team_name, scout_brain=None, top_n=3):
        """
        Menghasilkan rekomendasi konkret:
        1. Arahan Taktikal & Manajerial untuk sektor yang kritis.
        2. Rekomendasi pemain rekrutan spesifik dari database 3.060 pemain yang sesuai budget klub.
        """
        diag = self.evaluate_sectors(team_name)
        s_prof = self.get_club_squad_profile(team_name)
        if not diag or not s_prof:
            return []

        avg_budget = s_prof.get("avg_mv", 1_000_000_000)
        max_budget = avg_budget * 2.2 if avg_budget > 0 else 3_000_000_000

        recommendations = []
        # Evaluasi sektor yang membutuhkan perbaikan (skor < 78 atau top 2 prioritas terendah)
        priority_keys = diag["priorities"][:2]

        for sec_key in priority_keys:
            sec_info = diag["sectors"][sec_key]
            score = sec_info["score"]

            # Petakan sektor ke posisi lapangan spesifik
            target_positions = []
            target_profile = ""
            u22_req = False
            min_height = 0

            if sec_key == "Penjaga Gawang":
                target_positions = ["Kiper"]
                target_profile = "Penjaga Gawang berpengalaman dengan postur >= 185 cm dan refleks solid."
                min_height = 184
            elif sec_key == "Lini Pertahanan":
                if s_prof["cb_height_avg"] < 184 or s_prof["cb_count"] <= 3:
                    target_positions = ["Bek-Tengah"]
                    target_profile = "Bek Tengah tangguh (Tower Defender) dominan duel udara (>= 185 cm) untuk menutup set-piece."
                    min_height = 185
                elif s_prof["lb_count"] <= 1:
                    target_positions = ["Bek-Kiri"]
                    target_profile = "Bek Kiri murni berkaki dominan kiri dengan work-rate tinggi dan naluri crossing."
                else:
                    target_positions = ["Bek-Kanan", "Bek-Kiri"]
                    target_profile = "Bek sayap modern dengan kemampuan transisi bertahan-menyerang cepat."
            elif sec_key == "Lini Tengah":
                if s_prof["dm_count"] <= 1:
                    target_positions = ["Gel. Bertahan"]
                    target_profile = "Gelandang Bertahan (Anchor Destroyer) agresif sebagai filter pertama lini pertahanan."
                else:
                    target_positions = ["Gel. Tengah", "Gel. Serang"]
                    target_profile = "Playmaker / Gelandang Kreatif bervisi tinggi untuk mempercepat progresi serangan."
            elif sec_key == "Lini Serang":
                if s_prof["cf_count"] <= 1:
                    target_positions = ["Depan-Tengah"]
                    target_profile = "Striker Nomor 9 Murni (Clinical Finisher) dengan insting gol tajam di kotak penalti."
                else:
                    target_positions = ["Sayap Kiri", "Sayap Kanan"]
                    target_profile = "Winger lincah dengan atribut 1v1 tinggi untuk membongkar blok pertahanan rendah."
            else:  # Regenerasi & U-22
                target_positions = ["Sayap Kiri", "Bek-Tengah", "Gel. Tengah", "Depan-Tengah"]
                target_profile = "Wonderkid U-22 siap pakai untuk mengamankan regulasi starter wajib U-22 Liga Indonesia."
                u22_req = True

            # Cari pemain kandidat terbaik dari database
            candidates_df = pd.DataFrame()
            if not self.df_players.empty:
                cand_pool = self.df_players[
                    (self.df_players["team"] != team_name)
                    & (self.df_players["position"].isin(target_positions))
                    & (self.df_players["market_value_est"] > 0)
                    & (self.df_players["market_value_est"] <= max_budget)
                ].copy()

                if u22_req:
                    cand_pool = cand_pool[cand_pool["is_u22"] == True]
                if min_height > 0:
                    cand_pool = cand_pool[cand_pool["height_cm"] >= min_height]

                if not cand_pool.empty:
                    # Ranking kandidat berdasarkan Undervalued Ratio dan Usia
                    cand_pool["rec_score"] = (
                        cand_pool["market_value_est"] / 1e8 * 0.3
                        + (28.0 - cand_pool["age"].clip(17, 34)) * 2.0
                        + np.where(cand_pool["quota_status"] == "Lokal", 10.0, 5.0)
                    )
                    candidates_df = cand_pool.nlargest(top_n, "rec_score")[
                        [
                            "player_name",
                            "team",
                            "league",
                            "position",
                            "age",
                            "height_cm",
                            "foot",
                            "quota_status",
                            "market_value_raw",
                            "contract_expiry",
                        ]
                    ]

            recommendations.append({
                "sector": sec_key,
                "sector_name": sec_info["name"],
                "score": score,
                "status": sec_info["status"],
                "diagnosis": sec_info["diagnosis"],
                "tactical_profile": target_profile,
                "target_positions": target_positions,
                "candidates": candidates_df,
            })

        return recommendations

    # =========================================================================
    # 5. AI EXECUTIVE TACTICAL & CLUB DIAGNOSTIC REPORT
    # =========================================================================
    def generate_club_diagnostic_report(self, team_name, api_key=None):
        """
        Menghasilkan Laporan Diagnostik Taktikal Eksekutif untuk Dewan Direksi / Pelatih Kepala.
        - Memanggil gemini-3.8-flash jika api_key tersedia.
        - Menggunakan engine analitik internal ScoutBrain jika tanpa API key.
        """
        m_stats = self.get_club_match_stats(team_name)
        s_prof = self.get_club_squad_profile(team_name)
        diag = self.evaluate_sectors(team_name)
        recs = self.get_actionable_recommendations(team_name)

        if not m_stats or not s_prof or not diag:
            return "Data klub tidak ditemukan.", "Error"

        league = m_stats["league"]
        rank = m_stats["rank"]
        group = m_stats.get("group", "Utama")
        p = m_stats["played"]
        w, d, l = m_stats["won"], m_stats["drawn"], m_stats["lost"]
        pts, ppg = m_stats["points"], m_stats["ppg"]
        gf, ga, gd = m_stats["gf"], m_stats["ga"], m_stats["gd"]
        form = m_stats.get("form", "-")
        health = diag["overall_health_score"]

        # Susun ringkasan sektor
        sec_summaries = []
        for s_k, s_v in diag["sectors"].items():
            sec_summaries.append(f"- **{s_v['name']}**: Skor {s_v['score']}/100 ({s_v['status']}) — {s_v['diagnosis']}")

        rec_summaries = []
        for r in recs:
            cand_names = []
            if not r["candidates"].empty:
                for _, crow in r["candidates"].iterrows():
                    cand_names.append(f"{crow['player_name']} ({crow['team']}, {crow['position']}, {crow['market_value_raw']})")
            rec_summaries.append(
                f"- **Sektor {r['sector']}**: {r['tactical_profile']}\n  *Rekomendasi Pemain:* {'; '.join(cand_names) if cand_names else 'Pencarian scouting kustom'}"
            )

        if api_key and api_key.strip():
            prompt = f"""Anda adalah Direktur Teknik & Kepala Analis Taktik Sepak Bola profesional.
Buatlah Laporan Diagnostik Taktikal & Rencana Perbaikan Eksekutif untuk Klub berikut dalam Bahasa Indonesia:

- Nama Klub: {team_name}
- Kompetisi: {league} (Peringkat {rank}, {group})
- Rekam Pertandingan: {p} Laga | {w} Menang, {d} Seri, {l} Kalah | {pts} Poin (PPG {ppg})
- Produktivitas & Pertahanan: Gol Memasukkan: {gf} ({m_stats.get('gf_per_match', 0)}/laga) | Kemasukan: {ga} ({m_stats.get('ga_per_match', 0)}/laga) | Selisih Gol: {gd}
- Tren Performa 5 Laga Terakhir: {form}
- Profil Skuad: {s_prof['squad_size']} Pemain, Rerata Usia {s_prof['avg_age']} Thn, {s_prof['u22_count']} Pemain U-22 ({s_prof['u22_pct']}%), {s_prof['foreign_total']} Pemain Asing.
- Skor Kesehatan Sektor (0-100):
{chr(10).join(sec_summaries)}

Sektor yang Paling Membutuhkan Perbaikan:
{chr(10).join(rec_summaries)}

Susun laporan dengan struktur tajam:
1. Ringkasan Eksekutif & Raport Performa Musim Ini
2. Bedah Titik Lemah Taktikal (Mengapa Lini Kritis Mengalami Kendala)
3. Rekomendasi Solusi Taktik Pelatih (Formasi / Penyesuaian Pressing / Distribusi)
4. Rencana Aksi Transfer & Rekrutmen Pemain Spesifik"""

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

        # Fallback Internal Engine
        report_md = f"""### 📋 Executive Club Diagnostic & Tactical Report: **{team_name}**
**Kompetisi:** {league} | **Peringkat:** #{rank} ({group}) | **Overall Sector Health:** **{health}/100**

---

#### 1. Ringkasan Rapor & Performa Pertandingan
* **Rekam Laga:** {p} Main — **{w} Menang, {d} Seri, {l} Kalah** (**{pts} Poin**, PPG: **{ppg}**).
* **Produktivitas & Pertahanan:** Memasukkan **{gf} gol** ({m_stats.get('gf_per_match', 0):.2f}/laga) vs Kemasukan **{ga} gol** ({m_stats.get('ga_per_match', 0):.2f}/laga), Selisih Gol: **{gd:+d}**.
* **Tren Form Terakhir:** `{form}` | Rasio Kemenangan: **{m_stats.get('win_rate', 0)}%**.
* **Struktur Skuad:** {s_prof['squad_size']} pemain terdaftar, rerata usia **{s_prof['avg_age']:.1f} tahun**, **{s_prof['u22_count']} talenta U-22**, dan **{s_prof['foreign_total']} pemain asing**.

#### 2. Evaluasi Diagnosis 5 Sektor Lapangan
{chr(10).join(sec_summaries)}

#### 3. Rencana Aksi Solusi & Rekomendasi Transfer
{chr(10).join(rec_summaries)}
"""
        return report_md, "ClubAnalytics v6.0 Tactical Engine"
