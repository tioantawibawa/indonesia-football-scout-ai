import glob
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler


INDONESIAN_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "mei": 5, "may": 5,
    "jun": 6, "jul": 7, "agt": 8, "agu": 8, "aug": 8, "sep": 9,
    "okt": 10, "oct": 10, "nov": 11, "des": 12, "dec": 12,
}


class ScoutBrain:
    def __init__(self, data_path="data/processed/master_player_db.csv", raw_dir="data/raw"):
        self.data_path = data_path
        self.raw_dir = raw_dir

        # Load Data Utama
        self.df = pd.read_csv(data_path)

        # Standardize Columns untuk kompatibilitas penuh dengan app.py
        if "league_country" in self.df.columns and "league" not in self.df.columns:
            self.df.rename(columns={"league_country": "league"}, inplace=True)

        self._prepare_features()
        self._fit_role_clusters_and_valuation_model()

    # =========================================================================
    # 1. FEATURE ENGINEERING & PREPROCESSING
    # =========================================================================
    @staticmethod
    def _parse_contract_date(val):
        """Konversi string tanggal kontrak ('2027-05-31' atau '31 Mei 2027') menjadi datetime."""
        if pd.isna(val):
            return None
        s = str(val).strip()
        if not s or s in ("-", "0", "None", "nan"):
            return None

        # Format ISO YYYY-MM-DD
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            pass

        # Format Bahasa Indonesia: "31 Mei 2027"
        m = re.match(r"^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$", s)
        if m:
            day = int(m.group(1))
            mon_str = m.group(2)[:3].lower()
            year = int(m.group(3))
            mon = INDONESIAN_MONTHS.get(mon_str)
            if mon:
                try:
                    return datetime(year, mon, day)
                except ValueError:
                    return None
        return None

    def _prepare_features(self):
        """Menyiapkan fitur numerik, imputasi cerdas per posisi, dan fitur turunan."""
        df = self.df

        # Pastikan kolom-kolom baru selalu tersedia meskipun menggunakan CSV versi lama
        defaults = {
            "player_id": "",
            "shirt_number": 0,
            "competition_id": "UNK",
            "league_tier": 1,
            "position_group": "Midfielder",
            "age": 0,
            "birth_date": "-",
            "is_u22": False,
            "nationality": "Indonesia",
            "quota_status": "Lokal",
            "height_cm": 0,
            "foot": "kanan",
            "joined_date": "-",
            "previous_club": "-",
            "contract_expiry": "-",
            "market_value_raw": "-",
            "market_value_est": 0,
            "prev_market_value_est": 0,
            "mv_trend": "STABLE",
            "appearances": 0,
            "goals": 0,
            "assists": 0,
            "minutes_played": 0,
            "ppg": 0.0,
            "yellow_cards": 0,
            "red_cards": 0,
        }
        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val

        # Normalisasi tipe data numerik
        num_cols = [
            "age", "height_cm", "league_tier", "market_value_est",
            "prev_market_value_est", "appearances", "goals", "assists",
            "minutes_played", "ppg", "yellow_cards", "red_cards",
        ]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Imputasi usia (jika 0, gunakan median posisi; default 24)
        valid_ages = df[df["age"] > 0].groupby("position")["age"].median().to_dict()
        global_median_age = df.loc[df["age"] > 0, "age"].median()
        if pd.isna(global_median_age) or global_median_age <= 0:
            global_median_age = 25.0

        df["age_clean"] = df.apply(
            lambda r: r["age"] if r["age"] > 0 else valid_ages.get(r["position"], global_median_age),
            axis=1,
        )
        df["is_u22"] = (df["age"] > 0) & (df["age"] <= 22)

        # Imputasi tinggi badan per posisi (jika 0, gunakan median posisi)
        valid_heights = df[df["height_cm"] > 140].groupby("position")["height_cm"].median().to_dict()
        global_median_height = df.loc[df["height_cm"] > 140, "height_cm"].median()
        if pd.isna(global_median_height) or global_median_height <= 0:
            global_median_height = 176.0

        df["height_clean"] = df.apply(
            lambda r: r["height_cm"]
            if r["height_cm"] > 140
            else valid_heights.get(r["position"], global_median_height),
            axis=1,
        )

        # Transformasi Logaritmik Market Value untuk kestabilan jarak fitur
        df["log_mv"] = np.log1p(df["market_value_est"].clip(lower=0))

        # Momentum / Perubahan Nilai Pasar (%)
        prev_mv = df["prev_market_value_est"].where(df["prev_market_value_est"] > 0, df["market_value_est"])
        df["mv_change_idr"] = df["market_value_est"] - prev_mv
        df["mv_growth_pct"] = np.where(
            prev_mv > 0,
            ((df["market_value_est"] - prev_mv) / prev_mv) * 100.0,
            0.0,
        ).round(2)

        # Encoding Kaki Dominan
        foot_norm = df["foot"].astype(str).str.lower().str.strip()
        df["foot_left"] = foot_norm.isin(["kiri", "left", "keduanya", "both"]).astype(float)
        df["foot_right"] = foot_norm.isin(["kanan", "right", "keduanya", "both"]).astype(float)
        df["foot_both"] = foot_norm.isin(["keduanya", "both"]).astype(float)

        # Rasio Nilai Pemain terhadap Rata-rata Klub (Squad Standing / Key Player Index)
        team_avg_mv = df.groupby("team")["market_value_est"].transform(
            lambda s: max(s[s > 0].mean(), 100_000_000) if (s > 0).any() else 100_000_000
        )
        df["squad_value_ratio"] = (df["market_value_est"] / team_avg_mv).clip(0, 5.0)

        # Sisa Kontrak (dalam bulan) terhadap tanggal referensi
        ref_date = datetime.now()
        contract_dates = df["contract_expiry"].apply(self._parse_contract_date)
        df["contract_months_left"] = contract_dates.apply(
            lambda d: max(int(round((d - ref_date).days / 30.44)), -1) if pd.notna(d) else -1
        )

        # Fitur matriks untuk Similarity Engine
        self.features = [
            "age_clean",
            "log_mv",
            "market_value_est",
            "height_clean",
            "foot_left",
            "foot_right",
            "league_tier",
            "squad_value_ratio",
        ]
        self.scaler = StandardScaler()

    def _fit_role_clusters_and_valuation_model(self):
        """
        Melatih model Machine Learning:
        1. Expected Market Value Regressor (RandomForest) untuk mendeteksi pemain Undervalued.
        2. K-Means Role/Tier Archetype Clustering.
        """
        df = self.df

        # --- 1. Expected Market Value Model (Undervalued Detector) ---
        pos_codes = df["position"].astype("category").cat.codes
        quota_codes = df["quota_status"].astype("category").cat.codes

        X_val = pd.DataFrame({
            "age": df["age_clean"],
            "age_sq": df["age_clean"] ** 2,
            "height": df["height_clean"],
            "tier": df["league_tier"],
            "pos_code": pos_codes,
            "quota_code": quota_codes,
            "foot_both": df["foot_both"],
            "foot_left": df["foot_left"],
            "squad_ratio": df["squad_value_ratio"],
        })

        valued_mask = df["market_value_est"] > 0
        if valued_mask.sum() >= 30:
            rf = RandomForestRegressor(
                n_estimators=80,
                max_depth=8,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
            y_train = np.log1p(df.loc[valued_mask, "market_value_est"])
            rf.fit(X_val.loc[valued_mask], y_train)
            pred_log = rf.predict(X_val)
            df["expected_mv_est"] = np.expm1(pred_log).round(-5)
        else:
            df["expected_mv_est"] = df["market_value_est"]

        # Value Gap: Positif artinya Expected > Harga Aktual (Undervalued!)
        df["undervalued_ratio"] = np.where(
            df["market_value_est"] > 0,
            (df["expected_mv_est"] / df["market_value_est"]).round(2),
            1.0,
        )

        # --- 2. Player Archetype Clustering (K-Means) ---
        cluster_feats = df[["age_clean", "log_mv", "height_clean", "squad_value_ratio"]].fillna(0)
        X_clust = StandardScaler().fit_transform(cluster_feats)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df["cluster_id"] = kmeans.fit_predict(X_clust)

        # Beri label arketipe berdasarkan karakteristik tiap pemain
        def _assign_archetype(row):
            age = row["age_clean"]
            mv = row["market_value_est"]
            ratio = row["squad_value_ratio"]
            if age <= 22 and mv > 0:
                return "💎 Wonderkid / U-22 Prospect"
            if ratio >= 1.5 and age <= 29:
                return "⭐ Marquee / Key Star"
            if age >= 30 and ratio >= 1.0:
                return "🛡️ Veteran Leader"
            if row["undervalued_ratio"] >= 1.25 and age <= 26:
                return "🚀 Undervalued Moneyball"
            return "⚙️ Squad Rotation"

        df["player_archetype"] = df.apply(_assign_archetype, axis=1)

    # =========================================================================
    # 2. MULTI-ATTRIBUTE SIMILARITY ENGINE (REPLACEMENT FINDER)
    # =========================================================================
    def get_similar_players(
        self,
        player_name,
        top_n=10,
        same_foot_only=False,
        quota_filter=None,
        max_budget=None,
        target_leagues=None,
    ):
        """
        Mencari pemain pengganti 'apple-to-apple' menggunakan Hybrid Weighted Cosine + Gaussian Distance
        pada profil multi-atribut (Usia, Valuasi, Tinggi Badan, Kaki Dominan, Tier, dan Squad Standing).
        """
        if player_name not in self.df["player_name"].values:
            return None

        target_row = self.df[self.df["player_name"] == player_name].iloc[0]
        target_pos = target_row["position"]
        target_pos_group = target_row["position_group"]

        # Filter dasar: posisi harus sama
        df_pos = self.df[self.df["position"] == target_pos].copy()

        # Pastikan pemain target tetap ada di df_pos sebelum filter opsional diterapkan
        df_pos = df_pos.reset_index(drop=True)
        target_matches = df_pos[df_pos["player_name"] == player_name].index
        if len(target_matches) == 0:
            return None
        target_idx = target_matches[0]

        # Bobot fitur yang disesuaikan dengan grup posisi (Position-Aware Feature Weights)
        # Urutan: [age_clean, log_mv, market_value_est, height_clean, foot_left, foot_right, league_tier, squad_value_ratio]
        if target_pos_group in ("Goalkeeper", "Defender"):
            weights = np.array([1.4, 1.8, 1.2, 1.5, 1.1, 1.1, 0.8, 1.0])
        elif "Sayap" in target_pos or "Bek-Kiri" in target_pos or "Bek-Kanan" in target_pos:
            weights = np.array([1.5, 1.8, 1.2, 0.8, 1.6, 1.6, 0.8, 1.0])
        else:
            weights = np.array([1.5, 1.8, 1.3, 1.0, 1.1, 1.1, 0.8, 1.1])

        X = df_pos[self.features].values
        X_scaled = self.scaler.fit_transform(X) * weights
        target_vector = X_scaled[target_idx].reshape(1, -1)

        # Kombinasi Cosine Similarity (arah profil) + Gaussian RBF Distance (kedekatan nilai absolut)
        cos_sim = cosine_similarity(target_vector, X_scaled)[0]
        cos_sim_norm = (cos_sim + 1.0) / 2.0  # skala 0..1

        euc_dist = euclidean_distances(target_vector, X_scaled)[0]
        rbf_sim = np.exp(-0.25 * euc_dist)  # skala 0..1

        hybrid_scores = (0.45 * cos_sim_norm + 0.55 * rbf_sim) * 100.0
        df_pos["similarity_score"] = hybrid_scores.round(1)

        # Buang diri sendiri dari kandidat
        candidates = df_pos[df_pos["player_name"] != player_name].copy()

        # Terapkan filter opsional jika diminta
        if same_foot_only and target_row["foot"] in ("kanan", "kiri", "keduanya"):
            candidates = candidates[candidates["foot"] == target_row["foot"]]
        if quota_filter and quota_filter != "Semua":
            candidates = candidates[candidates["quota_status"] == quota_filter]
        if max_budget and max_budget > 0:
            candidates = candidates[candidates["market_value_est"] <= max_budget]
        if target_leagues:
            candidates = candidates[candidates["league"].isin(target_leagues)]

        result = candidates.nlargest(top_n, "similarity_score").copy()

        out_cols = [
            "player_name", "team", "league", "position", "age",
            "height_cm", "foot", "nationality", "quota_status",
            "market_value_raw", "similarity_score",
        ]
        available_cols = [c for c in out_cols if c in result.columns]
        return result[available_cols]

    # =========================================================================
    # 3. SMART SQUAD PLANNER & REGULASI LIGA (SQUAD PLANNER)
    # =========================================================================
    def recommend_for_team_needs(
        self,
        team_name,
        target_position,
        top_n=5,
        quota_filter="Semua",
        u22_only=False,
        preferred_foot="Semua",
        strategy="balanced",
        custom_max_budget=None,
    ):
        """
        Mencari rekomendasi pemain baru untuk posisi tertentu yang sesuai dengan kapasitas budget klub,
        regulasi kuota pemain (Lokal/ASEAN/Asia/Asing/U-22), dan strategi transfer klub.
        """
        team_players = self.df[self.df["team"] == team_name]
        if team_players.empty:
            return None

        valued_team_players = team_players[team_players["market_value_est"] > 0]
        if not valued_team_players.empty:
            avg_squad_value = valued_team_players["market_value_est"].nlargest(15).mean()
        else:
            avg_squad_value = 500_000_000  # Default Rp 500 Juta untuk klub Liga 3

        if pd.isna(avg_squad_value) or avg_squad_value <= 0:
            avg_squad_value = 1_000_000_000

        candidates = self.df[
            (self.df["position"] == target_position)
            & (self.df["team"] != team_name)
            & (self.df["age_clean"] > 0)
        ].copy()

        if candidates.empty:
            return pd.DataFrame()

        # Filter Regulasi & Preferensi Taktik
        if quota_filter and quota_filter != "Semua":
            candidates = candidates[candidates["quota_status"] == quota_filter]
        if u22_only:
            candidates = candidates[candidates["is_u22"] == True]
        if preferred_foot and preferred_foot != "Semua":
            candidates = candidates[candidates["foot"].str.lower() == preferred_foot.lower()]

        if candidates.empty:
            return pd.DataFrame()

        # Penentuan Rentang Budget
        if strategy == "moneyball":
            min_budget = 0
            max_budget = custom_max_budget if custom_max_budget else avg_squad_value * 1.2
        elif strategy == "wonderkid":
            min_budget = avg_squad_value * 0.15
            max_budget = custom_max_budget if custom_max_budget else avg_squad_value * 2.0
            candidates = candidates[candidates["age_clean"] <= 23]
        elif strategy == "proven":
            min_budget = avg_squad_value * 0.5
            max_budget = custom_max_budget if custom_max_budget else avg_squad_value * 3.0
            candidates = candidates[(candidates["age_clean"] >= 24) & (candidates["age_clean"] <= 30)]
        else:  # balanced
            min_budget = avg_squad_value * 0.25
            max_budget = custom_max_budget if custom_max_budget else avg_squad_value * 2.5

        filtered = candidates[
            (candidates["market_value_est"] >= min_budget)
            & (candidates["market_value_est"] <= max_budget)
        ].copy()

        # Fallback jika rentang budget terlalu sempit
        if filtered.empty:
            filtered = candidates[candidates["market_value_est"] <= max_budget].copy()
        if filtered.empty:
            filtered = candidates.copy()

        # Hitung Composite Scout Score (0 - 100)
        safe_age = filtered["age_clean"].clip(lower=16)
        max_cand_mv = max(float(filtered["market_value_est"].max()), 1.0)
        mv_norm = (filtered["market_value_est"] / max_cand_mv).clip(0, 1.0)

        if strategy == "moneyball":
            # Prioritaskan Undervalued Ratio & Efisiensi Usia
            raw_score = (
                filtered["undervalued_ratio"].clip(0.5, 3.0) * 35.0
                + (28.0 / safe_age) * 35.0
                + mv_norm * 20.0
                + np.where(filtered["mv_trend"] == "INCREASED", 10.0, 0.0)
            )
        elif strategy == "wonderkid":
            # Prioritaskan pemain muda dengan valuasi & potensi tertinggi
            youth_bonus = ((24.0 - safe_age).clip(lower=0) / 8.0) * 40.0
            raw_score = (
                youth_bonus
                + mv_norm * 45.0
                + np.where(filtered["mv_trend"] == "INCREASED", 15.0, 0.0)
            )
        else:
            # Balanced / Proven: Valuasi per usia + bonus momentum + postur
            val_per_age = filtered["market_value_est"] / safe_age
            max_vpa = max(val_per_age.max(), 1.0)
            raw_score = (
                (val_per_age / max_vpa) * 65.0
                + filtered["undervalued_ratio"].clip(0.5, 2.0) * 15.0
                + np.where(filtered["mv_trend"] == "INCREASED", 10.0, 0.0)
                + np.where((filtered["contract_months_left"] >= 0) & (filtered["contract_months_left"] <= 12), 10.0, 0.0)
            )

        filtered["scout_score"] = raw_score.clip(0, 99.9).round(1)

        out_cols = [
            "player_name", "team", "league", "age", "height_cm", "foot",
            "nationality", "quota_status", "contract_expiry",
            "market_value_raw", "player_archetype", "scout_score",
        ]
        available_cols = [c for c in out_cols if c in filtered.columns]
        return filtered.nlargest(top_n, "scout_score")[available_cols]

    # =========================================================================
    # 4. UNDERVALUED GEMS & PROMOTION RADAR (LIGA 2/3 -> LIGA 1)
    # =========================================================================
    def find_undervalued_gems(
        self,
        league_filter=None,
        position_filter=None,
        max_age=25,
        u22_only=False,
        top_n=15,
    ):
        """
        Mendeteksi pemain 'Undervalued Gems' (Expected Market Value jauh di atas harga saat ini,
        atau talenta muda Liga 2 / Liga 3 yang siap promosi ke kasta lebih tinggi).
        """
        df_sub = self.df[
            (self.df["market_value_est"] > 0)
            & (self.df["age_clean"] > 0)
            & (self.df["age_clean"] <= max_age)
        ].copy()

        if league_filter:
            if isinstance(league_filter, (list, tuple, set)):
                df_sub = df_sub[df_sub["league"].isin(league_filter)]
            elif league_filter != "Semua":
                df_sub = df_sub[df_sub["league"] == league_filter]

        if position_filter and position_filter != "Semua":
            df_sub = df_sub[df_sub["position"] == position_filter]

        if u22_only:
            df_sub = df_sub[df_sub["is_u22"] == True]

        if df_sub.empty:
            return pd.DataFrame()

        # Gem Score: gabungan Undervalued Ratio, usia muda, dan Squad Standing
        df_sub["gem_score"] = (
            df_sub["undervalued_ratio"].clip(0.5, 4.0) * 25.0
            + df_sub["squad_value_ratio"].clip(0, 3.0) * 15.0
            + ((26.0 - df_sub["age_clean"]).clip(lower=0) * 3.5)
            + np.where(df_sub["mv_trend"] == "INCREASED", 12.0, 0.0)
        ).round(1)

        df_sub["expected_mv_formatted"] = df_sub["expected_mv_est"].apply(
            lambda v: f"Rp{v/1e9:.2f}Mlyr." if v >= 1e9 else f"Rp{v/1e6:.0f}Jt."
        )

        out_cols = [
            "player_name", "team", "league", "position", "age",
            "nationality", "quota_status", "market_value_raw",
            "expected_mv_formatted", "undervalued_ratio", "gem_score",
        ]
        return df_sub.nlargest(top_n, "gem_score")[out_cols]

    # =========================================================================
    # 5. TIME-SERIES MARKET MOVERS & HISTORICAL TRAJECTORY
    # =========================================================================
    def get_market_movers(self, direction="gainers", league_filter=None, top_n=15):
        """
        Mendapatkan daftar pemain dengan kenaikan ('gainers') atau penurunan ('drops')
        nilai pasar terbesar berdasarkan perbandingan valuasi saat ini vs sebelumnya.
        """
        df_sub = self.df[self.df["market_value_est"] > 0].copy()
        if league_filter and league_filter != "Semua":
            if isinstance(league_filter, (list, tuple, set)):
                df_sub = df_sub[df_sub["league"].isin(league_filter)]
            else:
                df_sub = df_sub[df_sub["league"] == league_filter]

        if direction == "gainers":
            movers = df_sub[df_sub["mv_change_idr"] > 0].nlargest(top_n, "mv_change_idr").copy()
        else:
            movers = df_sub[df_sub["mv_change_idr"] < 0].nsmallest(top_n, "mv_change_idr").copy()

        movers["change_formatted"] = movers["mv_change_idr"].apply(
            lambda v: f"{'+' if v >= 0 else '-'}Rp{abs(v)/1e9:.2f}Mlyr."
            if abs(v) >= 1e9
            else f"{'+' if v >= 0 else '-'}Rp{abs(v)/1e6:.0f}Jt."
        )

        out_cols = [
            "player_name", "team", "league", "position", "age",
            "market_value_raw", "change_formatted", "mv_growth_pct", "mv_trend",
        ]
        return movers[out_cols]

    def get_player_value_history(self, player_name):
        """
        Membaca 25+ file snapshot historis di data/raw/real_scout_*.csv untuk membangun
        deret waktu (Time-Series) perkembangan nilai pasar seorang pemain.
        """
        records = []
        raw_files = sorted(glob.glob(os.path.join(self.raw_dir, "real_scout_*.csv")))

        for fpath in raw_files:
            try:
                fname = os.path.basename(fpath)
                m_date = re.search(r"(\d{8})", fname)
                snap_date = (
                    datetime.strptime(m_date.group(1), "%Y%m%d").strftime("%Y-%m-%d")
                    if m_date
                    else fname
                )
                df_snap = pd.read_csv(
                    fpath,
                    usecols=lambda c: c in ("player_name", "team", "market_value_est", "market_value_raw"),
                )
                match = df_snap[df_snap["player_name"] == player_name]
                if not match.empty:
                    row = match.iloc[0]
                    records.append({
                        "date": snap_date,
                        "team": row.get("team", "-"),
                        "market_value_est": int(row.get("market_value_est", 0)),
                        "market_value_raw": row.get("market_value_raw", "-"),
                    })
            except Exception:
                continue

        return pd.DataFrame(records)

    # =========================================================================
    # 6. CONTRACT EXPIRY & BOSMAN FREE-AGENT RADAR
    # =========================================================================
    def get_expiring_contracts(
        self,
        months_ahead=12,
        league_filter=None,
        position_filter=None,
        quota_filter=None,
        top_n=25,
    ):
        """
        Mencari pemain bernilai tinggi yang kontraknya akan habis dalam <= months_ahead bulan.
        """
        df_sub = self.df[
            (self.df["contract_months_left"] >= 0)
            & (self.df["contract_months_left"] <= months_ahead)
        ].copy()

        if league_filter and league_filter != "Semua":
            if isinstance(league_filter, (list, tuple, set)):
                df_sub = df_sub[df_sub["league"].isin(league_filter)]
            else:
                df_sub = df_sub[df_sub["league"] == league_filter]

        if position_filter and position_filter != "Semua":
            df_sub = df_sub[df_sub["position"] == position_filter]

        if quota_filter and quota_filter != "Semua":
            df_sub = df_sub[df_sub["quota_status"] == quota_filter]

        out_cols = [
            "player_name", "team", "league", "position", "age",
            "nationality", "quota_status", "contract_expiry",
            "contract_months_left", "market_value_raw", "market_value_est",
        ]
        return df_sub.nlargest(top_n, "market_value_est")[out_cols]

    # =========================================================================
    # 7. HEAD-TO-HEAD RADAR COMPARISON ENGINE
    # =========================================================================
    def compare_players(self, player_names):
        """
        Menghasilkan skor persentil 6 dimensi (0-100) untuk perbandingan Head-to-Head (Radar Chart).
        Dimensi:
          1. Market Valuation (Persentil nilai pasar)
          2. Age Prime & Upside (Investasi usia muda/emas)
          3. Physical Stature (Postur tinggi badan relatif posisi)
          4. Squad Standing (Peran bintang terhadap skuad klubnya)
          5. Market Momentum (Pertumbuhan nilai pasar)
          6. Value Efficiency (Efisiensi harga terhadap ekspektasi model)
        """
        df = self.df.copy()

        # Hitung persentil global/posisi
        mv_pct = df["market_value_est"].rank(pct=True) * 100.0
        age_score = (100.0 - ((df["age_clean"] - 21.0).clip(lower=0) * 4.5)).clip(20.0, 99.0)
        height_pct = df.groupby("position_group")["height_clean"].rank(pct=True) * 100.0
        standing_pct = df["squad_value_ratio"].rank(pct=True) * 100.0
        momentum_score = (50.0 + df["mv_growth_pct"].clip(-40.0, 50.0)).clip(10.0, 99.0)
        efficiency_pct = df["undervalued_ratio"].rank(pct=True) * 100.0

        df["_radar_valuation"] = mv_pct.round(1)
        df["_radar_age_upside"] = age_score.round(1)
        df["_radar_physical"] = height_pct.round(1)
        df["_radar_standing"] = standing_pct.round(1)
        df["_radar_momentum"] = momentum_score.round(1)
        df["_radar_efficiency"] = efficiency_pct.round(1)

        subset = df[df["player_name"].isin(player_names)].drop_duplicates(subset=["player_name"])
        radar_records = []
        for _, r in subset.iterrows():
            radar_records.append({
                "player_name": r["player_name"],
                "team": r["team"],
                "league": r["league"],
                "position": r["position"],
                "age": int(r["age"]),
                "height_cm": int(r["height_cm"]),
                "foot": r["foot"],
                "nationality": r["nationality"],
                "quota_status": r["quota_status"],
                "market_value_raw": r["market_value_raw"],
                "archetype": r["player_archetype"],
                "metrics": {
                    "Market Valuation": float(r["_radar_valuation"]),
                    "Age & Upside": float(r["_radar_age_upside"]),
                    "Physical Stature": float(r["_radar_physical"]),
                    "Squad Standing": float(r["_radar_standing"]),
                    "Market Momentum": float(r["_radar_momentum"]),
                    "Moneyball Value": float(r["_radar_efficiency"]),
                },
            })
        return radar_records
