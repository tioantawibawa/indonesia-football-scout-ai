import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class ScoutBrain:
    def __init__(self, data_path='data/processed/master_player_db.csv'):
        # Load Data
        self.df = pd.read_csv(data_path)
        
        # Standardize Columns
        if 'league_country' in self.df.columns:
            self.df.rename(columns={'league_country': 'league'}, inplace=True)
            
        # Features for similarity (Age & Value)
        # Kita menggunakan Umur dan Harga Pasar sebagai "Profil" pemain
        self.features = ['age', 'market_value_est']
        self.df[self.features] = self.df[self.features].fillna(0)
        
        self.scaler = StandardScaler()
        
    def get_similar_players(self, player_name, top_n=10):
        """
        Mencari pemain yang mirip secara profil (Umur & Harga).
        Digunakan untuk mencari pengganti 'apple-to-apple'.
        """
        # Cek apakah pemain ada di database
        if player_name not in self.df['player_name'].values:
            return None
            
        # Ambil data pemain target
        target_player_data = self.df[self.df['player_name'] == player_name].iloc[0]
        target_pos = target_player_data['position']
        
        # Filter: Kandidat harus memiliki POSISI YANG SAMA
        df_pos = self.df[self.df['position'] == target_pos].reset_index(drop=True)
        
        # Prepare Data Matrix
        X = df_pos[self.features]
        X_scaled = self.scaler.fit_transform(X)
        
        # Cari index pemain target di dataframe yang sudah difilter
        target_idx = df_pos[df_pos['player_name'] == player_name].index[0]
        target_vector = X_scaled[target_idx].reshape(1, -1)
        
        # Hitung Similarity (Cosine)
        similarity_scores = cosine_similarity(target_vector, X_scaled)[0]
        
        # Sorting dari yang paling mirip
        # [1:top_n+1] artinya skip index 0 (karena index 0 adalah diri sendiri)
        similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
        
        result = df_pos.iloc[similar_indices].copy()
        result['similarity_score'] = (similarity_scores[similar_indices] * 100).round(1)
        
        return result[['player_name', 'team', 'league', 'position', 'age', 'market_value_raw', 'similarity_score']]

    def recommend_for_team_needs(self, team_name, target_position, top_n=5):
        """
        Mencari pemain untuk posisi tertentu yang sesuai dengan budget klub.
        (Fitur Tab Squad Planner)
        """
        team_players = self.df[self.df['team'] == team_name]
        
        if team_players.empty:
            return None
            
        avg_squad_value = team_players['market_value_est'].nlargest(15).mean()
        
        if pd.isna(avg_squad_value) or avg_squad_value == 0:
            avg_squad_value = 1000000000 # 1 Milyar default
            
        candidates = self.df[
            (self.df['position'] == target_position) & 
            (self.df['team'] != team_name)
        ].copy()
        
        if candidates.empty:
            return pd.DataFrame()

        min_budget = avg_squad_value * 0.3
        max_budget = avg_squad_value * 2.5 
        
        filtered_candidates = candidates[
            (candidates['market_value_est'] >= min_budget) & 
            (candidates['market_value_est'] <= max_budget)
        ].copy()
        
        safe_age = filtered_candidates['age'].replace(0, 25)
        filtered_candidates['scout_score'] = (filtered_candidates['market_value_est'] / safe_age)
        
        return filtered_candidates.nlargest(top_n, 'scout_score')[['player_name', 'team', 'league', 'age', 'market_value_raw']]
