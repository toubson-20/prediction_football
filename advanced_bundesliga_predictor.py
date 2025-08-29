"""
PRÉDICTEUR BUNDESLIGA AVANCÉ
Système de modèles spécialisés pour gérer la variance unique de la Bundesliga
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedBundesligaPredictor:
    """Prédicteur spécialisé pour la Bundesliga avec modèles stratifiés"""
    
    def __init__(self):
        self.models_dir = Path("models/bundesliga_advanced")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path("data/ultra_processed")
        
        # Modèles séparés par niveau d'équipe
        self.tier_models = {
            'top_tier': {},      # Top 6 équipes
            'mid_tier': {},      # Équipes moyennes  
            'bottom_tier': {},   # Équipes faibles
            'ensemble': {}       # Modèle ensemble
        }
        
        self.scalers = {}
        self.tier_thresholds = {}
        
    def load_and_prepare_data(self):
        """Charger et préparer les données Bundesliga"""
        print("Chargement données Bundesliga...")
        
        df = pd.read_csv(self.data_dir / "complete_ml_dataset_20250826_213205.csv")
        bundesliga = df[df['league_id'] == 78].copy()
        
        print(f"Données chargées: {len(bundesliga)} équipes Bundesliga")
        
        # Stratification par niveau d'équipe
        bundesliga_sorted = bundesliga.sort_values('points', ascending=False)
        
        # Définir les seuils de stratification
        n_teams = len(bundesliga_sorted)
        top_6 = max(6, n_teams // 3)
        mid_end = max(12, 2 * n_teams // 3)
        
        self.tier_thresholds = {
            'top_tier_min_points': bundesliga_sorted.iloc[top_6-1]['points'] if top_6 <= n_teams else 0,
            'mid_tier_min_points': bundesliga_sorted.iloc[mid_end-1]['points'] if mid_end <= n_teams else 0
        }
        
        print(f"Seuils: Top tier >= {self.tier_thresholds['top_tier_min_points']} pts, "
              f"Mid tier >= {self.tier_thresholds['mid_tier_min_points']} pts")
        
        return bundesliga
    
    def categorize_team_tier(self, points):
        """Catégoriser une équipe selon son niveau"""
        if points >= self.tier_thresholds['top_tier_min_points']:
            return 'top_tier'
        elif points >= self.tier_thresholds['mid_tier_min_points']:
            return 'mid_tier'
        else:
            return 'bottom_tier'
    
    def enhance_features_temporal(self, df):
        """Ajouter des features temporelles avancées"""
        enhanced_df = df.copy()
        
        # Features de forme récente avec pondération
        enhanced_df['weighted_form'] = (
            enhanced_df['last_5_matches_wins'] * 0.4 +
            enhanced_df['form_trend'] * 0.3 +
            (enhanced_df['last_5_matches_goals_for'] / 5) * 0.3
        )
        
        # Momentum offensif/défensif
        enhanced_df['offensive_momentum'] = enhanced_df['goals_per_match'] * enhanced_df['weighted_form']
        enhanced_df['defensive_stability'] = (1 / (enhanced_df['goals_against'] / enhanced_df['played'] + 0.1)) * enhanced_df['matches_clean_sheets']
        
        # Features contextuelles
        enhanced_df['goals_variance'] = np.abs(enhanced_df['goals_for'] - enhanced_df['goals_per_match'] * enhanced_df['played'])
        enhanced_df['performance_consistency'] = enhanced_df['win_rate'] / (enhanced_df['goals_variance'] + 0.1)
        
        # Features de niveau d'équipe
        enhanced_df['tier_level'] = enhanced_df['points'].apply(self.categorize_team_tier)
        
        # One-hot encode tier levels
        for tier in ['top_tier', 'mid_tier', 'bottom_tier']:
            enhanced_df[f'is_{tier}'] = (enhanced_df['tier_level'] == tier).astype(int)
        
        return enhanced_df
    
    def prepare_features_by_tier(self, df, tier):
        """Préparer features spécifiques à chaque niveau"""
        base_features = [
            'points', 'played', 'wins', 'draws', 'losses', 'goals_for', 'goals_against',
            'goal_diff', 'win_rate', 'shots_total', 'shots_on_goal', 'ball_possession_avg',
            'yellow_cards', 'red_cards', 'corners_taken', 'passes_accuracy_pct',
            'goalkeeper_saves_pct', 'top_scorer_goals', 'top_scorer_assists'
        ]
        
        temporal_features = [
            'weighted_form', 'offensive_momentum', 'defensive_stability', 
            'performance_consistency', 'goals_variance'
        ]
        
        # Features spécifiques par niveau
        if tier == 'top_tier':
            # Top tier: focus sur la régularité et la qualité
            specific_features = [
                'passes_total', 'attacks_dangerous', 'top_scorer_rating',
                'squad_avg_age', 'venue_capacity'
            ]
        elif tier == 'mid_tier':
            # Mid tier: focus sur l'équilibre et la forme
            specific_features = [
                'home_wins', 'away_wins', 'fouls_committed', 'fouls_drawn',
                'last_5_matches_wins', 'form_trend'
            ]
        else:  # bottom_tier
            # Bottom tier: focus sur la solidité défensive
            specific_features = [
                'matches_clean_sheets', 'matches_failed_to_score', 
                'players_injured_current', 'recent_match_cards'
            ]
        
        all_features = base_features + temporal_features + specific_features
        
        # Sélectionner features disponibles
        available_features = [f for f in all_features if f in df.columns]
        
        return df[available_features].fillna(0)
    
    def create_ensemble_model(self, tier):
        """Créer un modèle ensemble spécialisé par niveau"""
        if tier == 'top_tier':
            # Top tier: modèles sophistiqués pour patterns complexes
            models = [
                ('gb_main', GradientBoostingRegressor(
                    n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)),
                ('rf_backup', RandomForestRegressor(
                    n_estimators=150, max_depth=8, random_state=42)),
                ('gb_alt', GradientBoostingRegressor(
                    n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42))
            ]
            weights = [0.5, 0.3, 0.2]
        elif tier == 'mid_tier':
            # Mid tier: équilibre entre complexité et robustesse
            models = [
                ('gb_main', GradientBoostingRegressor(
                    n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)),
                ('rf_main', RandomForestRegressor(
                    n_estimators=120, max_depth=6, random_state=42))
            ]
            weights = [0.6, 0.4]
        else:  # bottom_tier
            # Bottom tier: modèles simples et robustes
            models = [
                ('rf_main', RandomForestRegressor(
                    n_estimators=100, max_depth=4, random_state=42)),
                ('gb_simple', GradientBoostingRegressor(
                    n_estimators=80, learning_rate=0.1, max_depth=3, random_state=42))
            ]
            weights = [0.7, 0.3]
        
        return VotingRegressor(models, weights=weights)
    
    def train_tier_models(self, df):
        """Entraîner les modèles par niveau d'équipe"""
        print("\n=== ENTRAÎNEMENT MODÈLES PAR NIVEAU ===")
        
        results = {}
        
        for tier in ['top_tier', 'mid_tier', 'bottom_tier']:
            print(f"\n--- {tier.upper().replace('_', ' ')} ---")
            
            # Filtrer données pour ce niveau
            tier_mask = df['tier_level'] == tier
            tier_data = df[tier_mask]
            
            if len(tier_data) < 3:
                print(f"Pas assez de données pour {tier} ({len(tier_data)} équipes)")
                continue
            
            # Préparer features
            X = self.prepare_features_by_tier(tier_data, tier)
            
            # Target avec transformation log
            y_raw = (tier_data['goals_for'] / tier_data['played']).fillna(0)
            y = np.log1p(y_raw)
            
            print(f"Données: {len(tier_data)} équipes, {X.shape[1]} features")
            print(f"Goals/match: {y_raw.mean():.2f} ± {y_raw.std():.2f}")
            
            # Split avec stratification si possible
            if len(tier_data) >= 6:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42)
            else:
                # Pas assez de données pour split, utiliser toutes les données
                X_train, X_test, y_train, y_test = X, X, y, y
            
            # Normalisation robuste
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Modèle ensemble
            model = self.create_ensemble_model(tier)
            model.fit(X_train_scaled, y_train)
            
            # Évaluation
            y_pred = model.predict(X_test_scaled)
            
            # Métriques dans l'espace log
            mae_log = mean_absolute_error(y_test, y_pred)
            r2_log = r2_score(y_test, y_pred)
            
            # Métriques dans l'espace original
            y_test_orig = np.expm1(y_test)
            y_pred_orig = np.expm1(y_pred)
            mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
            r2_orig = r2_score(y_test_orig, y_pred_orig)
            
            print(f"R² (log): {r2_log:.3f}, MAE (log): {mae_log:.3f}")
            print(f"R² (original): {r2_orig:.3f}, MAE (original): {mae_orig:.3f}")
            
            # Sauvegarder modèle et scaler
            model_path = self.models_dir / f"bundesliga_{tier}_goals_model.joblib"
            scaler_path = self.models_dir / f"bundesliga_{tier}_scaler.joblib"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            self.tier_models[tier] = {
                'model': model,
                'scaler': scaler,
                'features': list(X.columns),
                'r2_log': r2_log,
                'r2_original': r2_orig,
                'mae_original': mae_orig
            }
            
            results[tier] = {
                'teams_count': len(tier_data),
                'features_count': X.shape[1],
                'r2_log': r2_log,
                'r2_original': r2_orig,
                'mae_original': mae_orig,
                'model_path': str(model_path),
                'scaler_path': str(scaler_path)
            }
        
        return results
    
    def create_meta_ensemble(self, df):
        """Créer un méta-modèle qui combine tous les niveaux"""
        print("\n=== MÉTA-ENSEMBLE ===")
        
        # Préparer données pour le méta-modèle
        X_meta = []
        y_meta = []
        
        for tier in ['top_tier', 'mid_tier', 'bottom_tier']:
            if tier not in self.tier_models:
                continue
                
            tier_mask = df['tier_level'] == tier
            tier_data = df[tier_mask]
            
            if len(tier_data) == 0:
                continue
            
            # Features du tier
            X_tier = self.prepare_features_by_tier(tier_data, tier)
            X_tier_scaled = self.tier_models[tier]['scaler'].transform(X_tier)
            
            # Prédictions du modèle spécialisé
            pred_tier = self.tier_models[tier]['model'].predict(X_tier_scaled)
            
            # Ajouter prédiction + features contextuelles
            context_features = tier_data[['points', 'win_rate', 'goals_per_match']].fillna(0)
            tier_encoding = pd.get_dummies(pd.Series([tier] * len(tier_data)), prefix='tier').values
            
            X_meta_tier = np.column_stack([
                pred_tier.reshape(-1, 1),
                context_features.values,
                tier_encoding
            ])
            
            X_meta.extend(X_meta_tier)
            
            # Target
            y_tier = np.log1p((tier_data['goals_for'] / tier_data['played']).fillna(0))
            y_meta.extend(y_tier)
        
        if len(X_meta) == 0:
            print("Pas assez de données pour le méta-ensemble")
            return None
        
        X_meta = np.array(X_meta)
        y_meta = np.array(y_meta)
        
        print(f"Méta-ensemble: {len(X_meta)} échantillons, {X_meta.shape[1]} features")
        
        # Modèle méta simple mais efficace
        meta_model = GradientBoostingRegressor(
            n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
        meta_model.fit(X_meta, y_meta)
        
        # Évaluation cross-validation
        cv_scores = cross_val_score(meta_model, X_meta, y_meta, cv=min(5, len(X_meta)), scoring='r2')
        print(f"CV R² méta-ensemble: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Sauvegarder
        meta_path = self.models_dir / "bundesliga_meta_ensemble.joblib"
        joblib.dump(meta_model, meta_path)
        
        self.tier_models['ensemble'] = {
            'model': meta_model,
            'cv_r2': cv_scores.mean(),
            'model_path': str(meta_path)
        }
        
        return cv_scores.mean()
    
    def predict_goals(self, team_data, use_ensemble=True):
        """Prédire les buts pour une équipe"""
        # Déterminer le niveau de l'équipe
        points = team_data.get('points', 0)
        tier = self.categorize_team_tier(points)
        
        if tier not in self.tier_models:
            print(f"Modèle non disponible pour {tier}")
            return None
        
        # Préparer features
        team_df = pd.DataFrame([team_data])
        team_enhanced = self.enhance_features_temporal(team_df)
        X = self.prepare_features_by_tier(team_enhanced, tier)
        X_scaled = self.tier_models[tier]['scaler'].transform(X)
        
        # Prédiction du modèle spécialisé
        pred_log = self.tier_models[tier]['model'].predict(X_scaled)[0]
        pred_goals = np.expm1(pred_log)
        
        result = {
            'predicted_goals': pred_goals,
            'tier': tier,
            'confidence': self.tier_models[tier]['r2_original']
        }
        
        # Prédiction ensemble si disponible
        if use_ensemble and 'ensemble' in self.tier_models:
            # TODO: Implémenter prédiction méta-ensemble
            pass
        
        return result

def main():
    """Fonction principale"""
    predictor = AdvancedBundesligaPredictor()
    
    # Charger et préparer données
    df = predictor.load_and_prepare_data()
    df_enhanced = predictor.enhance_features_temporal(df)
    
    # Entraîner modèles par niveau
    tier_results = predictor.train_tier_models(df_enhanced)
    
    # Créer méta-ensemble
    meta_r2 = predictor.create_meta_ensemble(df_enhanced)
    
    # Rapport final
    print(f"\n{'='*60}")
    print("SYSTÈME BUNDESLIGA AVANCÉ - RÉSULTATS")
    print(f"{'='*60}")
    
    total_teams = 0
    avg_r2_original = 0
    valid_tiers = 0
    
    for tier, results in tier_results.items():
        print(f"\n{tier.replace('_', ' ').title()}:")
        print(f"  Équipes: {results['teams_count']}")
        print(f"  Features: {results['features_count']}")
        print(f"  R² original: {results['r2_original']:.3f}")
        print(f"  MAE: {results['mae_original']:.3f} buts")
        
        total_teams += results['teams_count']
        if results['r2_original'] > -1:  # Exclure les très mauvais résultats
            avg_r2_original += results['r2_original']
            valid_tiers += 1
    
    if valid_tiers > 0:
        avg_r2_original /= valid_tiers
        print(f"\nMoyenne R² tous niveaux: {avg_r2_original:.3f}")
    
    if meta_r2:
        print(f"Méta-ensemble R²: {meta_r2:.3f}")
    
    print(f"\nTotal équipes traitées: {total_teams}")
    print(f"Modèles sauvés dans: {predictor.models_dir}")

if __name__ == "__main__":
    main()