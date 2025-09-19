#!/usr/bin/env python3
"""
ENHANCED CHAMPIONS LEAGUE MODEL TRAINER
Système hybride utilisant les données de championnat pour améliorer les prédictions Champions League
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import pytz
import json

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

from config import Config

class EnhancedChampionsLeagueTrainer:
    """Entraîneur hybride pour modèles Champions League avec données cross-compétitions"""

    def __init__(self):
        self.paris_tz = pytz.timezone(Config.TIMEZONE_PARIS)

        # Chemins
        self.ml_data_dir = Path("data/ml_ready")
        self.models_dir = Path("models/complete_models")
        self.current_data_dir = Path("data/current_2025")

        # Créer dossiers
        for dir_path in [self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.model_types = [
            'next_match_result',
            'goals_scored',
            'both_teams_score',
            'win_probability',
            'over_2_5_goals'
        ]

        # Mapping des ligues nationales vers Champions League
        self.domestic_leagues = {
            39: 'premier_league',    # Premier League
            140: 'la_liga',         # La Liga
            135: 'serie_a',         # Serie A
            61: 'ligue_1',          # Ligue 1
            78: 'bundesliga'        # Bundesliga
        }

        self.setup_logging()

    def setup_logging(self):
        """Configuration logging"""
        log_file = Path("logs/enhanced_cl_trainer.log")
        log_file.parent.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_combined_dataset(self) -> pd.DataFrame:
        """Charger dataset combiné avec toutes les compétitions"""
        combined_file = self.ml_data_dir / "complete_combined_ml_dataset.csv"

        if not combined_file.exists():
            self.logger.warning("Dataset combiné introuvable, création en cours...")
            self.create_combined_dataset()

        df = pd.read_csv(combined_file)
        self.logger.info(f"Dataset combiné chargé: {len(df)} matchs, {df['league_id'].nunique()} compétitions")

        # Vérifier les ligues présentes
        league_counts = df['league_id'].value_counts()
        for league_id, count in league_counts.items():
            league_name = self.domestic_leagues.get(league_id, f"League_{league_id}")
            if league_id == 2:
                league_name = "Champions League"
            self.logger.info(f"  {league_name}: {count} matchs")

        return df

    def create_combined_dataset(self):
        """Créer dataset combiné si inexistant"""
        from create_complete_combined_dataset import create_complete_combined_dataset
        try:
            create_complete_combined_dataset()
            self.logger.info("Dataset combiné créé avec succès")
        except Exception as e:
            self.logger.error(f"Erreur création dataset combiné: {e}")
            raise

    def get_team_domestic_performance(self, df: pd.DataFrame, team_id: int,
                                    reference_date: Optional[str] = None) -> Dict:
        """Obtenir les performances récentes d'une équipe en championnat national"""

        # Données des championnats nationaux pour cette équipe (domicile + extérieur)
        domestic_data = df[
            ((df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)) &
            (df['league_id'].isin(self.domestic_leagues.keys()))
        ].copy()

        if len(domestic_data) == 0:
            return self.get_default_domestic_stats()

        # Si date de référence fournie, filtrer les matchs antérieurs
        if reference_date:
            domestic_data = domestic_data[
                pd.to_datetime(domestic_data['date']) < pd.to_datetime(reference_date)
            ]

        # Prendre les 10 derniers matchs pour la forme récente
        recent_matches = domestic_data.tail(10)

        if len(recent_matches) == 0:
            return self.get_default_domestic_stats()

        # Calculer les résultats et stats pour cette équipe spécifique
        team_results = []
        team_goals_for = []
        team_goals_against = []

        for _, match in domestic_data.iterrows():
            if match['home_team_id'] == team_id:
                # Équipe à domicile
                goals_for = match['home_goals']
                goals_against = match['away_goals']
                if goals_for > goals_against:
                    result = 'W'
                elif goals_for == goals_against:
                    result = 'D'
                else:
                    result = 'L'
            else:
                # Équipe à l'extérieur
                goals_for = match['away_goals']
                goals_against = match['home_goals']
                if goals_for > goals_against:
                    result = 'W'
                elif goals_for == goals_against:
                    result = 'D'
                else:
                    result = 'L'

            team_results.append(result)
            team_goals_for.append(goals_for)
            team_goals_against.append(goals_against)

        # Stats récentes (10 derniers)
        recent_results = team_results[-10:]
        recent_goals_for = team_goals_for[-10:]
        recent_goals_against = team_goals_against[-10:]

        # Calculer statistiques de forme
        stats = {
            # Forme récente (10 derniers matchs)
            'domestic_recent_wins': recent_results.count('W'),
            'domestic_recent_draws': recent_results.count('D'),
            'domestic_recent_losses': recent_results.count('L'),
            'domestic_recent_win_rate': recent_results.count('W') / len(recent_results) if recent_results else 0,

            # Performance offensive/défensive récente
            'domestic_recent_goals_for': np.mean(recent_goals_for) if recent_goals_for else 0,
            'domestic_recent_goals_against': np.mean(recent_goals_against) if recent_goals_against else 0,
            'domestic_recent_goal_diff': np.mean(recent_goals_for) - np.mean(recent_goals_against) if recent_goals_for and recent_goals_against else 0,

            # Forme sur l'ensemble de la saison
            'domestic_total_matches': len(domestic_data),
            'domestic_season_win_rate': team_results.count('W') / len(team_results) if team_results else 0,
            'domestic_season_goals_per_game': np.mean(team_goals_for) if team_goals_for else 0,
            'domestic_season_conceded_per_game': np.mean(team_goals_against) if team_goals_against else 0,

            # Ligue d'origine (coefficient de force)
            'domestic_league_strength': self.get_league_strength_coefficient(
                domestic_data['league_id'].iloc[0] if len(domestic_data) > 0 else 39
            ),

            # Consistency (écart-type des performances)
            'domestic_consistency': 1 / (np.std(team_goals_for) + 1) if team_goals_for else 0.5,
            'domestic_defensive_consistency': 1 / (np.std(team_goals_against) + 1) if team_goals_against else 0.5
        }

        return stats

    def get_default_domestic_stats(self) -> Dict:
        """Stats par défaut si aucune donnée domestique disponible"""
        return {
            'domestic_recent_wins': 3.0,
            'domestic_recent_draws': 3.0,
            'domestic_recent_losses': 4.0,
            'domestic_recent_win_rate': 0.3,
            'domestic_recent_goals_for': 1.2,
            'domestic_recent_goals_against': 1.3,
            'domestic_recent_goal_diff': -0.1,
            'domestic_total_matches': 10.0,
            'domestic_season_win_rate': 0.35,
            'domestic_season_goals_per_game': 1.2,
            'domestic_season_conceded_per_game': 1.3,
            'domestic_league_strength': 0.6,
            'domestic_consistency': 0.5,
            'domestic_defensive_consistency': 0.5
        }

    def get_league_strength_coefficient(self, league_id: int) -> float:
        """Coefficient de force de la ligue (basé sur coefficients UEFA approximatifs)"""
        coefficients = {
            39: 1.0,    # Premier League (référence)
            140: 0.95,  # La Liga
            135: 0.85,  # Serie A
            78: 0.80,   # Bundesliga
            61: 0.75,   # Ligue 1
            2: 1.2      # Champions League (plus fort)
        }
        return coefficients.get(league_id, 0.6)  # Défaut pour autres ligues

    def create_enhanced_cl_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Créer features enrichies pour Champions League avec données domestiques"""

        # Commencer avec données CL existantes
        cl_data = df[df['league_id'] == 2].copy()
        self.logger.info(f"Données CL de base: {len(cl_data)} matchs")

        if len(cl_data) == 0:
            raise ValueError("Aucune donnée Champions League trouvée")

        # Enrichir avec données domestiques pour chaque équipe
        enhanced_features = []

        for idx, row in cl_data.iterrows():
            # Features de base CL
            enhanced_row = row.to_dict()

            # Ajouter performances domestiques équipe domicile
            if pd.notna(row.get('home_team_id')):
                home_domestic_stats = self.get_team_domestic_performance(
                    df, int(row['home_team_id']), row.get('date')
                )
                for key, value in home_domestic_stats.items():
                    enhanced_row[f'home_{key}'] = value

            # Ajouter performances domestiques équipe extérieur
            if pd.notna(row.get('away_team_id')):
                away_domestic_stats = self.get_team_domestic_performance(
                    df, int(row['away_team_id']), row.get('date')
                )
                for key, value in away_domestic_stats.items():
                    enhanced_row[f'away_{key}'] = value

            # Features comparatives
            if 'home_domestic_recent_win_rate' in enhanced_row and 'away_domestic_recent_win_rate' in enhanced_row:
                enhanced_row['domestic_form_advantage'] = (
                    enhanced_row['home_domestic_recent_win_rate'] -
                    enhanced_row['away_domestic_recent_win_rate']
                )

                enhanced_row['domestic_attack_advantage'] = (
                    enhanced_row['home_domestic_recent_goals_for'] -
                    enhanced_row['away_domestic_recent_goals_for']
                )

                enhanced_row['domestic_defense_advantage'] = (
                    enhanced_row['away_domestic_recent_goals_against'] -
                    enhanced_row['home_domestic_recent_goals_against']
                )

            enhanced_features.append(enhanced_row)

        enhanced_df = pd.DataFrame(enhanced_features)
        self.logger.info(f"Features enrichies créées: {len(enhanced_df)} échantillons, {len(enhanced_df.columns)} colonnes")

        return enhanced_df

    def prepare_enhanced_training_data(self, df: pd.DataFrame, target_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Préparer données d'entraînement enrichies pour Champions League"""

        # Créer features enrichies
        enhanced_df = self.create_enhanced_cl_features(df)

        if len(enhanced_df) < 10:
            raise ValueError(f"Pas assez de données CL enrichies: {len(enhanced_df)} échantillons")

        # Définir target selon le type
        if target_type == 'next_match_result':
            enhanced_df['target'] = enhanced_df['home_win'].astype(float)
        elif target_type == 'goals_scored':
            enhanced_df['target'] = enhanced_df['home_goals'] + enhanced_df['away_goals']
        elif target_type == 'both_teams_score':
            enhanced_df['target'] = ((enhanced_df['home_goals'] > 0) & (enhanced_df['away_goals'] > 0)).astype(float)
        elif target_type == 'win_probability':
            enhanced_df['target'] = enhanced_df['home_win'].astype(float)
        elif target_type == 'over_2_5_goals':
            enhanced_df['target'] = ((enhanced_df['home_goals'] + enhanced_df['away_goals']) > 2.5).astype(float)

        # Sélectionner features numériques
        feature_columns = []
        for col in enhanced_df.columns:
            if col not in ['target', 'date', 'home_team', 'away_team', 'result'] and enhanced_df[col].dtype in ['int64', 'float64']:
                feature_columns.append(col)

        X = enhanced_df[feature_columns].fillna(0)
        y = enhanced_df['target'].fillna(0)

        # Sélection des meilleures features
        if len(feature_columns) > 50:
            selector = SelectKBest(score_func=f_regression, k=50)
            X_selected = selector.fit_transform(X, y)
            selected_features = np.array(feature_columns)[selector.get_support()]
            X = pd.DataFrame(X_selected, columns=selected_features)

        self.logger.info(f"Données d'entraînement préparées: {X.shape}, Target: {y.shape}")
        self.logger.info(f"Features principales: {list(X.columns[:10])}")

        return X.values, y.values

    def train_enhanced_cl_model(self, X: np.ndarray, y: np.ndarray, model_type: str) -> Tuple[object, object, Dict]:
        """Entraîner modèle CL enrichi"""

        # Split avec plus de données de test pour validation robuste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=Config.RANDOM_STATE
        )

        # Scaler robuste
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Modèle optimisé pour CL
        if model_type in ['next_match_result', 'win_probability', 'both_teams_score', 'over_2_5_goals']:
            model = GradientBoostingRegressor(
                n_estimators=150,  # Plus d'arbres pour complexité
                learning_rate=0.08, # Learning rate plus faible
                max_depth=8,       # Profondeur accrue
                subsample=0.8,     # Bagging pour robustesse
                random_state=Config.RANDOM_STATE
            )
        else:
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=3,
                random_state=Config.RANDOM_STATE
            )

        # Entraînement
        model.fit(X_train_scaled, y_train)

        # Évaluation
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'mse': mse,
            'r2': r2,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'training_date': datetime.now(self.paris_tz).isoformat(),
            'model_type': 'enhanced_champions_league',
            'features_count': X.shape[1]
        }

        return model, scaler, metrics

    def train_all_enhanced_cl_models(self):
        """Entraîner tous les modèles CL enrichis"""
        start_time = datetime.now(self.paris_tz)
        self.logger.info(f"=== ENTRAINEMENT MODELES CL ENRICHIS - {start_time.strftime('%d/%m/%Y %H:%M')} ===")

        try:
            # Charger dataset combiné
            df = self.load_combined_dataset()

            # Entraîner chaque type de modèle
            for model_type in self.model_types:
                try:
                    self.logger.info(f"Entraînement modèle CL enrichi: {model_type}")

                    # Préparer données enrichies
                    X, y = self.prepare_enhanced_training_data(df, model_type)

                    # Entraîner modèle
                    model, scaler, metrics = self.train_enhanced_cl_model(X, y, model_type)

                    # Sauvegarder avec suffixe "_enhanced"
                    model_file = self.models_dir / f"enhanced_cl_{model_type}.joblib"
                    scaler_file = self.models_dir / f"enhanced_cl_scaler_{model_type}.joblib"

                    joblib.dump(model, model_file)
                    joblib.dump(scaler, scaler_file)

                    # Sauvegarder métriques
                    metrics_file = self.models_dir / f"enhanced_cl_metrics_{model_type}.json"
                    with open(metrics_file, 'w') as f:
                        json.dump(metrics, f, indent=2)

                    self.logger.info(f"  SUCCES Modele sauve: R2 = {metrics['r2']:.4f}, MSE = {metrics['mse']:.6f}")
                    self.logger.info(f"     Features: {metrics['features_count']}, Train: {metrics['train_size']}, Test: {metrics['test_size']}")

                except Exception as e:
                    self.logger.error(f"  ERREUR {model_type}: {e}")
                    import traceback
                    traceback.print_exc()

            elapsed = datetime.now(self.paris_tz) - start_time
            self.logger.info(f"=== ENTRAINEMENT TERMINE - Durée: {elapsed} ===")

        except Exception as e:
            self.logger.error(f"Erreur générale: {e}")
            raise

if __name__ == "__main__":
    print("=" * 70)
    print("ENHANCED CHAMPIONS LEAGUE TRAINER - MODELES HYBRIDES")
    print("=" * 70)

    try:
        trainer = EnhancedChampionsLeagueTrainer()
        trainer.train_all_enhanced_cl_models()
        print("\nSUCCES! Modeles CL enrichis crees avec donnees cross-competitions")

    except Exception as e:
        print(f"\nERREUR: {e}")
        import traceback
        traceback.print_exc()