#!/usr/bin/env python3
"""
GOALS PREDICTION SPECIALIST TRAINER
Modèles dédiés pour prédire le nombre de buts avec features spécifiques par ligue
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import pytz
import json

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression

from config import Config

class GoalsSpecialistTrainer:
    """Entraîneur spécialisé pour prédiction de buts avec adaptation par ligue"""

    def __init__(self):
        self.paris_tz = pytz.timezone(Config.TIMEZONE_PARIS)

        # Chemins
        self.ml_data_dir = Path("data/ml_ready")
        self.models_dir = Path("models/complete_models")

        # Configuration leagues avec leurs caractéristiques
        self.league_profiles = {
            39: {  # Premier League
                'name': 'premier_league',
                'style': 'high_intensity',
                'avg_goals': 2.8,
                'tempo': 'fast',
                'physicality': 'high',
                'tactical': 'direct'
            },
            140: {  # La Liga
                'name': 'la_liga',
                'style': 'technical',
                'avg_goals': 2.6,
                'tempo': 'medium',
                'physicality': 'medium',
                'tactical': 'possession'
            },
            135: {  # Serie A
                'name': 'serie_a',
                'style': 'tactical',
                'avg_goals': 2.4,
                'tempo': 'slow',
                'physicality': 'medium',
                'tactical': 'defensive'
            },
            61: {  # Ligue 1
                'name': 'ligue_1',
                'style': 'physical',
                'avg_goals': 2.5,
                'tempo': 'medium',
                'physicality': 'high',
                'tactical': 'counter'
            },
            78: {  # Bundesliga
                'name': 'bundesliga',
                'style': 'attacking',
                'avg_goals': 3.0,
                'tempo': 'very_fast',
                'physicality': 'medium',
                'tactical': 'pressing'
            },
            2: {  # Champions League
                'name': 'champions_league',
                'style': 'elite',
                'avg_goals': 2.7,
                'tempo': 'variable',
                'physicality': 'high',
                'tactical': 'adaptive'
            }
        }

        self.setup_logging()

    def setup_logging(self):
        """Configuration logging"""
        log_file = Path("logs/goals_specialist.log")
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

    def create_league_specific_features(self, df: pd.DataFrame, league_id: int) -> pd.DataFrame:
        """Créer features spécifiques selon le profil de la ligue"""

        goals_df = df.copy()
        league_profile = self.league_profiles.get(league_id, self.league_profiles[39])

        # Target : total des buts
        goals_df['total_goals'] = goals_df['home_goals'] + goals_df['away_goals']

        # Features de base offensives
        goals_df['total_shots'] = goals_df['home_total_shots'] + goals_df['away_total_shots']
        goals_df['total_shots_on_target'] = goals_df['home_shots_on_goal'] + goals_df['away_shots_on_goal']
        goals_df['total_shots_inside'] = goals_df['home_shots_insidebox'] + goals_df['away_shots_insidebox']
        goals_df['total_corners'] = goals_df['home_corner_kicks'] + goals_df['away_corner_kicks']

        # Efficacité offensive
        goals_df['shot_accuracy'] = goals_df['total_shots_on_target'] / (goals_df['total_shots'] + 1)
        goals_df['inside_box_ratio'] = goals_df['total_shots_inside'] / (goals_df['total_shots'] + 1)

        # Features spécifiques selon le style de ligue
        if league_profile['style'] == 'high_intensity':  # Premier League
            # Focus sur intensité physique et transitions rapides
            goals_df['intensity_factor'] = (
                goals_df['home_fouls'] + goals_df['away_fouls'] +
                (goals_df['home_yellow_cards'] + goals_df['away_yellow_cards']) * 2
            )
            goals_df['transition_speed'] = goals_df['total_shots'] / (goals_df['home_total_passes'] + goals_df['away_total_passes'] + 1) * 1000
            goals_df['physical_game'] = goals_df['intensity_factor'] * goals_df['transition_speed']

        elif league_profile['style'] == 'technical':  # La Liga
            # Focus sur technique et possession
            goals_df['technical_quality'] = (
                (goals_df['home_passes_%'] + goals_df['away_passes_%']) / 2 +
                goals_df['shot_accuracy'] * 100
            ) / 2
            goals_df['possession_control'] = abs(goals_df['home_ball_possession'] - 50)
            goals_df['precision_play'] = goals_df['technical_quality'] * (50 - goals_df['possession_control'])

        elif league_profile['style'] == 'tactical':  # Serie A
            # Focus sur organisation défensive
            goals_df['defensive_solidity'] = 100 - (goals_df['total_shots_on_target'] / 2)
            goals_df['tactical_discipline'] = (goals_df['home_passes_%'] + goals_df['away_passes_%']) / 2
            goals_df['organized_attack'] = goals_df['total_corners'] + goals_df['total_shots_inside']

        elif league_profile['style'] == 'physical':  # Ligue 1
            # Focus sur physique et contre-attaques
            goals_df['physical_intensity'] = (
                goals_df['home_fouls'] + goals_df['away_fouls'] +
                abs(goals_df['home_ball_possession'] - goals_df['away_ball_possession'])
            )
            goals_df['counter_potential'] = goals_df['total_shots'] - goals_df['total_corners'] * 2

        elif league_profile['style'] == 'attacking':  # Bundesliga
            # Focus sur pressing et attaques rapides
            goals_df['pressing_intensity'] = goals_df['total_shots'] + goals_df['total_corners'] * 2
            goals_df['attacking_momentum'] = (
                goals_df['home_shots_insidebox'] + goals_df['away_shots_insidebox'] +
                goals_df['total_corners']
            )
            goals_df['pace_factor'] = goals_df['total_shots'] / 90  # Estimé par minute

        elif league_profile['style'] == 'elite':  # Champions League
            # Combinaison de tous les facteurs
            goals_df['elite_quality'] = (
                goals_df['shot_accuracy'] * 2 +
                goals_df['inside_box_ratio'] +
                (goals_df['home_passes_%'] + goals_df['away_passes_%']) / 200
            )

        # Features universelles mais pondérées selon la ligue
        tempo_multiplier = {
            'very_fast': 1.3, 'fast': 1.1, 'medium': 1.0, 'slow': 0.8, 'variable': 1.0
        }[league_profile['tempo']]

        goals_df['weighted_attack_power'] = (
            goals_df['total_shots_on_target'] * 0.4 +
            goals_df['total_shots_inside'] * 0.3 +
            goals_df['total_corners'] * 0.2 +
            goals_df['shot_accuracy'] * 0.1
        ) * tempo_multiplier

        # Expected Goals proxy adapté par ligue
        xg_weights = {
            39: {'shots_on': 0.25, 'inside': 0.35, 'corners': 0.15, 'accuracy': 0.25},  # Premier
            140: {'shots_on': 0.20, 'inside': 0.30, 'corners': 0.20, 'accuracy': 0.30},  # La Liga
            135: {'shots_on': 0.30, 'inside': 0.40, 'corners': 0.15, 'accuracy': 0.15},  # Serie A
            61: {'shots_on': 0.35, 'inside': 0.25, 'corners': 0.25, 'accuracy': 0.15},   # Ligue 1
            78: {'shots_on': 0.20, 'inside': 0.40, 'corners': 0.25, 'accuracy': 0.15},   # Bundesliga
            2: {'shots_on': 0.25, 'inside': 0.35, 'corners': 0.20, 'accuracy': 0.20}    # CL
        }

        weights = xg_weights.get(league_id, xg_weights[39])
        goals_df['league_xg'] = (
            goals_df['total_shots_on_target'] * weights['shots_on'] +
            goals_df['total_shots_inside'] * weights['inside'] +
            goals_df['total_corners'] * weights['corners'] +
            goals_df['shot_accuracy'] * 100 * weights['accuracy']
        )

        # Facteur de variance par ligue (certaines ligues plus prévisibles)
        variance_factors = {39: 1.2, 140: 0.9, 135: 0.8, 61: 1.1, 78: 1.3, 2: 1.0}
        goals_df['league_variance'] = variance_factors.get(league_id, 1.0)

        # Features d'équilibre du match
        goals_df['match_balance'] = 100 - abs(goals_df['home_ball_possession'] - 50)
        goals_df['shot_distribution'] = abs(goals_df['home_total_shots'] - goals_df['away_total_shots'])

        # Game state features
        goals_df['game_flow'] = (
            goals_df['total_shots'] + goals_df['total_corners'] +
            (goals_df['home_fouls'] + goals_df['away_fouls']) * 0.5
        )

        return goals_df

    def select_goals_features(self, df: pd.DataFrame, league_id: int) -> List[str]:
        """Sélectionner features optimales selon la ligue"""

        # Features de base
        base_features = [
            'total_shots', 'total_shots_on_target', 'total_shots_inside',
            'total_corners', 'shot_accuracy', 'inside_box_ratio',
            'weighted_attack_power', 'league_xg', 'match_balance',
            'shot_distribution', 'game_flow'
        ]

        # Features spécifiques par style de ligue
        league_profile = self.league_profiles.get(league_id, self.league_profiles[39])

        if league_profile['style'] == 'high_intensity':
            base_features.extend(['intensity_factor', 'transition_speed', 'physical_game'])
        elif league_profile['style'] == 'technical':
            base_features.extend(['technical_quality', 'possession_control', 'precision_play'])
        elif league_profile['style'] == 'tactical':
            base_features.extend(['defensive_solidity', 'tactical_discipline', 'organized_attack'])
        elif league_profile['style'] == 'physical':
            base_features.extend(['physical_intensity', 'counter_potential'])
        elif league_profile['style'] == 'attacking':
            base_features.extend(['pressing_intensity', 'attacking_momentum', 'pace_factor'])
        elif league_profile['style'] == 'elite':
            base_features.extend(['elite_quality'])

        # Features additionnelles communes
        common_features = [
            'home_shots_on_goal', 'away_shots_on_goal',
            'home_total_shots', 'away_total_shots',
            'home_ball_possession', 'away_ball_possession',
            'home_passes_%', 'away_passes_%'
        ]

        # Filtrer features existantes
        all_features = base_features + common_features
        available_features = [f for f in all_features if f in df.columns]

        self.logger.info(f"Features goals {league_profile['name']}: {len(available_features)}")

        return available_features

    def train_goals_specialist(self, X: np.ndarray, y: np.ndarray, league_id: int) -> Tuple[object, object, Dict]:
        """Entraîner modèle goals spécialisé pour une ligue"""

        league_profile = self.league_profiles[league_id]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=Config.RANDOM_STATE
        )

        # Scaling adapté selon la ligue
        if league_profile['style'] in ['technical', 'tactical']:
            scaler = StandardScaler()  # Plus stable pour ligues techniques
        else:
            scaler = RobustScaler()  # Résistant aux outliers pour ligues physiques

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Modèles adaptés selon le profil de ligue
        if league_profile['avg_goals'] > 2.8:  # Ligues à haut score
            models = {
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=300, learning_rate=0.08, max_depth=8,
                    subsample=0.8, random_state=Config.RANDOM_STATE
                ),
                'random_forest': RandomForestRegressor(
                    n_estimators=250, max_depth=12, min_samples_split=3,
                    random_state=Config.RANDOM_STATE
                )
            }
        elif league_profile['style'] == 'technical':  # Ligues techniques
            models = {
                'ridge': Ridge(alpha=1.0, random_state=Config.RANDOM_STATE),
                'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=Config.RANDOM_STATE),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=200, learning_rate=0.1, max_depth=6,
                    random_state=Config.RANDOM_STATE
                )
            }
        else:  # Ligues standard
            models = {
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=250, learning_rate=0.1, max_depth=7,
                    random_state=Config.RANDOM_STATE
                ),
                'random_forest': RandomForestRegressor(
                    n_estimators=200, max_depth=10,
                    random_state=Config.RANDOM_STATE
                ),
                'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
            }

        best_model = None
        best_score = -np.inf
        best_model_name = None

        # Cross-validation pour sélection
        for model_name, model in models.items():
            try:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                avg_cv_score = cv_scores.mean()

                self.logger.info(f"  {model_name}: CV R2 = {avg_cv_score:.3f}")

                if avg_cv_score > best_score:
                    best_score = avg_cv_score
                    best_model = model
                    best_model_name = model_name
            except Exception as e:
                self.logger.warning(f"  {model_name} failed: {e}")

        # Entraîner le meilleur
        best_model.fit(X_train_scaled, y_train)

        # Évaluation
        y_pred = best_model.predict(X_test_scaled)

        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'cv_score': best_score,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'avg_goals': y.mean(),
            'goals_std': y.std(),
            'model_used': best_model_name,
            'league_style': league_profile['style'],
            'training_date': datetime.now(self.paris_tz).isoformat()
        }

        return best_model, scaler, metrics

    def train_all_goals_specialists(self):
        """Entraîner tous les modèles goals spécialisés"""
        start_time = datetime.now(self.paris_tz)
        self.logger.info(f"=== ENTRAINEMENT GOALS SPECIALISTS - {start_time.strftime('%d/%m/%Y %H:%M')} ===")

        try:
            # Charger dataset
            combined_file = self.ml_data_dir / "complete_combined_ml_dataset.csv"
            df = pd.read_csv(combined_file)
            self.logger.info(f"Dataset chargé: {len(df)} matchs")

            # Entraîner pour chaque ligue
            for league_id, league_profile in self.league_profiles.items():
                try:
                    self.logger.info(f"Entraînement Goals spécialisé: {league_profile['name']} (ID: {league_id})")
                    self.logger.info(f"  Style: {league_profile['style']}, Avg goals: {league_profile['avg_goals']}")

                    # Filtrer par ligue
                    league_data = df[df['league_id'] == league_id].copy()

                    if len(league_data) < 30:
                        self.logger.warning(f"  Pas assez de données: {len(league_data)} matchs")
                        continue

                    # Créer features spécifiques
                    enhanced_data = self.create_league_specific_features(league_data, league_id)
                    feature_cols = self.select_goals_features(enhanced_data, league_id)

                    # Préparer données
                    X = enhanced_data[feature_cols].fillna(0).values
                    y = enhanced_data['total_goals'].values

                    # Entraîner
                    model, scaler, metrics = self.train_goals_specialist(X, y, league_id)

                    # Sauvegarder
                    model_file = self.models_dir / f"goals_specialist_{league_id}.joblib"
                    scaler_file = self.models_dir / f"goals_specialist_scaler_{league_id}.joblib"

                    joblib.dump(model, model_file)
                    joblib.dump(scaler, scaler_file)

                    # Métriques
                    metrics_file = self.models_dir / f"goals_specialist_metrics_{league_id}.json"
                    with open(metrics_file, 'w') as f:
                        json.dump(metrics, f, indent=2)

                    # Features
                    features_file = self.models_dir / f"goals_specialist_features_{league_id}.json"
                    with open(features_file, 'w') as f:
                        json.dump(feature_cols, f, indent=2)

                    self.logger.info(f"  SUCCES R2={metrics['r2']:.3f} MAE={metrics['mae']:.2f} Avg={metrics['avg_goals']:.1f}")
                    self.logger.info(f"  Modele: {metrics['model_used']}, Train: {metrics['train_size']}, Test: {metrics['test_size']}")

                except Exception as e:
                    self.logger.error(f"  ERREUR {league_profile['name']}: {e}")
                    import traceback
                    traceback.print_exc()

            elapsed = datetime.now(self.paris_tz) - start_time
            self.logger.info(f"=== ENTRAINEMENT GOALS TERMINE - Durée: {elapsed} ===")

        except Exception as e:
            self.logger.error(f"Erreur générale: {e}")
            raise

if __name__ == "__main__":
    print("=" * 70)
    print("GOALS SPECIALIST TRAINER - MODELES PREDICTION BUTS PAR LIGUE")
    print("=" * 70)

    try:
        trainer = GoalsSpecialistTrainer()
        trainer.train_all_goals_specialists()
        print("\nSUCCES! Modeles Goals specialises entraines")

    except Exception as e:
        print(f"\nERREUR: {e}")
        import traceback
        traceback.print_exc()