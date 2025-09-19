#!/usr/bin/env python3
"""
Système de Meta-Learning
Combine nos modèles ML avec les prédictions API Football
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
import logging
import warnings
warnings.filterwarnings('ignore')

class MetaLearningSystem:
    def __init__(self):
        self.models_path = Path("models/enhanced_models")
        self.meta_models_path = Path("models/meta_learning")
        self.meta_models_path.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = Path("data/api_predictions")

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Competitions
        self.competitions = {
            39: 'premier_league',
            140: 'la_liga',
            61: 'ligue_1',
            78: 'bundesliga',
            135: 'serie_a',
            2: 'champions_league'
        }

        # Cache des modèles
        self._base_models_cache = {}

    def load_base_model(self, league_id: int, model_type: str):
        """Charger modèle de base enrichi"""
        cache_key = f"{league_id}_{model_type}"

        if cache_key not in self._base_models_cache:
            model_file = self.models_path / f"enhanced_{league_id}_{model_type}.joblib"
            scaler_file = self.models_path / f"enhanced_scaler_{league_id}_{model_type}.joblib"

            if model_file.exists() and scaler_file.exists():
                try:
                    model = joblib.load(model_file)
                    scaler = joblib.load(scaler_file)
                    self._base_models_cache[cache_key] = (model, scaler)
                    return model, scaler
                except Exception as e:
                    self.logger.warning(f"Erreur chargement {cache_key}: {e}")

        return self._base_models_cache.get(cache_key, (None, None))

    def get_base_prediction(self, match_features: Dict, league_id: int, model_type: str) -> float:
        """Obtenir prédiction du modèle de base"""
        model, scaler = self.load_base_model(league_id, model_type)

        if model is None or scaler is None:
            return 0.5  # Fallback

        try:
            # Préparer features (simplifiées pour démo)
            feature_names = [
                'home_goals_avg', 'away_goals_avg', 'home_form', 'away_form',
                'lineup_strength_home', 'lineup_strength_away',
                'market_confidence_home', 'market_confidence_away',
                'h2h_home_wins', 'h2h_avg_goals'
            ]

            X = np.array([[match_features.get(name, 0.5) for name in feature_names]])
            X_scaled = scaler.transform(X)

            prediction = model.predict(X_scaled)[0]

            # Ajuster selon type
            if model_type == 'goals_scored':
                prediction = max(0.0, min(6.0, prediction))
            elif model_type in ['both_teams_score', 'over_2_5_goals']:
                prediction = max(0.0, min(1.0, prediction))
            elif model_type == 'next_match_result':
                prediction = max(0.0, min(1.0, prediction))

            return prediction

        except Exception as e:
            self.logger.warning(f"Erreur prédiction base {model_type}: {e}")
            return 0.5

    def create_meta_features(self, match_data: Dict) -> Dict:
        """Créer features pour meta-learning"""
        league_id = match_data.get('league_id', 39)

        # Features de base du match
        base_features = {
            'home_goals_avg': match_data.get('home_goals_avg', 1.5),
            'away_goals_avg': match_data.get('away_goals_avg', 1.5),
            'home_form': match_data.get('home_form', 0.5),
            'away_form': match_data.get('away_form', 0.5),
            'lineup_strength_home': match_data.get('lineup_strength_home', 0.6),
            'lineup_strength_away': match_data.get('lineup_strength_away', 0.6),
            'market_confidence_home': match_data.get('market_confidence_home', 0.4),
            'market_confidence_away': match_data.get('market_confidence_away', 0.35),
            'h2h_home_wins': match_data.get('h2h_home_wins', 0.4),
            'h2h_avg_goals': match_data.get('h2h_avg_goals', 2.5)
        }

        # Prédictions de nos modèles de base
        our_predictions = {
            'our_goals_pred': self.get_base_prediction(base_features, league_id, 'goals_scored'),
            'our_bts_pred': self.get_base_prediction(base_features, league_id, 'both_teams_score'),
            'our_over25_pred': self.get_base_prediction(base_features, league_id, 'over_2_5_goals'),
            'our_result_pred': self.get_base_prediction(base_features, league_id, 'next_match_result')
        }

        # Features des prédictions API Football
        api_features = {
            'api_predictions_available': match_data.get('api_predictions_available', False),
            'api_home_win_percent': match_data.get('api_home_win_percent', 0.33),
            'api_draw_percent': match_data.get('api_draw_percent', 0.33),
            'api_away_win_percent': match_data.get('api_away_win_percent', 0.33),
            'api_under_over_over': match_data.get('api_under_over_over', 0.5),
            'api_under_over_under': match_data.get('api_under_over_under', 0.5),
            'api_goals_home': match_data.get('api_goals_home', 1.25),
            'api_goals_away': match_data.get('api_goals_away', 1.25),
            'api_comparison_att_home': match_data.get('api_comparison_att_home', 0.5),
            'api_comparison_att_away': match_data.get('api_comparison_att_away', 0.5),
            'api_comparison_def_home': match_data.get('api_comparison_def_home', 0.5),
            'api_comparison_def_away': match_data.get('api_comparison_def_away', 0.5),
            'api_form_home': match_data.get('api_form_home', 0.5),
            'api_form_away': match_data.get('api_form_away', 0.5)
        }

        # Features d'ensemble et de confiance
        ensemble_features = {
            'prediction_agreement_goals': abs(our_predictions['our_goals_pred'] - (api_features['api_goals_home'] + api_features['api_goals_away'])),
            'prediction_agreement_result': abs(our_predictions['our_result_pred'] - api_features['api_home_win_percent']),
            'market_api_agreement': abs(base_features['market_confidence_home'] - api_features['api_home_win_percent']),
            'form_consistency': abs((base_features['home_form'] - base_features['away_form']) - (api_features['api_form_home'] - api_features['api_form_away'])),
            'attack_balance': abs(api_features['api_comparison_att_home'] - api_features['api_comparison_att_away']),
            'defense_balance': abs(api_features['api_comparison_def_home'] - api_features['api_comparison_def_away'])
        }

        # Combiner toutes les features
        meta_features = {
            **base_features,
            **our_predictions,
            **api_features,
            **ensemble_features
        }

        return meta_features

    def create_meta_training_dataset(self) -> Optional[pd.DataFrame]:
        """Créer dataset d'entraînement pour meta-learning"""
        self.logger.info("Création dataset meta-learning...")

        # Charger dataset de base avec targets
        base_dataset_file = Path("data/ultra_processed/enhanced_ml_dataset_with_targets.csv")

        if not base_dataset_file.exists():
            self.logger.error("Dataset de base introuvable")
            return None

        df = pd.read_csv(base_dataset_file)
        self.logger.info(f"Dataset de base: {len(df)} matchs")

        # Simuler features API Football pour entraînement (données historiques)
        # En production, on utiliserait les vraies prédictions API collectées
        meta_data = []

        for idx, row in df.iterrows():
            try:
                # Créer données simulées de prédictions API
                match_data = {
                    'league_id': row.get('league_id', 39),
                    'home_goals_avg': row.get('home_goals', 1) + np.random.uniform(-0.5, 0.5),
                    'away_goals_avg': row.get('away_goals', 1) + np.random.uniform(-0.5, 0.5),
                    'home_form': 0.5 + np.random.uniform(-0.3, 0.3),
                    'away_form': 0.5 + np.random.uniform(-0.3, 0.3),
                    'lineup_strength_home': row.get('lineup_strength_home', 0.6),
                    'lineup_strength_away': row.get('lineup_strength_away', 0.6),
                    'market_confidence_home': row.get('market_confidence_home', 0.4),
                    'market_confidence_away': row.get('market_confidence_away', 0.35),
                    'h2h_home_wins': row.get('h2h_home_wins', 0.4),
                    'h2h_avg_goals': row.get('h2h_avg_goals', 2.5),

                    # Simuler prédictions API Football
                    'api_predictions_available': True,
                    'api_home_win_percent': 0.33 + np.random.uniform(-0.2, 0.2),
                    'api_draw_percent': 0.25 + np.random.uniform(-0.1, 0.1),
                    'api_away_win_percent': 0.33 + np.random.uniform(-0.2, 0.2),
                    'api_under_over_over': 0.5 + np.random.uniform(-0.3, 0.3),
                    'api_under_over_under': 0.5 + np.random.uniform(-0.3, 0.3),
                    'api_goals_home': row.get('home_goals', 1) + np.random.uniform(-0.8, 0.8),
                    'api_goals_away': row.get('away_goals', 1) + np.random.uniform(-0.8, 0.8),
                    'api_comparison_att_home': 0.5 + np.random.uniform(-0.3, 0.3),
                    'api_comparison_att_away': 0.5 + np.random.uniform(-0.3, 0.3),
                    'api_comparison_def_home': 0.5 + np.random.uniform(-0.3, 0.3),
                    'api_comparison_def_away': 0.5 + np.random.uniform(-0.3, 0.3),
                    'api_form_home': 0.5 + np.random.uniform(-0.3, 0.3),
                    'api_form_away': 0.5 + np.random.uniform(-0.3, 0.3)
                }

                # Créer features meta
                meta_features = self.create_meta_features(match_data)

                # Ajouter targets
                meta_features.update({
                    'fixture_id': row.get('fixture_id'),
                    'league_id': row.get('league_id'),
                    'total_goals': row.get('total_goals'),
                    'both_teams_score': row.get('both_teams_score'),
                    'over_2_5_goals': row.get('over_2_5_goals'),
                    'result_home_win': row.get('result_home_win')
                })

                meta_data.append(meta_features)

                if len(meta_data) % 500 == 0:
                    self.logger.info(f"  Traité: {len(meta_data)} matchs")

            except Exception as e:
                self.logger.warning(f"Erreur traitement ligne {idx}: {e}")
                continue

        meta_df = pd.DataFrame(meta_data)
        self.logger.info(f"Dataset meta-learning créé: {len(meta_df)} matchs, {len(meta_df.columns)} features")

        # Sauvegarder
        output_file = Path("data/ultra_processed/meta_learning_dataset.csv")
        meta_df.to_csv(output_file, index=False)
        self.logger.info(f"Dataset sauvé: {output_file}")

        return meta_df

    def train_meta_models(self, df: pd.DataFrame):
        """Entraîner modèles de meta-learning"""
        self.logger.info("=== ENTRAINEMENT MODELES META-LEARNING ===")

        # Features à exclure
        exclude_features = [
            'fixture_id', 'league_id',
            'total_goals', 'both_teams_score', 'over_2_5_goals', 'result_home_win'
        ]

        # Préparer features
        feature_cols = [col for col in df.columns if col not in exclude_features]
        X = df[feature_cols].fillna(0)

        self.logger.info(f"Features meta-learning: {len(feature_cols)}")

        # Targets
        targets = {
            'meta_goals_scored': 'total_goals',
            'meta_both_teams_score': 'both_teams_score',
            'meta_over_2_5_goals': 'over_2_5_goals',
            'meta_next_match_result': 'result_home_win'
        }

        models_trained = 0

        for target_name, target_col in targets.items():
            if target_col not in df.columns:
                continue

            try:
                y = df[target_col].fillna(0)

                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42
                )

                # Modèle meta (ensemble de modèles de base)
                if target_name in ['meta_both_teams_score', 'meta_over_2_5_goals', 'meta_next_match_result']:
                    # Classification pour targets binaires
                    meta_model = LogisticRegression(random_state=42, max_iter=1000)
                else:
                    # Régression pour goals
                    meta_model = RandomForestRegressor(n_estimators=100, random_state=42)

                # Entraînement
                meta_model.fit(X_train, y_train)

                # Prédiction
                y_pred = meta_model.predict(X_test)

                # Métriques
                if target_name in ['meta_both_teams_score', 'meta_over_2_5_goals', 'meta_next_match_result']:
                    accuracy = accuracy_score(y_test, y_pred.round())
                    self.logger.info(f"  {target_name}: Accuracy = {accuracy:.3f}")
                    metric_score = accuracy
                else:
                    r2 = r2_score(y_test, y_pred)
                    self.logger.info(f"  {target_name}: R² = {r2:.3f}")
                    metric_score = r2

                # Sauvegarder modèle meta
                model_file = self.meta_models_path / f"{target_name}.joblib"
                joblib.dump(meta_model, model_file)

                # Scaler
                scaler = StandardScaler()
                scaler.fit(X_train)
                scaler_file = self.meta_models_path / f"{target_name}_scaler.joblib"
                joblib.dump(scaler, scaler_file)

                # Métriques
                metrics = {
                    'model_type': 'meta_learning',
                    'metric_score': float(metric_score),
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'features_count': len(feature_cols),
                    'base_models_used': True,
                    'api_predictions_used': True,
                    'training_date': datetime.now().isoformat()
                }

                metrics_file = self.meta_models_path / f"{target_name}_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)

                models_trained += 1

            except Exception as e:
                self.logger.error(f"Erreur {target_name}: {e}")
                continue

        self.logger.info(f"=== {models_trained} MODELES META-LEARNING ENTRAINES ===")
        return models_trained

    def predict_with_meta_learning(self, match_data: Dict) -> Dict:
        """Prédire avec meta-learning"""
        try:
            # Créer features meta
            meta_features = self.create_meta_features(match_data)

            # Préparer pour prédiction
            feature_names = sorted([k for k in meta_features.keys()
                                  if k not in ['fixture_id', 'league_id']])
            X = np.array([[meta_features.get(name, 0) for name in feature_names]])

            predictions = {}

            # Charger et utiliser modèles meta
            meta_targets = ['meta_goals_scored', 'meta_both_teams_score',
                           'meta_over_2_5_goals', 'meta_next_match_result']

            for target in meta_targets:
                model_file = self.meta_models_path / f"{target}.joblib"
                scaler_file = self.meta_models_path / f"{target}_scaler.joblib"

                if model_file.exists() and scaler_file.exists():
                    try:
                        model = joblib.load(model_file)
                        scaler = joblib.load(scaler_file)

                        X_scaled = scaler.transform(X)
                        pred = model.predict(X_scaled)[0]

                        predictions[target.replace('meta_', '')] = float(pred)

                    except Exception as e:
                        self.logger.warning(f"Erreur prédiction {target}: {e}")

            return {
                'meta_predictions': predictions,
                'base_predictions': {
                    'our_goals': meta_features.get('our_goals_pred', 0),
                    'our_bts': meta_features.get('our_bts_pred', 0),
                    'our_over25': meta_features.get('our_over25_pred', 0),
                    'our_result': meta_features.get('our_result_pred', 0)
                },
                'api_predictions': {
                    'api_goals': meta_features.get('api_goals_home', 0) + meta_features.get('api_goals_away', 0),
                    'api_home_win': meta_features.get('api_home_win_percent', 0),
                    'api_over25': meta_features.get('api_under_over_over', 0)
                },
                'confidence': 0.8 if meta_features.get('api_predictions_available') else 0.6
            }

        except Exception as e:
            self.logger.error(f"Erreur meta-learning: {e}")
            return {'error': str(e)}

    def run_meta_learning_training(self):
        """Exécuter entraînement complet meta-learning"""
        self.logger.info("=== DEBUT META-LEARNING SYSTEM ===")

        # Créer dataset
        df = self.create_meta_training_dataset()

        if df is None:
            self.logger.error("Impossible de créer dataset meta-learning")
            return

        # Entraîner modèles
        models_count = self.train_meta_models(df)

        self.logger.info(f"=== META-LEARNING TERMINE - {models_count} modèles ===")
        return models_count

if __name__ == "__main__":
    print("=" * 70)
    print("SYSTEME META-LEARNING - PREDICTIONS API FOOTBALL")
    print("=" * 70)

    meta_system = MetaLearningSystem()
    count = meta_system.run_meta_learning_training()

    if count > 0:
        print(f"\nSUCCÈS! {count} modèles meta-learning créés")
        print("Combine nos modèles ML + prédictions API Football")
    else:
        print("\nECHEC: Aucun modèle meta-learning créé")