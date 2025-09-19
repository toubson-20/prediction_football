#!/usr/bin/env python3
"""
OVER 2.5 GOALS SPECIALIST TRAINER
Modèles dédiés pour prédire Over 2.5 Goals avec features spécialisées
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

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif

from config import Config

class Over25SpecialistTrainer:
    """Entraîneur spécialisé pour Over 2.5 Goals avec features optimisées"""

    def __init__(self):
        self.paris_tz = pytz.timezone(Config.TIMEZONE_PARIS)

        # Chemins
        self.ml_data_dir = Path("data/ml_ready")
        self.models_dir = Path("models/complete_models")

        # Configuration
        self.leagues = {
            39: 'premier_league',
            140: 'la_liga',
            135: 'serie_a',
            61: 'ligue_1',
            78: 'bundesliga',
            2: 'champions_league'
        }

        self.setup_logging()

    def setup_logging(self):
        """Configuration logging"""
        log_file = Path("logs/over25_specialist.log")
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

    def create_over25_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Créer features spécialisées pour Over 2.5 Goals"""

        # Copier le dataframe
        o25_df = df.copy()

        # Calculer Over 2.5 target
        o25_df['over25_target'] = (o25_df['home_goals'] + o25_df['away_goals'] > 2.5).astype(int)

        # Features offensives combinées
        o25_df['total_attack_power'] = (
            o25_df['home_shots_on_goal'] + o25_df['away_shots_on_goal'] +
            (o25_df['home_shots_insidebox'] + o25_df['away_shots_insidebox']) * 1.2 +
            (o25_df['home_corner_kicks'] + o25_df['away_corner_kicks']) * 0.4
        )

        # Intensité offensive globale
        o25_df['total_shots'] = o25_df['home_total_shots'] + o25_df['away_total_shots']
        o25_df['total_shots_on_target'] = o25_df['home_shots_on_goal'] + o25_df['away_shots_on_goal']

        # Efficacité offensive
        o25_df['combined_shot_accuracy'] = o25_df['total_shots_on_target'] / (o25_df['total_shots'] + 1)

        # Pression défensive (plus de pression = plus d'erreurs = plus de buts)
        o25_df['defensive_pressure'] = (
            o25_df['home_fouls'] + o25_df['away_fouls'] +
            (o25_df['home_yellow_cards'] + o25_df['away_yellow_cards']) * 2 +
            (o25_df['home_red_cards'] + o25_df['away_red_cards']) * 5
        )

        # Rythme de jeu (plus de passes = jeu plus ouvert)
        o25_df['game_tempo'] = (
            o25_df['home_total_passes'] + o25_df['away_total_passes']
        ) / (o25_df['home_passes_%'] + o25_df['away_passes_%'] + 1)

        # Domination vs équilibre (équilibre = plus de buts souvent)
        o25_df['possession_difference'] = abs(o25_df['home_ball_possession'] - 50)
        o25_df['possession_balance'] = 50 - o25_df['possession_difference']  # Plus équilibré = plus de buts

        # Phases d'attaque (corners + tirs)
        o25_df['total_attack_phases'] = (
            o25_df['home_corner_kicks'] + o25_df['away_corner_kicks'] +
            o25_df['home_shots_insidebox'] + o25_df['away_shots_insidebox']
        )

        # Chances créées (basé sur passes clés estimées)
        o25_df['estimated_key_passes'] = (
            (o25_df['home_total_passes'] + o25_df['away_total_passes']) * 0.05 +  # ~5% des passes sont clés
            (o25_df['home_corner_kicks'] + o25_df['away_corner_kicks']) * 2  # Corners = chances
        )

        # Faiblesse défensive combinée
        o25_df['defensive_weakness'] = (
            o25_df['total_shots_on_target'] +
            (100 - o25_df['home_passes_%']) + (100 - o25_df['away_passes_%']) +  # Erreurs de passes
            o25_df['defensive_pressure'] * 0.3
        )

        # Multiplicateur d'attaque (les deux équipes attaquent)
        o25_df['dual_attack_threat'] = (
            np.sqrt(o25_df['home_shots_on_goal'] * o25_df['away_shots_on_goal']) +
            np.sqrt(o25_df['home_shots_insidebox'] * o25_df['away_shots_insidebox'])
        )

        # Instabilité du match (plus d'instabilité = plus de buts)
        o25_df['match_instability'] = (
            o25_df['defensive_pressure'] +
            abs(o25_df['home_total_shots'] - o25_df['away_total_shots']) +
            abs(o25_df['home_ball_possession'] - o25_df['away_ball_possession'])
        )

        # Facteur d'ouverture du jeu
        o25_df['game_openness'] = (
            o25_df['total_attack_phases'] +
            o25_df['possession_balance'] * 0.1 +
            o25_df['game_tempo'] * 0.001
        )

        # Expected goals proxy (basé sur qualité des tirs)
        o25_df['xg_proxy'] = (
            o25_df['home_shots_insidebox'] * 0.3 + o25_df['away_shots_insidebox'] * 0.3 +
            o25_df['home_shots_on_goal'] * 0.2 + o25_df['away_shots_on_goal'] * 0.2 +
            (o25_df['home_corner_kicks'] + o25_df['away_corner_kicks']) * 0.1
        )

        # Intensité par minute (hypothèse: matchs de 90 min)
        o25_df['intensity_per_minute'] = o25_df['total_attack_power'] / 90

        return o25_df

    def select_over25_features(self, df: pd.DataFrame) -> List[str]:
        """Sélectionner les meilleures features pour Over 2.5"""

        # Features spécialisées Over 2.5
        specialized_features = [
            'total_attack_power', 'total_shots', 'total_shots_on_target',
            'combined_shot_accuracy', 'defensive_pressure', 'game_tempo',
            'possession_balance', 'total_attack_phases', 'estimated_key_passes',
            'defensive_weakness', 'dual_attack_threat', 'match_instability',
            'game_openness', 'xg_proxy', 'intensity_per_minute'
        ]

        # Features de base importantes
        base_features = [
            'home_shots_on_goal', 'away_shots_on_goal',
            'home_total_shots', 'away_total_shots',
            'home_shots_insidebox', 'away_shots_insidebox',
            'home_corner_kicks', 'away_corner_kicks',
            'home_ball_possession', 'away_ball_possession',
            'home_total_passes', 'away_total_passes',
            'home_fouls', 'away_fouls'
        ]

        # Combiner et filtrer
        all_features = specialized_features + base_features
        available_features = [f for f in all_features if f in df.columns]

        self.logger.info(f"Features Over 2.5 sélectionnées: {len(available_features)}")

        return available_features

    def train_over25_specialist(self, X: np.ndarray, y: np.ndarray, league_name: str) -> Tuple[object, object, Dict]:
        """Entraîner modèle spécialisé Over 2.5"""

        # Split avec stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=Config.RANDOM_STATE, stratify=y
        )

        # Scaling
        scaler = RobustScaler()  # RobustScaler pour Over 2.5 (données peuvent être outliers)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Modèles optimisés pour Over 2.5
        models = {
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=250,  # Plus d'arbres pour complexité
                learning_rate=0.08,
                max_depth=7,
                subsample=0.85,
                random_state=Config.RANDOM_STATE
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=Config.RANDOM_STATE
            ),
            'logistic': LogisticRegression(
                C=0.5,  # Plus de régularisation
                penalty='l2',
                random_state=Config.RANDOM_STATE,
                max_iter=1000
            )
        }

        best_model = None
        best_score = 0
        best_model_name = None

        # Cross-validation pour sélection
        for model_name, model in models.items():
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
            avg_cv_score = cv_scores.mean()

            self.logger.info(f"  {model_name}: CV F1 = {avg_cv_score:.3f}")

            if avg_cv_score > best_score:
                best_score = avg_cv_score
                best_model = model
                best_model_name = model_name

        # Entraîner le meilleur
        best_model.fit(X_train_scaled, y_train)

        # Évaluation
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'cv_score': best_score,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'over25_rate': y.mean(),
            'model_used': best_model_name,
            'training_date': datetime.now(self.paris_tz).isoformat()
        }

        return best_model, scaler, metrics

    def train_all_over25_specialists(self):
        """Entraîner tous les modèles Over 2.5 spécialisés"""
        start_time = datetime.now(self.paris_tz)
        self.logger.info(f"=== ENTRAINEMENT OVER 2.5 SPECIALISTS - {start_time.strftime('%d/%m/%Y %H:%M')} ===")

        try:
            # Charger dataset
            combined_file = self.ml_data_dir / "complete_combined_ml_dataset.csv"
            df = pd.read_csv(combined_file)
            self.logger.info(f"Dataset chargé: {len(df)} matchs")

            # Créer features Over 2.5
            o25_df = self.create_over25_features(df)
            self.logger.info(f"Features Over 2.5 créées: {len(o25_df.columns)} colonnes")

            # Sélectionner features
            feature_cols = self.select_over25_features(o25_df)

            # Entraîner pour chaque ligue
            for league_id, league_name in self.leagues.items():
                try:
                    self.logger.info(f"Entraînement Over 2.5 spécialisé: {league_name} (ID: {league_id})")

                    # Filtrer par ligue
                    league_data = o25_df[o25_df['league_id'] == league_id].copy()

                    if len(league_data) < 20:
                        self.logger.warning(f"  Pas assez de données: {len(league_data)} matchs")
                        continue

                    # Préparer données
                    X = league_data[feature_cols].fillna(0).values
                    y = league_data['over25_target'].values

                    if len(np.unique(y)) < 2:
                        self.logger.warning(f"  Pas assez de variabilité dans les targets")
                        continue

                    # Entraîner
                    model, scaler, metrics = self.train_over25_specialist(X, y, league_name)

                    # Sauvegarder
                    model_file = self.models_dir / f"over25_specialist_{league_id}.joblib"
                    scaler_file = self.models_dir / f"over25_specialist_scaler_{league_id}.joblib"

                    joblib.dump(model, model_file)
                    joblib.dump(scaler, scaler_file)

                    # Métriques
                    metrics_file = self.models_dir / f"over25_specialist_metrics_{league_id}.json"
                    with open(metrics_file, 'w') as f:
                        json.dump(metrics, f, indent=2)

                    # Features
                    features_file = self.models_dir / f"over25_specialist_features_{league_id}.json"
                    with open(features_file, 'w') as f:
                        json.dump(feature_cols, f, indent=2)

                    self.logger.info(f"  SUCCES F1={metrics['f1']:.3f} AUC={metrics['auc']:.3f} Over25_Rate={metrics['over25_rate']:.1%}")
                    self.logger.info(f"  Modele: {metrics['model_used']}, Train: {metrics['train_size']}, Test: {metrics['test_size']}")

                except Exception as e:
                    self.logger.error(f"  ERREUR {league_name}: {e}")

            elapsed = datetime.now(self.paris_tz) - start_time
            self.logger.info(f"=== ENTRAINEMENT OVER 2.5 TERMINE - Durée: {elapsed} ===")

        except Exception as e:
            self.logger.error(f"Erreur générale: {e}")
            raise

if __name__ == "__main__":
    print("=" * 70)
    print("OVER 2.5 SPECIALIST TRAINER - MODELES OVER 2.5 GOALS")
    print("=" * 70)

    try:
        trainer = Over25SpecialistTrainer()
        trainer.train_all_over25_specialists()
        print("\nSUCCES! Modeles Over 2.5 specialises entraines")

    except Exception as e:
        print(f"\nERREUR: {e}")
        import traceback
        traceback.print_exc()