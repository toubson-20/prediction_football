#!/usr/bin/env python3
"""
BOTH TEAMS SCORE SPECIALIST TRAINER
Modèles dédiés pour prédire Both Teams Score avec features spécialisées
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

class BTSSpecialistTrainer:
    """Entraîneur spécialisé pour Both Teams Score avec features optimisées"""

    def __init__(self):
        self.paris_tz = pytz.timezone(Config.TIMEZONE_PARIS)

        # Chemins
        self.ml_data_dir = Path("data/ml_ready")
        self.models_dir = Path("models/complete_models")

        # Créer dossiers
        self.models_dir.mkdir(parents=True, exist_ok=True)

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
        log_file = Path("logs/bts_specialist.log")
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

    def create_bts_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Créer features spécialisées pour Both Teams Score"""

        # Copier le dataframe
        bts_df = df.copy()

        # Calculer BTS target (les deux équipes marquent)
        bts_df['bts_target'] = ((bts_df['home_goals'] > 0) & (bts_df['away_goals'] > 0)).astype(int)

        # Features offensives spécifiques
        bts_df['home_attack_strength'] = (
            bts_df['home_shots_on_goal'] +
            bts_df['home_shots_insidebox'] * 1.5 +
            bts_df['home_corner_kicks'] * 0.3
        )

        bts_df['away_attack_strength'] = (
            bts_df['away_shots_on_goal'] +
            bts_df['away_shots_insidebox'] * 1.5 +
            bts_df['away_corner_kicks'] * 0.3
        )

        # Features défensives
        bts_df['home_defense_weakness'] = (
            bts_df['away_shots_on_goal'] +
            bts_df['away_total_shots'] * 0.5 +
            bts_df['home_fouls'] * 0.2  # Plus de fautes = défense sous pression
        )

        bts_df['away_defense_weakness'] = (
            bts_df['home_shots_on_goal'] +
            bts_df['home_total_shots'] * 0.5 +
            bts_df['away_fouls'] * 0.2
        )

        # Ratios critiques pour BTS
        bts_df['attack_balance'] = (
            bts_df['home_attack_strength'] * bts_df['away_attack_strength']
        ) / (bts_df['home_attack_strength'] + bts_df['away_attack_strength'] + 1)

        bts_df['defense_weakness_combined'] = (
            bts_df['home_defense_weakness'] + bts_df['away_defense_weakness']
        )

        # Features de pressing/intensité
        bts_df['game_intensity'] = (
            bts_df['home_total_shots'] + bts_df['away_total_shots'] +
            bts_df['home_fouls'] + bts_df['away_fouls'] +
            bts_df['home_corner_kicks'] + bts_df['away_corner_kicks']
        )

        # Domination territoriale (possession équilibrée = plus de chances BTS)
        bts_df['possession_balance'] = 50 - abs(bts_df['home_ball_possession'] - 50)

        # Efficacité offensive
        bts_df['home_shot_accuracy'] = bts_df['home_shots_on_goal'] / (bts_df['home_total_shots'] + 1)
        bts_df['away_shot_accuracy'] = bts_df['away_shots_on_goal'] / (bts_df['away_total_shots'] + 1)

        # Features de rythme de jeu
        bts_df['total_passes'] = bts_df['home_total_passes'] + bts_df['away_total_passes']
        bts_df['pass_accuracy_avg'] = (bts_df['home_passes_%'] + bts_df['away_passes_%']) / 2

        # Features psychologiques (cartons = tension)
        bts_df['cards_tension'] = (
            bts_df['home_yellow_cards'] + bts_df['away_yellow_cards'] +
            (bts_df['home_red_cards'] + bts_df['away_red_cards']) * 3
        )

        # Équilibre du match (important pour BTS)
        bts_df['match_balance'] = abs(
            (bts_df['home_shots_on_goal'] + bts_df['home_corner_kicks']) -
            (bts_df['away_shots_on_goal'] + bts_df['away_corner_kicks'])
        )

        # Goal threat indicators
        bts_df['home_goal_threat'] = (
            bts_df['home_shots_on_goal'] * 0.4 +
            bts_df['home_shots_insidebox'] * 0.3 +
            bts_df['home_corner_kicks'] * 0.2 +
            (100 - bts_df['away_passes_%']) * 0.1  # Erreurs défensives
        )

        bts_df['away_goal_threat'] = (
            bts_df['away_shots_on_goal'] * 0.4 +
            bts_df['away_shots_insidebox'] * 0.3 +
            bts_df['away_corner_kicks'] * 0.2 +
            (100 - bts_df['home_passes_%']) * 0.1
        )

        # Minimum threat (les deux doivent avoir une menace minimum)
        bts_df['min_goal_threat'] = np.minimum(bts_df['home_goal_threat'], bts_df['away_goal_threat'])

        return bts_df

    def select_bts_features(self, df: pd.DataFrame) -> List[str]:
        """Sélectionner les meilleures features pour BTS"""

        # Features de base toujours incluses
        base_features = [
            'home_attack_strength', 'away_attack_strength',
            'home_defense_weakness', 'away_defense_weakness',
            'attack_balance', 'defense_weakness_combined',
            'game_intensity', 'possession_balance',
            'home_goal_threat', 'away_goal_threat', 'min_goal_threat'
        ]

        # Features candidates additionnelles
        candidate_features = [
            'home_shots_on_goal', 'away_shots_on_goal',
            'home_shots_insidebox', 'away_shots_insidebox',
            'home_total_shots', 'away_total_shots',
            'home_corner_kicks', 'away_corner_kicks',
            'home_shot_accuracy', 'away_shot_accuracy',
            'total_passes', 'pass_accuracy_avg',
            'cards_tension', 'match_balance',
            'home_ball_possession', 'away_ball_possession',
            'home_fouls', 'away_fouls'
        ]

        # Combiner et filtrer features existantes
        all_features = base_features + candidate_features
        available_features = [f for f in all_features if f in df.columns]

        self.logger.info(f"Features BTS sélectionnées: {len(available_features)}")

        return available_features

    def train_bts_specialist(self, X: np.ndarray, y: np.ndarray, league_name: str) -> Tuple[object, object, Dict]:
        """Entraîner modèle spécialisé BTS"""

        # Split avec stratification pour équilibrer les classes
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=Config.RANDOM_STATE, stratify=y
        )

        # Scaling adapté pour BTS
        scaler = StandardScaler()  # StandardScaler souvent meilleur pour classification
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Ensemble de modèles pour BTS
        models = {
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                random_state=Config.RANDOM_STATE
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=Config.RANDOM_STATE
            ),
            'logistic': LogisticRegression(
                C=1.0,
                penalty='l2',
                random_state=Config.RANDOM_STATE,
                max_iter=1000
            )
        }

        best_model = None
        best_score = 0
        best_model_name = None

        # Tester chaque modèle
        for model_name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
            avg_cv_score = cv_scores.mean()

            self.logger.info(f"  {model_name}: CV F1 = {avg_cv_score:.3f}")

            if avg_cv_score > best_score:
                best_score = avg_cv_score
                best_model = model
                best_model_name = model_name

        # Entraîner le meilleur modèle
        best_model.fit(X_train_scaled, y_train)

        # Évaluation complète
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
            'bts_rate': y.mean(),  # Taux de BTS dans les données
            'model_used': best_model_name,
            'training_date': datetime.now(self.paris_tz).isoformat()
        }

        return best_model, scaler, metrics

    def train_all_bts_specialists(self):
        """Entraîner tous les modèles BTS spécialisés"""
        start_time = datetime.now(self.paris_tz)
        self.logger.info(f"=== ENTRAINEMENT BTS SPECIALISTS - {start_time.strftime('%d/%m/%Y %H:%M')} ===")

        try:
            # Charger dataset combiné
            combined_file = self.ml_data_dir / "complete_combined_ml_dataset.csv"
            if not combined_file.exists():
                raise FileNotFoundError("Dataset combiné non trouvé")

            df = pd.read_csv(combined_file)
            self.logger.info(f"Dataset chargé: {len(df)} matchs")

            # Créer features BTS spécialisées
            bts_df = self.create_bts_features(df)
            self.logger.info(f"Features BTS créées: {len(bts_df.columns)} colonnes")

            # Sélectionner features optimales
            feature_cols = self.select_bts_features(bts_df)

            # Entraîner pour chaque ligue
            for league_id, league_name in self.leagues.items():
                try:
                    self.logger.info(f"Entraînement BTS spécialisé: {league_name} (ID: {league_id})")

                    # Filtrer par ligue
                    league_data = bts_df[bts_df['league_id'] == league_id].copy()

                    if len(league_data) < 20:
                        self.logger.warning(f"  Pas assez de données: {len(league_data)} matchs")
                        continue

                    # Préparer données
                    X = league_data[feature_cols].fillna(0).values
                    y = league_data['bts_target'].values

                    if len(np.unique(y)) < 2:
                        self.logger.warning(f"  Pas assez de variabilité dans les targets")
                        continue

                    # Entraîner modèle spécialisé
                    model, scaler, metrics = self.train_bts_specialist(X, y, league_name)

                    # Sauvegarder
                    model_file = self.models_dir / f"bts_specialist_{league_id}.joblib"
                    scaler_file = self.models_dir / f"bts_specialist_scaler_{league_id}.joblib"

                    joblib.dump(model, model_file)
                    joblib.dump(scaler, scaler_file)

                    # Sauvegarder métriques
                    metrics_file = self.models_dir / f"bts_specialist_metrics_{league_id}.json"
                    with open(metrics_file, 'w') as f:
                        json.dump(metrics, f, indent=2)

                    # Sauvegarder liste des features
                    features_file = self.models_dir / f"bts_specialist_features_{league_id}.json"
                    with open(features_file, 'w') as f:
                        json.dump(feature_cols, f, indent=2)

                    self.logger.info(f"  SUCCES F1={metrics['f1']:.3f} AUC={metrics['auc']:.3f} BTS_Rate={metrics['bts_rate']:.1%}")
                    self.logger.info(f"  Modele: {metrics['model_used']}, Train: {metrics['train_size']}, Test: {metrics['test_size']}")

                except Exception as e:
                    self.logger.error(f"  ERREUR {league_name}: {e}")
                    import traceback
                    traceback.print_exc()

            elapsed = datetime.now(self.paris_tz) - start_time
            self.logger.info(f"=== ENTRAINEMENT BTS TERMINE - Durée: {elapsed} ===")

        except Exception as e:
            self.logger.error(f"Erreur générale: {e}")
            raise

if __name__ == "__main__":
    print("=" * 70)
    print("BTS SPECIALIST TRAINER - MODELES BOTH TEAMS SCORE")
    print("=" * 70)

    try:
        trainer = BTSSpecialistTrainer()
        trainer.train_all_bts_specialists()
        print("\nSUCCES! Modeles BTS specialises entraines")

    except Exception as e:
        print(f"\nERREUR: {e}")
        import traceback
        traceback.print_exc()