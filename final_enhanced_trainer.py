#!/usr/bin/env python3
"""
Entraîneur Final Modèles Enrichis
Version finale avec targets correctes
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
import warnings
warnings.filterwarnings('ignore')

def train_final_enhanced_models():
    """Entraîner modèles enrichis version finale"""

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("=== ENTRAINEMENT FINAL MODELES ENRICHIS ===")

    # Chemins
    models_path = Path("models/enhanced_models")
    models_path.mkdir(parents=True, exist_ok=True)

    # Charger dataset avec targets
    dataset_file = 'data/ultra_processed/enhanced_ml_dataset_with_targets.csv'
    logger.info(f"Chargement: {dataset_file}")

    df = pd.read_csv(dataset_file)
    logger.info(f"Dataset: {len(df)} matchs, {len(df.columns)} colonnes")

    # Competitions
    competitions = {
        39: 'premier_league',
        140: 'la_liga',
        61: 'ligue_1',
        78: 'bundesliga',
        2: 'champions_league'
    }

    # Targets avec colonnes correctes
    target_mapping = {
        'goals_scored': 'total_goals',
        'both_teams_score': 'both_teams_score',
        'over_2_5_goals': 'over_2_5_goals',
        'next_match_result': 'result_home_win'
    }

    # Features à exclure
    exclude_features = [
        'fixture_id', 'date', 'home_team_id', 'away_team_id',
        'home_team_name', 'away_team_name', 'league_id',
        'home_goals', 'away_goals', 'home_win', 'draw', 'away_win',
        'total_goals', 'both_teams_score', 'over_2_5_goals',
        'result_home_win', 'result_draw', 'result_away_win'
    ]

    models_created = 0
    total_attempts = 0

    # Entraîner par ligue
    for league_id, league_name in competitions.items():
        logger.info(f"\n=== {league_name.upper()} (ID: {league_id}) ===")

        # Filtrer données de la ligue
        league_df = df[df['league_id'] == league_id].copy()

        if len(league_df) < 50:
            logger.warning(f"Pas assez de données: {len(league_df)} matchs")
            continue

        logger.info(f"Données: {len(league_df)} matchs")

        # Préparer features
        available_cols = [col for col in league_df.columns if col not in exclude_features]
        X = league_df[available_cols].select_dtypes(include=[np.number]).fillna(0)

        logger.info(f"Features: {len(X.columns)}")

        # Entraîner chaque target
        for target_name, target_col in target_mapping.items():
            total_attempts += 1

            if target_col not in league_df.columns:
                logger.warning(f"Target {target_col} manquante")
                continue

            try:
                y = league_df[target_col].fillna(0)

                # Vérifier variance
                if y.var() == 0:
                    logger.warning(f"Pas de variance pour {target_name}")
                    continue

                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42
                )

                # Tester plusieurs modèles
                models = {
                    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'gradient_boosting': GradientBoostingRegressor(random_state=42),
                    'ridge': Ridge(alpha=1.0)
                }

                best_model = None
                best_score = -np.inf
                best_name = ""

                for name, model in models.items():
                    try:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
                        cv_mean = cv_scores.mean()

                        if cv_mean > best_score:
                            best_score = cv_mean
                            best_model = model
                            best_name = name
                    except:
                        continue

                if best_model is None:
                    logger.warning(f"Aucun modèle valide pour {target_name}")
                    continue

                # Entraîner meilleur modèle
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)

                # Métriques
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                logger.info(f"  {target_name}: R²={r2:.3f} MAE={mae:.3f} ({best_name})")

                # Sauvegarder modèle
                model_file = models_path / f"enhanced_{league_id}_{target_name}.joblib"
                joblib.dump(best_model, model_file)

                # Scaler
                scaler = StandardScaler()
                scaler.fit(X_train)
                scaler_file = models_path / f"enhanced_scaler_{league_id}_{target_name}.joblib"
                joblib.dump(scaler, scaler_file)

                # Métriques JSON
                metrics = {
                    'r2': float(r2),
                    'mse': float(mse),
                    'mae': float(mae),
                    'cv_score': float(best_score),
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'features_count': len(X.columns),
                    'model_used': best_name,
                    'enhanced_features': True,
                    'training_date': datetime.now().isoformat(),
                    'league_name': league_name
                }

                metrics_file = models_path / f"enhanced_metrics_{league_id}_{target_name}.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)

                models_created += 1

            except Exception as e:
                logger.error(f"  Erreur {target_name}: {e}")
                continue

    logger.info(f"\n=== RÉSULTATS FINAUX ===")
    logger.info(f"Modèles créés: {models_created}/{total_attempts}")
    logger.info(f"Taux de succès: {models_created/total_attempts*100:.1f}%")

    return models_created

if __name__ == "__main__":
    print("="*70)
    print("ENTRAINEUR FINAL MODELES ENRICHIS")
    print("="*70)

    count = train_final_enhanced_models()

    if count > 0:
        print(f"\nSUCCÈS! {count} modèles enrichis créés")
        print("Modèles disponibles dans: models/enhanced_models/")
    else:
        print("\nECHEC: Aucun modèle créé")