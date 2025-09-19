#!/usr/bin/env python3
"""
Entraîneur Rapide Modèles Enrichis
Version simplifiée et robuste
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
import warnings
warnings.filterwarnings('ignore')

def train_enhanced_models():
    """Entraîner modèles enrichis rapidement"""

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=== ENTRAINEMENT RAPIDE MODELES ENRICHIS ===")

    # Chemins
    models_path = Path("models/enhanced_models")
    models_path.mkdir(parents=True, exist_ok=True)

    # Charger dataset enrichi
    ultra_dir = Path("data/ultra_processed")
    csv_files = list(ultra_dir.glob("enhanced_*.csv"))

    if not csv_files:
        logger.error("Aucun dataset enrichi trouvé")
        return

    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Chargement: {latest_file}")

    df = pd.read_csv(latest_file)
    logger.info(f"Dataset: {len(df)} matchs, {len(df.columns)} colonnes")

    # Competitions
    competitions = {
        39: 'premier_league',
        140: 'la_liga',
        61: 'ligue_1',
        78: 'bundesliga',
        2: 'champions_league'
    }

    # Targets
    targets = [
        'goals_scored',
        'both_teams_score',
        'over_2_5_goals',
        'next_match_result'
    ]

    target_columns = {
        'goals_scored': 'total_goals',
        'both_teams_score': 'both_teams_score',
        'over_2_5_goals': 'over_2_5_goals',
        'next_match_result': 'result_home_win'
    }

    # Features à exclure
    exclude_features = [
        'fixture_id', 'date', 'home_team', 'away_team',
        'league_id', 'home_team_id', 'away_team_id',
        'total_goals', 'both_teams_score', 'over_2_5_goals',
        'result_home_win', 'result_draw', 'result_away_win'
    ]

    total_models = 0
    successful_models = 0

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
        feature_cols = [col for col in league_df.columns if col not in exclude_features]
        X = league_df[feature_cols].fillna(0)

        logger.info(f"Features: {len(feature_cols)}")

        # Entraîner chaque target
        for target_name in targets:
            total_models += 1
            target_col = target_columns.get(target_name)

            if target_col not in league_df.columns:
                logger.warning(f"Target {target_col} manquante")
                continue

            try:
                y = league_df[target_col].fillna(0)

                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )

                # Modèle simple mais efficace
                model = RandomForestRegressor(
                    n_estimators=50,  # Réduit pour vitesse
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )

                # Entraînement
                model.fit(X_train, y_train)

                # Prédiction
                y_pred = model.predict(X_test)

                # Métriques
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                logger.info(f"  {target_name}: R²={r2:.3f} MAE={mae:.3f}")

                # Sauvegarder modèle
                model_file = models_path / f"enhanced_{league_id}_{target_name}.joblib"
                joblib.dump(model, model_file)

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
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'features_count': len(feature_cols),
                    'model_used': 'random_forest',
                    'enhanced_features': True,
                    'training_date': datetime.now().isoformat()
                }

                metrics_file = models_path / f"enhanced_metrics_{league_id}_{target_name}.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)

                successful_models += 1

            except Exception as e:
                logger.error(f"  Erreur {target_name}: {e}")
                continue

    logger.info(f"\n=== RÉSULTATS ===")
    logger.info(f"Modèles entraînés: {successful_models}/{total_models}")
    logger.info(f"Modèles sauvés dans: {models_path}")

    return successful_models

if __name__ == "__main__":
    print("="*60)
    print("ENTRAINEUR RAPIDE MODELES ENRICHIS")
    print("="*60)

    count = train_enhanced_models()

    if count > 0:
        print(f"\nSUCCÈS! {count} modèles enrichis créés")
    else:
        print("\nECHEC: Aucun modèle créé")