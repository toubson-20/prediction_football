#!/usr/bin/env python3
"""
Préparateur de Targets pour Modèles Enrichis
Crée les colonnes cibles à partir des données existantes
"""

import pandas as pd
import numpy as np
import logging

def prepare_enhanced_targets():
    """Préparer targets pour entraînement enrichi"""

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=== PREPARATION TARGETS ENRICHIS ===")

    # Charger dataset enrichi
    df = pd.read_csv('data/ultra_processed/enhanced_ml_dataset_20250919_164429.csv')
    logger.info(f"Dataset original: {len(df)} matchs, {len(df.columns)} colonnes")

    # Créer colonnes targets
    logger.info("Création colonnes targets...")

    # 1. Total goals
    df['total_goals'] = df['home_goals'] + df['away_goals']

    # 2. Both teams score
    df['both_teams_score'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)

    # 3. Over 2.5 goals
    df['over_2_5_goals'] = (df['total_goals'] > 2.5).astype(int)

    # 4. Result home win
    df['result_home_win'] = df['home_win']

    # 5. Result draw
    df['result_draw'] = df['draw']

    # 6. Result away win
    df['result_away_win'] = df['away_win']

    logger.info("Targets créées:")
    logger.info(f"  total_goals: moyenne = {df['total_goals'].mean():.2f}")
    logger.info(f"  both_teams_score: taux = {df['both_teams_score'].mean():.2%}")
    logger.info(f"  over_2_5_goals: taux = {df['over_2_5_goals'].mean():.2%}")
    logger.info(f"  result_home_win: taux = {df['result_home_win'].mean():.2%}")

    # Sauvegarder dataset avec targets
    output_file = 'data/ultra_processed/enhanced_ml_dataset_with_targets.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"Dataset avec targets sauvé: {output_file}")
    logger.info(f"Nouvelles dimensions: {len(df)} matchs, {len(df.columns)} colonnes")

    return output_file

if __name__ == "__main__":
    print("="*60)
    print("PREPARATEUR TARGETS ENRICHIS")
    print("="*60)

    output_file = prepare_enhanced_targets()
    print(f"\nSUCCÈS! Dataset préparé: {output_file}")