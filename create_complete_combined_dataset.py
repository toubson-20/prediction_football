#!/usr/bin/env python3
"""
Créer un dataset combiné COMPLET incluant TOUTES les compétitions
Inclut Champions League, Ligue 1, et toutes les autres compétitions
"""

import pandas as pd
from pathlib import Path
import logging

def create_complete_combined_dataset():
    """Créer un dataset combiné incluant toutes les compétitions"""

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    ml_data_dir = Path("data/ml_ready")

    # Tous les fichiers de compétitions individuelles
    competition_files = [
        "premier_league_39_ml_dataset.csv",
        "la_liga_140_ml_dataset.csv",
        "bundesliga_78_ml_dataset.csv",
        "ligue_1_61_ml_dataset.csv",
        "champions_league_2_ml_dataset.csv"
    ]

    all_dataframes = []

    logger.info("Combinaison de toutes les compétitions...")

    for file_name in competition_files:
        file_path = ml_data_dir / file_name

        if file_path.exists():
            logger.info(f"Chargement: {file_name}")
            df = pd.read_csv(file_path)
            logger.info(f"  Matchs: {len(df)}")
            logger.info(f"  League ID: {df['league_id'].iloc[0] if len(df) > 0 else 'N/A'}")
            all_dataframes.append(df)
        else:
            logger.warning(f"Fichier non trouvé: {file_name}")

    if not all_dataframes:
        raise ValueError("Aucun fichier de compétition trouvé!")

    # Combiner tous les DataFrames
    logger.info("Fusion des datasets...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    logger.info(f"Dataset combiné créé:")
    logger.info(f"  Total matchs: {len(combined_df)}")
    logger.info(f"  Ligues: {combined_df['league_id'].nunique()}")
    logger.info(f"  Répartition par ligue:")

    league_counts = combined_df['league_id'].value_counts().sort_index()
    for league_id, count in league_counts.items():
        logger.info(f"    Ligue {league_id}: {count} matchs")

    # Sauvegarder
    output_file = ml_data_dir / "complete_combined_ml_dataset.csv"
    combined_df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Dataset complet sauvé: {output_file}")

    return combined_df

if __name__ == "__main__":
    print("=" * 60)
    print("CREATION DATASET COMBINE COMPLET - TOUTES COMPETITIONS")
    print("=" * 60)

    try:
        df = create_complete_combined_dataset()
        print(f"\nSUCCES!")
        print(f"Dataset combiné créé avec {len(df)} matchs")
        print(f"Incluant {df['league_id'].nunique()} compétitions")

    except Exception as e:
        print(f"\nERREUR: {e}")
        import traceback
        traceback.print_exc()