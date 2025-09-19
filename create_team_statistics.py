#!/usr/bin/env python3
"""
Création de statistiques d'équipes pour l'entraînement ML
Agrège les données de matchs individuels en statistiques par équipe
Compatible avec auto_model_trainer.py existant
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

class TeamStatisticsCreator:
    def __init__(self):
        self.ml_data_dir = Path("data/ml_ready")

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def aggregate_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrégation des statistiques par équipe"""
        team_stats_list = []

        # Traiter chaque équipe home et away
        for league_id in df['league_id'].unique():
            league_data = df[df['league_id'] == league_id].copy()

            # Obtenir toutes les équipes uniques de cette ligue
            home_teams = set(league_data['home_team_id'].unique())
            away_teams = set(league_data['away_team_id'].unique())
            all_teams = home_teams.union(away_teams)

            for team_id in all_teams:
                # Matchs à domicile
                home_matches = league_data[league_data['home_team_id'] == team_id].copy()
                # Matchs à l'extérieur
                away_matches = league_data[league_data['away_team_id'] == team_id].copy()

                if len(home_matches) == 0 and len(away_matches) == 0:
                    continue

                # Calcul des statistiques
                team_stats = self.calculate_team_statistics(
                    team_id, league_id, home_matches, away_matches, league_data
                )

                if team_stats:
                    team_stats_list.append(team_stats)

        return pd.DataFrame(team_stats_list)

    def calculate_team_statistics(self, team_id: int, league_id: int,
                                home_matches: pd.DataFrame, away_matches: pd.DataFrame,
                                league_data: pd.DataFrame) -> dict:
        """Calculer statistiques complètes d'une équipe"""

        # Informations de base
        team_name = ""
        if len(home_matches) > 0:
            team_name = home_matches.iloc[0]['home_team_name']
        elif len(away_matches) > 0:
            team_name = away_matches.iloc[0]['away_team_name']

        # Total matchs joués
        total_matches = len(home_matches) + len(away_matches)

        if total_matches == 0:
            return None

        # Buts marqués et encaissés
        goals_for = 0
        goals_against = 0
        wins = 0
        draws = 0
        losses = 0

        # Statistiques détaillées
        total_shots = 0
        total_shots_on_target = 0
        total_possession = 0
        total_passes = 0
        total_passes_accurate = 0
        total_fouls = 0
        total_cards = 0
        total_corners = 0

        # Matchs à domicile
        for _, match in home_matches.iterrows():
            goals_for += match.get('home_goals', 0)
            goals_against += match.get('away_goals', 0)

            if match.get('home_win', 0) == 1:
                wins += 1
            elif match.get('draw', 0) == 1:
                draws += 1
            else:
                losses += 1

            # Stats détaillées
            total_shots += match.get('home_total_shots', 0) or 0
            total_shots_on_target += match.get('home_shots_on_goal', 0) or 0
            total_possession += match.get('home_ball_possession', 0) or 0
            total_passes += match.get('home_total_passes', 0) or 0
            total_passes_accurate += match.get('home_passes_accurate', 0) or 0
            total_fouls += match.get('home_fouls', 0) or 0
            total_cards += match.get('home_cards', 0) or 0
            total_corners += match.get('home_corner_kicks', 0) or 0

        # Matchs à l'extérieur
        for _, match in away_matches.iterrows():
            goals_for += match.get('away_goals', 0)
            goals_against += match.get('home_goals', 0)

            if match.get('away_win', 0) == 1:
                wins += 1
            elif match.get('draw', 0) == 1:
                draws += 1
            else:
                losses += 1

            # Stats détaillées
            total_shots += match.get('away_total_shots', 0) or 0
            total_shots_on_target += match.get('away_shots_on_goal', 0) or 0
            total_possession += match.get('away_ball_possession', 0) or 0
            total_passes += match.get('away_total_passes', 0) or 0
            total_passes_accurate += match.get('away_passes_accurate', 0) or 0
            total_fouls += match.get('away_fouls', 0) or 0
            total_cards += match.get('away_cards', 0) or 0
            total_corners += match.get('away_corner_kicks', 0) or 0

        # Calculs des moyennes et ratios
        win_rate = wins / total_matches if total_matches > 0 else 0
        avg_goals_for = goals_for / total_matches if total_matches > 0 else 0
        avg_goals_against = goals_against / total_matches if total_matches > 0 else 0

        # Statistiques avancées
        avg_possession = total_possession / total_matches if total_matches > 0 else 0
        avg_shots_per_match = total_shots / total_matches if total_matches > 0 else 0
        shot_accuracy = total_shots_on_target / total_shots if total_shots > 0 else 0
        pass_accuracy = total_passes_accurate / total_passes if total_passes > 0 else 0

        # Créer le dictionnaire des statistiques (format attendu par auto_model_trainer)
        stats = {
            'team_id': team_id,
            'team_name': team_name,
            'league_id': league_id,
            'season': 2025,  # Saison actuelle
            'matches_played': total_matches,

            # Statistiques principales (attendues par auto_model_trainer)
            'win_rate': win_rate,
            'avg_goals_for': avg_goals_for,
            'avg_goals_against': avg_goals_against,

            # Statistiques détaillées (features pour ML)
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'points': wins * 3 + draws,
            'goal_difference': goals_for - goals_against,

            # Performance offensive
            'total_goals_scored': goals_for,
            'avg_shots_per_match': avg_shots_per_match,
            'shot_accuracy': shot_accuracy,
            'avg_corners_per_match': total_corners / total_matches if total_matches > 0 else 0,

            # Performance défensive
            'total_goals_conceded': goals_against,
            'clean_sheets': sum(1 for _, m in home_matches.iterrows() if m.get('away_goals', 0) == 0) +
                          sum(1 for _, m in away_matches.iterrows() if m.get('home_goals', 0) == 0),

            # Statistiques de jeu
            'avg_possession': avg_possession,
            'pass_accuracy': pass_accuracy,
            'avg_fouls_per_match': total_fouls / total_matches if total_matches > 0 else 0,
            'avg_cards_per_match': total_cards / total_matches if total_matches > 0 else 0,

            # Métadonnées
            'update_date': datetime.now().strftime('%Y-%m-%d'),
            'update_timestamp': datetime.now().isoformat()
        }

        # Assurer que toutes les valeurs numériques sont définies
        for key, value in stats.items():
            if isinstance(value, (int, float)) and (pd.isna(value) or np.isinf(value)):
                stats[key] = 0.0

        return stats

    def create_team_statistics_dataset(self, input_file: str = "complete_combined_ml_dataset.csv") -> pd.DataFrame:
        """Créer dataset de statistiques d'équipes à partir des matchs"""

        input_path = self.ml_data_dir / input_file

        if not input_path.exists():
            raise FileNotFoundError(f"Dataset de matchs non trouvé: {input_path}")

        self.logger.info(f"Chargement dataset matchs: {input_path}")
        matches_df = pd.read_csv(input_path)
        self.logger.info(f"Données chargées: {len(matches_df)} matchs")

        # Créer statistiques d'équipes
        self.logger.info("Création des statistiques d'équipes...")
        team_stats_df = self.aggregate_team_stats(matches_df)

        self.logger.info(f"Statistiques créées: {len(team_stats_df)} équipes")

        # Sauvegarder
        output_file = self.ml_data_dir / "team_statistics_ml_dataset.csv"
        team_stats_df.to_csv(output_file, index=False, encoding='utf-8')
        self.logger.info(f"Dataset équipes sauvé: {output_file}")

        return team_stats_df

def main():
    print("=" * 60)
    print("CREATION STATISTIQUES D'EQUIPES POUR ENTRAINEMENT ML")
    print("=" * 60)

    creator = TeamStatisticsCreator()

    try:
        team_stats_df = creator.create_team_statistics_dataset()

        print(f"\nCREATION REUSSIE!")
        print(f"Total équipes: {len(team_stats_df)}")
        print(f"Features: {len(team_stats_df.columns)}")
        print(f"Ligues: {team_stats_df['league_id'].nunique()}")

        # Afficher quelques stats
        print(f"\nÉquipes par ligue:")
        print(team_stats_df['league_id'].value_counts().sort_index())

    except Exception as e:
        print(f"\nERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()