#!/usr/bin/env python3
"""
Traitement des donnees de competitions en datasets ML
Transforme la structure data/competitions en datasets prets pour l'entrainement
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

class CompetitionDataProcessor:
    def __init__(self):
        self.base_path = Path("data/competitions")
        self.output_path = Path("data/ml_ready")
        
        # Competition mapping
        self.competitions = {
            'ligue_1_61': 'Ligue 1',
            'premier_league_39': 'Premier League', 
            'la_liga_140': 'La Liga',
            'bundesliga_78': 'Bundesliga',
            'champions_league_2': 'Champions League',
            'europa_league_3': 'Europa League'
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def extract_match_features(self, fixture_data, stats_data=None, events_data=None, lineups_data=None):
        """Extraire features d'un match"""
        try:
            fixture = fixture_data['fixture']
            teams = fixture_data['teams']
            goals = fixture_data['goals']
            
            # Features basiques
            features = {
                'fixture_id': fixture['id'],
                'date': fixture['date'],
                'home_team_id': teams['home']['id'],
                'away_team_id': teams['away']['id'],
                'home_team_name': teams['home']['name'],
                'away_team_name': teams['away']['name'],
                'home_goals': goals['home'] if goals['home'] is not None else 0,
                'away_goals': goals['away'] if goals['away'] is not None else 0,
                'home_win': 1 if goals['home'] and goals['away'] and goals['home'] > goals['away'] else 0,
                'draw': 1 if goals['home'] and goals['away'] and goals['home'] == goals['away'] else 0,
                'away_win': 1 if goals['home'] and goals['away'] and goals['away'] > goals['home'] else 0,
            }
            
            # Features des statistiques
            if stats_data and 'response' in stats_data:
                for team_stats in stats_data['response']:
                    team_prefix = 'home' if team_stats['team']['id'] == features['home_team_id'] else 'away'
                    
                    for stat in team_stats.get('statistics', []):
                        stat_name = stat['type'].lower().replace(' ', '_')
                        value = stat['value']
                        
                        # Convertir les pourcentages
                        if isinstance(value, str) and '%' in value:
                            try:
                                value = float(value.replace('%', ''))
                            except:
                                value = 0
                        elif value is None:
                            value = 0
                        
                        features[f'{team_prefix}_{stat_name}'] = value
            
            # Features des events
            if events_data and 'response' in events_data:
                home_cards = away_cards = 0
                home_subs = away_subs = 0
                
                for event in events_data['response']:
                    if event['team']['id'] == features['home_team_id']:
                        if event['type'] in ['Card']:
                            home_cards += 1
                        elif event['type'] in ['subst']:
                            home_subs += 1
                    else:
                        if event['type'] in ['Card']:
                            away_cards += 1
                        elif event['type'] in ['subst']:
                            away_subs += 1
                
                features.update({
                    'home_cards': home_cards,
                    'away_cards': away_cards,
                    'home_substitutions': home_subs,
                    'away_substitutions': away_subs
                })
            
            return features
            
        except Exception as e:
            self.logger.error(f"Erreur extraction features: {e}")
            return None

    def process_season_data(self, comp_dir, season):
        """Traiter une saison d'une competition"""
        season_path = comp_dir / f"seasons/{season}"
        
        if not season_path.exists():
            return []
        
        # Charger fixtures
        fixtures_file = season_path / "fixtures/fixtures.json"
        if not fixtures_file.exists():
            return []
            
        with open(fixtures_file, 'r', encoding='utf-8') as f:
            fixtures_data = json.load(f)
        
        matches = []
        fixtures = fixtures_data.get('response', [])
        
        self.logger.info(f"Traitement {len(fixtures)} matchs pour saison {season}")
        
        for fixture in fixtures:
            fixture_id = fixture['fixture']['id']
            
            # Charger donnees supplementaires
            stats_data = self.load_optional_data(season_path / f"statistics/fixture_{fixture_id}_stats.json")
            events_data = self.load_optional_data(season_path / f"events/fixture_{fixture_id}_events.json")
            lineups_data = self.load_optional_data(season_path / f"lineups/fixture_{fixture_id}_lineups.json")
            
            # Extraire features
            match_features = self.extract_match_features(
                fixture, stats_data, events_data, lineups_data
            )
            
            if match_features:
                matches.append(match_features)
        
        return matches

    def load_optional_data(self, file_path):
        """Charger donnees optionnelles"""
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Erreur lecture {file_path}: {e}")
        return None

    def process_competition(self, comp_name):
        """Traiter une competition complete"""
        comp_dir = self.base_path / comp_name
        
        if not comp_dir.exists():
            self.logger.warning(f"Competition {comp_name} n'existe pas")
            return None
        
        self.logger.info(f"Traitement competition: {comp_name}")
        
        all_matches = []
        seasons = [2021, 2022, 2023, 2024, 2025]
        
        for season in seasons:
            season_matches = self.process_season_data(comp_dir, season)
            all_matches.extend(season_matches)
        
        if not all_matches:
            self.logger.warning(f"Aucun match trouve pour {comp_name}")
            return None
        
        # Convertir en DataFrame
        df = pd.DataFrame(all_matches)
        
        # Ajouter informations competition
        df['competition'] = self.competitions.get(comp_name, comp_name)
        df['league_id'] = comp_name.split('_')[-1] if '_' in comp_name else comp_name
        
        self.logger.info(f"Dataset {comp_name}: {len(df)} matchs, {len(df.columns)} features")
        
        return df

    def process_all_competitions(self):
        """Traiter toutes les competitions"""
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        all_datasets = {}
        
        for comp_name in self.competitions.keys():
            df = self.process_competition(comp_name)
            if df is not None:
                all_datasets[comp_name] = df
                
                # Sauvegarder dataset individuel
                output_file = self.output_path / f"{comp_name}_ml_dataset.csv"
                df.to_csv(output_file, index=False, encoding='utf-8')
                self.logger.info(f"Dataset sauve: {output_file}")
        
        # Creer dataset combine
        if all_datasets:
            combined_df = pd.concat(all_datasets.values(), ignore_index=True)
            combined_file = self.output_path / "combined_ml_dataset.csv"
            combined_df.to_csv(combined_file, index=False, encoding='utf-8')
            self.logger.info(f"Dataset combine sauve: {combined_file} ({len(combined_df)} matchs)")
            
            return combined_df
        
        return None

def main():
    print("=" * 60)
    print("TRAITEMENT DONNEES COMPETITIONS EN DATASETS ML")
    print("=" * 60)
    
    processor = CompetitionDataProcessor()
    
    try:
        combined_df = processor.process_all_competitions()
        
        if combined_df is not None:
            print(f"\nTRAITEMENT REUSSI!")
            print(f"Total matchs: {len(combined_df)}")
            print(f"Features: {len(combined_df.columns)}")
            print(f"Competitions: {combined_df['competition'].nunique()}")
        else:
            print("\nERREUR: Aucun dataset genere")
            
    except Exception as e:
        print(f"\nERREUR TRAITEMENT: {e}")

if __name__ == "__main__":
    main()