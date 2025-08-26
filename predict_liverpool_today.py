"""
PREDICTEUR LIVERPOOL AUJOURD'HUI
Genere predictions completes pour le match de Liverpool du jour
"""

import requests
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import sys
import json

sys.path.append('src')
from config import API_FOOTBALL_CONFIG

class LiverpoolTodayPredictor:
    """Predicteur pour match Liverpool aujourd'hui"""
    
    def __init__(self):
        self.api_config = API_FOOTBALL_CONFIG
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {"X-RapidAPI-Key": self.api_config["api_key"]}
        
        # Charger dataset complet
        data_dir = Path("data/ultra_processed")
        dataset_files = list(data_dir.glob("complete_ml_dataset_*.csv"))
        
        if dataset_files:
            latest_file = max(dataset_files, key=lambda f: f.stat().st_mtime)
            self.df = pd.read_csv(latest_file)
            print(f"Dataset charge: {latest_file.name}")
        else:
            raise FileNotFoundError("Dataset ML non trouve")
        
        # Modeles Premier League
        self.models_dir = Path("models/complete_models")
        self.liverpool_id = 40
        self.premier_league_id = 39
    
    def find_liverpool_match_today(self):
        """Chercher match Liverpool aujourd'hui"""
        
        print("RECHERCHE MATCH LIVERPOOL AUJOURD'HUI")
        print("="*50)
        
        # Periode de recherche
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        tomorrow = today + timedelta(days=1)
        
        try:
            # Chercher fixtures Liverpool
            response = requests.get(
                f"{self.base_url}/fixtures",
                headers=self.headers,
                params={
                    "team": self.liverpool_id,
                    "season": 2025,
                    "from": yesterday.strftime("%Y-%m-%d"),
                    "to": tomorrow.strftime("%Y-%m-%d")
                }
            )
            
            if response.status_code == 200:
                fixtures_data = response.json()
                
                if fixtures_data.get('response'):
                    for fixture in fixtures_data['response']:
                        match_date = fixture['fixture']['date']
                        match_datetime = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                        
                        # Verifier si c'est aujourd'hui
                        if match_datetime.date() == today.date():
                            
                            home_team = fixture['teams']['home']
                            away_team = fixture['teams']['away']
                            
                            print(f"MATCH TROUVE:")
                            print(f"Date: {match_datetime.strftime('%Y-%m-%d %H:%M')}")
                            print(f"Domicile: {home_team['name']} (ID: {home_team['id']})")
                            print(f"Exterieur: {away_team['name']} (ID: {away_team['id']})")
                            print(f"Ligue: {fixture['league']['name']}")
                            print(f"Statut: {fixture['fixture']['status']['long']}")
                            
                            return {
                                'fixture_id': fixture['fixture']['id'],
                                'home_team_id': home_team['id'],
                                'away_team_id': away_team['id'],
                                'home_team_name': home_team['name'],
                                'away_team_name': away_team['name'],
                                'league_id': fixture['league']['id'],
                                'match_datetime': match_datetime,
                                'venue': fixture['fixture']['venue']['name'] if fixture['fixture']['venue'] else 'Unknown'
                            }
                    
                    print("Aucun match Liverpool trouve aujourd'hui")
                    return None
                else:
                    print("Aucune donnee de match retournee par l'API")
                    return None
            else:
                print(f"ERREUR API: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"ERREUR recherche match: {e}")
            return None
    
    def get_team_data(self, team_id):
        """Recuperer donnees equipe du dataset"""
        
        team_data = self.df[self.df['team_id'] == team_id]
        
        if len(team_data) > 0:
            return team_data.iloc[0]
        else:
            print(f"WARNING: Donnees equipe {team_id} non trouvees dans dataset")
            return None
    
    def load_prediction_models(self):
        """Charger modeles de prediction Premier League"""
        
        models = {}
        
        # Types de predictions disponibles
        prediction_types = [
            'next_match_result', 'goals_scored', 'goals_conceded',
            'clean_sheet', 'both_teams_score', 'over_2_5_goals',
            'win_probability', 'draw_probability', 'lose_probability'
        ]
        
        for pred_type in prediction_types:
            model_file = self.models_dir / f"complete_{self.premier_league_id}_{pred_type}.joblib"
            scaler_file = self.models_dir / f"complete_scaler_{self.premier_league_id}_{pred_type}.joblib"
            
            if model_file.exists() and scaler_file.exists():
                try:
                    model = joblib.load(model_file)
                    scaler = joblib.load(scaler_file)
                    models[pred_type] = {'model': model, 'scaler': scaler}
                except Exception as e:
                    print(f"Erreur chargement {pred_type}: {e}")
        
        print(f"Modeles charges: {len(models)}")
        return models
    
    def prepare_team_features(self, team_data):
        """Preparer features equipe pour prediction"""
        
        if team_data is None:
            return None
        
        # Features principales pour prediction
        feature_columns = [
            'points', 'played', 'wins', 'draws', 'losses', 'goals_for', 'goals_against',
            'goal_diff', 'win_rate', 'goals_per_match', 'home_wins', 'away_wins',
            'shots_total', 'ball_possession_avg', 'corners_taken', 'yellow_cards',
            'red_cards', 'passes_total', 'passes_accuracy_pct', 'top_scorer_goals',
            'squad_avg_age', 'form_trend', 'matches_clean_sheets'
        ]
        
        # Extraire features disponibles
        features = []
        feature_names = []
        
        for feature in feature_columns:
            if feature in team_data.index:
                value = team_data[feature]
                if pd.isna(value):
                    value = 0.0
                features.append(float(value))
                feature_names.append(feature)
        
        return np.array(features).reshape(1, -1), feature_names
    
    def generate_predictions(self, match_info):
        """Generer predictions completes pour le match"""
        
        print(f"\n=== PREDICTIONS {match_info['home_team_name']} vs {match_info['away_team_name']} ===")
        print(f"Date: {match_info['match_datetime'].strftime('%Y-%m-%d %H:%M')}")
        print(f"Venue: {match_info['venue']}")
        
        # Charger donnees equipes
        home_team_data = self.get_team_data(match_info['home_team_id'])
        away_team_data = self.get_team_data(match_info['away_team_id'])
        
        if home_team_data is None or away_team_data is None:
            print("ERREUR: Donnees equipes manquantes")
            return
        
        # Charger modeles
        models = self.load_prediction_models()
        
        if not models:
            print("ERREUR: Aucun modele disponible")
            return
        
        print(f"\nSTATISTIQUES EQUIPES:")
        print(f"{match_info['home_team_name']} (Domicile):")
        print(f"  Position: {int(home_team_data['position'])}e")
        print(f"  Points: {int(home_team_data['points'])}")
        print(f"  Matchs: {int(home_team_data['played'])}")
        print(f"  Buts pour: {int(home_team_data['goals_for'])}")
        print(f"  Buts contre: {int(home_team_data['goals_against'])}")
        print(f"  Forme: {home_team_data['win_rate']:.3f}")
        
        print(f"\n{match_info['away_team_name']} (Exterieur):")
        print(f"  Position: {int(away_team_data['position'])}e")
        print(f"  Points: {int(away_team_data['points'])}")
        print(f"  Matchs: {int(away_team_data['played'])}")
        print(f"  Buts pour: {int(away_team_data['goals_for'])}")
        print(f"  Buts contre: {int(away_team_data['goals_against'])}")
        print(f"  Forme: {away_team_data['win_rate']:.3f}")
        
        # Preparer features pour predictions
        home_features, home_feature_names = self.prepare_team_features(home_team_data)
        away_features, away_feature_names = self.prepare_team_features(away_team_data)
        
        if home_features is None or away_features is None:
            print("ERREUR: Preparation features echouee")
            return
        
        print(f"\nPREDICTIONS ML (Modeles R2 > 0.8):")
        print("-" * 60)
        
        # Generer predictions pour chaque type
        predictions_results = {}
        
        for pred_type, model_data in models.items():
            try:
                model = model_data['model']
                scaler = model_data['scaler']
                
                # Prediction equipe domicile
                home_features_scaled = scaler.transform(home_features)
                home_pred = model.predict(home_features_scaled)[0]
                
                # Prediction equipe exterieur
                away_features_scaled = scaler.transform(away_features)
                away_pred = model.predict(away_features_scaled)[0]
                
                predictions_results[pred_type] = {
                    'home': home_pred,
                    'away': away_pred
                }
                
                # Affichage selon le type de prediction
                if pred_type == 'next_match_result':
                    print(f"Resultat du match:")
                    print(f"  {match_info['home_team_name']}: {home_pred:.3f}")
                    print(f"  {match_info['away_team_name']}: {away_pred:.3f}")
                    if home_pred > away_pred:
                        print(f"  -> Avantage: {match_info['home_team_name']}")
                    else:
                        print(f"  -> Avantage: {match_info['away_team_name']}")
                
                elif pred_type == 'goals_scored':
                    print(f"Buts marques prevus:")
                    print(f"  {match_info['home_team_name']}: {home_pred:.1f}")
                    print(f"  {match_info['away_team_name']}: {away_pred:.1f}")
                    print(f"  -> Score prevu: {home_pred:.1f} - {away_pred:.1f}")
                
                elif pred_type in ['win_probability', 'draw_probability', 'lose_probability']:
                    prob_home = home_pred * 100
                    prob_away = away_pred * 100
                    print(f"{pred_type.replace('_', ' ').title()}:")
                    print(f"  {match_info['home_team_name']}: {prob_home:.1f}%")
                    print(f"  {match_info['away_team_name']}: {prob_away:.1f}%")
                
                elif pred_type == 'clean_sheet':
                    print(f"Clean sheet:")
                    print(f"  {match_info['home_team_name']}: {home_pred:.1%}")
                    print(f"  {match_info['away_team_name']}: {away_pred:.1%}")
                
                elif pred_type == 'both_teams_score':
                    bts_prob = (home_pred + away_pred) / 2
                    print(f"Both Teams Score: {bts_prob:.1%}")
                
                elif pred_type == 'over_2_5_goals':
                    over25_prob = (home_pred + away_pred) / 2
                    print(f"Over 2.5 Goals: {over25_prob:.1%}")
                
                print("")
                
            except Exception as e:
                print(f"Erreur prediction {pred_type}: {e}")
        
        # Resume final
        if 'goals_scored' in predictions_results:
            home_goals = predictions_results['goals_scored']['home']
            away_goals = predictions_results['goals_scored']['away']
            
            print("RESUME PREDICTIONS:")
            print(f"Score prevu: {match_info['home_team_name']} {home_goals:.1f} - {away_goals:.1f} {match_info['away_team_name']}")
            
            if home_goals > away_goals:
                print(f"Vainqueur prevu: {match_info['home_team_name']}")
            elif away_goals > home_goals:
                print(f"Vainqueur prevu: {match_info['away_team_name']}")
            else:
                print("Match nul prevu")

def main():
    """Fonction principale"""
    
    try:
        predictor = LiverpoolTodayPredictor()
        
        # Chercher match Liverpool aujourd'hui
        match_info = predictor.find_liverpool_match_today()
        
        if match_info:
            # Generer predictions
            predictor.generate_predictions(match_info)
        else:
            print("Aucun match Liverpool trouve aujourd'hui")
            print("Verification des prochains matchs...")
            
            # Alternative: chercher prochain match Liverpool
            tomorrow = datetime.now() + timedelta(days=1)
            week_later = datetime.now() + timedelta(days=7)
            
            response = requests.get(
                f"{predictor.base_url}/fixtures",
                headers=predictor.headers,
                params={
                    "team": predictor.liverpool_id,
                    "season": 2025,
                    "from": tomorrow.strftime("%Y-%m-%d"),
                    "to": week_later.strftime("%Y-%m-%d")
                }
            )
            
            if response.status_code == 200:
                fixtures_data = response.json()
                
                if fixtures_data.get('response'):
                    next_fixture = fixtures_data['response'][0]
                    next_date = datetime.fromisoformat(next_fixture['fixture']['date'].replace('Z', '+00:00'))
                    
                    print(f"Prochain match Liverpool:")
                    print(f"Date: {next_date.strftime('%Y-%m-%d %H:%M')}")
                    print(f"Adversaire: {next_fixture['teams']['home']['name']} vs {next_fixture['teams']['away']['name']}")
        
        print(f"\nPREDICTIONS TERMINEES!")
        
    except Exception as e:
        print(f"ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()