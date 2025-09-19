"""
SYSTEME DE PREDICTION FOOTBALL UNIFIE
Système central utilisant API Football avec fuseau horaire Paris et modèles ML existants
"""

import asyncio
import aiohttp
import pytz
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import sys
import os

# Configuration centralisée
try:
    from config import Config
except ImportError:
    print("Erreur: Impossible d'importer config.py")
    sys.exit(1)

class FootballPredictionSystem:
    """Système principal de prédiction football"""
    
    def __init__(self):
        # Configuration API Football
        self.api_key = Config.FOOTBALL_API_KEY
        self.base_url = Config.FOOTBALL_API_BASE_URL
        self.headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }
        
        # Timezone Paris
        self.paris_tz = pytz.timezone(Config.TIMEZONE_PARIS)
        
        # Ligues suivies
        self.leagues = Config.TARGET_LEAGUES
        
        # Modèles ML chargés
        self.models = {}
        self.scalers = {}
        
        # Cache des données
        self.matches_cache = {}
        self.teams_cache = {}
        
        # Charger les modèles existants
        self.load_existing_models()
    
    def load_existing_models(self):
        """Charger les modèles ML existants"""
        models_dir = Path("models/complete_models")
        
        if not models_dir.exists():
            print(f"ATTENTION: Dossier modèles {models_dir} inexistant")
            return
        
        # Modèles par ligue
        model_types = [
            'next_match_result', 'goals_scored', 'both_teams_score',
            'win_probability', 'over_2_5_goals'
        ]
        
        for league_id in [39, 140, 61, 78, 135]:  # Premier League, La Liga, Ligue 1, Bundesliga, Serie A
            self.models[league_id] = {}
            self.scalers[league_id] = {}
            
            for model_type in model_types:
                model_file = models_dir / f"complete_{league_id}_{model_type}.joblib"
                scaler_file = models_dir / f"complete_scaler_{league_id}_{model_type}.joblib"
                
                if model_file.exists():
                    try:
                        self.models[league_id][model_type] = joblib.load(model_file)
                        if scaler_file.exists():
                            self.scalers[league_id][model_type] = joblib.load(scaler_file)
                    except Exception as e:
                        print(f"Erreur chargement {model_file}: {e}")
        
        total_models = sum(len(league_models) for league_models in self.models.values())
        print(f"Modèles ML chargés: {total_models} modèles")
    
    def get_paris_time_now(self) -> datetime:
        """Heure actuelle Paris"""
        return datetime.now(self.paris_tz)
    
    def get_paris_date_string(self, date_offset: int = 0) -> str:
        """Date Paris au format API (YYYY-MM-DD)"""
        target_date = self.get_paris_time_now() + timedelta(days=date_offset)
        return target_date.strftime('%Y-%m-%d')
    
    async def find_matches_today(self) -> List[Dict]:
        """Trouver tous les matchs d'aujourd'hui dans nos ligues"""
        print(f"RECHERCHE MATCHS AUJOURD'HUI - {self.get_paris_date_string()}")
        print("=" * 60)
        
        all_matches = []
        today_str = self.get_paris_date_string()
        
        async with aiohttp.ClientSession() as session:
            # Chercher dans toutes nos ligues
            for league_name, league_id in self.leagues.items():
                print(f"Recherche {league_name}...")
                
                try:
                    url = f"{self.base_url}/fixtures"
                    params = {
                        'league': league_id,
                        'date': today_str,
                        'season': Config.CURRENT_SEASON
                    }
                    
                    async with session.get(url, headers=self.headers, params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            
                            if data.get('response'):
                                matches = data['response']
                                print(f"  {len(matches)} match(s) trouvé(s)")
                                
                                for match in matches:
                                    # Convertir heure UTC vers Paris
                                    utc_time = datetime.fromisoformat(match['fixture']['date'].replace('Z', '+00:00'))
                                    paris_time = utc_time.astimezone(self.paris_tz)
                                    
                                    match_info = {
                                        'fixture_id': match['fixture']['id'],
                                        'league_name': league_name,
                                        'league_id': league_id,
                                        'home_team': match['teams']['home']['name'],
                                        'away_team': match['teams']['away']['name'],
                                        'home_id': match['teams']['home']['id'],
                                        'away_id': match['teams']['away']['id'],
                                        'paris_time': paris_time,
                                        'status': match['fixture']['status']['long'],
                                        'venue': match['fixture']['venue']['name']
                                    }
                                    
                                    all_matches.append(match_info)
                                    
                                    print(f"    {paris_time.strftime('%H:%M')} - {match_info['home_team']} vs {match_info['away_team']}")
                            else:
                                print("  Aucun match")
                        else:
                            print(f"  Erreur API: {resp.status}")
                            
                except Exception as e:
                    print(f"  Erreur {league_name}: {e}")
                
                # Délai entre requêtes
                await asyncio.sleep(Config.API_RATE_LIMIT_DELAY)
        
        return all_matches
    
    async def get_team_current_stats(self, team_id: int, league_id: int) -> Dict:
        """Récupérer stats actuelles d'une équipe"""
        cache_key = f"team_{team_id}_{league_id}"
        
        if cache_key in self.teams_cache:
            return self.teams_cache[cache_key]
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/teams/statistics"
            params = {
                'team': team_id,
                'league': league_id,
                'season': Config.CURRENT_SEASON
            }
            
            async with session.get(url, headers=self.headers, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('response'):
                        stats = self._process_team_stats(data['response'])
                        self.teams_cache[cache_key] = stats
                        return stats
                
        return self._get_default_team_stats()
    
    def _process_team_stats(self, raw_stats: Dict) -> Dict:
        """Traiter les stats d'équipe de l'API"""
        fixtures = raw_stats.get('fixtures', {})
        goals = raw_stats.get('goals', {})
        
        played = fixtures.get('played', {}).get('total', 0) or 1  # Éviter division par zéro
        
        return {
            'matches_played': played,
            'wins': fixtures.get('wins', {}).get('total', 0),
            'draws': fixtures.get('draws', {}).get('total', 0),
            'losses': fixtures.get('loses', {}).get('total', 0),
            'goals_for': goals.get('for', {}).get('total', {}).get('total', 0),
            'goals_against': goals.get('against', {}).get('total', {}).get('total', 0),
            'avg_goals_for': float(goals.get('for', {}).get('average', {}).get('total', '0') or 0),
            'avg_goals_against': float(goals.get('against', {}).get('average', {}).get('total', '0') or 0),
            'home_wins': fixtures.get('wins', {}).get('home', 0),
            'away_wins': fixtures.get('wins', {}).get('away', 0),
            'clean_sheets': raw_stats.get('clean_sheet', {}).get('total', 0),
            'win_rate': fixtures.get('wins', {}).get('total', 0) / played if played > 0 else 0
        }
    
    def _get_default_team_stats(self) -> Dict:
        """Stats par défaut si API indisponible"""
        return {
            'matches_played': 10,
            'wins': 5,
            'draws': 3,
            'losses': 2,
            'goals_for': 15,
            'goals_against': 12,
            'avg_goals_for': 1.5,
            'avg_goals_against': 1.2,
            'home_wins': 3,
            'away_wins': 2,
            'clean_sheets': 3,
            'win_rate': 0.5
        }
    
    async def get_match_prediction_data(self, match_info: Dict) -> Dict:
        """Récupérer toutes les données pour prédiction"""
        print(f"Collecte données: {match_info['home_team']} vs {match_info['away_team']}")
        
        # Stats des équipes
        home_stats = await self.get_team_current_stats(match_info['home_id'], match_info['league_id'])
        away_stats = await self.get_team_current_stats(match_info['away_id'], match_info['league_id'])
        
        # Déterminer l'étape de prédiction selon temps restant
        time_until = match_info['paris_time'] - self.get_paris_time_now()
        hours_until = time_until.total_seconds() / 3600
        
        # Tracker les données utilisées
        used_data = {
            'team_stats': True,  # Toujours disponible
            'formations': False,
            'lineups': False,
            'weather': False,
            'injuries': False,
            'referee': False,
            'live_events': False
        }
        
        if hours_until <= 1:
            stage = "LATE"  # Lineups disponibles
            used_data.update({
                'formations': True,
                'lineups': True,
                'weather': True,
                'referee': True,
                'injuries': True
            })
        elif hours_until <= 24:
            stage = "MID"   # Formations probables
            used_data.update({
                'formations': True,
                'injuries': True
            })
        else:
            stage = "EARLY" # Stats équipes seulement
        
        return {
            'match_info': match_info,
            'home_stats': home_stats,
            'away_stats': away_stats,
            'hours_until_match': hours_until,
            'prediction_stage': stage,
            'data_completeness': self._calculate_data_completeness(stage),
            'used_data_sources': used_data
        }
    
    def _calculate_data_completeness(self, stage: str) -> float:
        """Calculer complétude données selon étape"""
        if stage == "LATE":
            return 0.95
        elif stage == "MID":
            return 0.75
        else:  # EARLY
            return 0.50
    
    def create_ml_features(self, prediction_data: Dict) -> np.ndarray:
        """Créer features pour modèles ML"""
        home_stats = prediction_data['home_stats']
        away_stats = prediction_data['away_stats']
        
        # Features de base (compatible avec modèles existants)
        features = [
            # Stats équipe domicile
            home_stats['win_rate'],
            home_stats['avg_goals_for'],
            home_stats['avg_goals_against'],
            home_stats['matches_played'],
            home_stats['home_wins'] / max(1, home_stats['matches_played'] * 0.5),
            home_stats['clean_sheets'] / max(1, home_stats['matches_played']),
            
            # Stats équipe extérieur
            away_stats['win_rate'],
            away_stats['avg_goals_for'],
            away_stats['avg_goals_against'],
            away_stats['matches_played'],
            away_stats['away_wins'] / max(1, away_stats['matches_played'] * 0.5),
            away_stats['clean_sheets'] / max(1, away_stats['matches_played']),
            
            # Features comparatives
            home_stats['avg_goals_for'] - away_stats['avg_goals_against'],
            away_stats['avg_goals_for'] - home_stats['avg_goals_against'],
            home_stats['win_rate'] - away_stats['win_rate'],
            
            # Features contextuelles
            1.0,  # Avantage domicile
            prediction_data['data_completeness'],
            prediction_data['hours_until_match'] / 24.0  # Normaliser heures
        ]
        
        # Étendre à 53 features si nécessaire (pour compatibilité)
        while len(features) < 53:
            features.append(0.0)  # Padding avec zéros
        
        return np.array(features[:53]).reshape(1, -1)  # Garder seulement 53 features
    
    async def predict_match(self, match_info: Dict) -> Dict:
        """Prédire résultat d'un match"""
        # Collecter données
        prediction_data = await self.get_match_prediction_data(match_info)
        
        # Créer features ML
        features = self.create_ml_features(prediction_data)
        
        league_id = match_info['league_id']
        predictions = {}
        
        # Utiliser modèles selon la ligue
        if league_id in self.models and self.models[league_id]:
            print(f"  Utilisation modèles ML pour ligue {league_id}")
            
            for model_type, model in self.models[league_id].items():
                try:
                    # Appliquer scaler si disponible
                    input_features = features
                    if league_id in self.scalers and model_type in self.scalers[league_id]:
                        input_features = self.scalers[league_id][model_type].transform(features)
                    
                    # Prédiction
                    prediction = model.predict(input_features)[0]
                    
                    # Ajuster confiance selon étape
                    base_confidence = 0.85
                    stage_bonus = {
                        'LATE': 0.10,
                        'MID': 0.05,
                        'EARLY': 0.0
                    }
                    confidence = min(0.95, base_confidence + stage_bonus.get(prediction_data['prediction_stage'], 0))
                    
                    predictions[model_type] = {
                        'prediction': float(prediction),
                        'confidence': confidence,
                        'model_used': f"complete_{league_id}_{model_type}"
                    }
                    
                except Exception as e:
                    print(f"    Erreur modèle {model_type}: {e}")
        
        else:
            print(f"  Modèles non disponibles pour ligue {league_id}, utilisation fallback")
            predictions = self._fallback_predictions(prediction_data)
        
        return {
            'match': {
                'home_team': match_info['home_team'],
                'away_team': match_info['away_team'],
                'league': match_info['league_name'],
                'paris_time': match_info['paris_time'].strftime('%d/%m/%Y %H:%M'),
                'hours_until': prediction_data['hours_until_match']
            },
            'stage': prediction_data['prediction_stage'],
            'data_completeness': prediction_data['data_completeness'],
            'used_data_sources': prediction_data['used_data_sources'],
            'predictions': predictions
        }
    
    def _fallback_predictions(self, prediction_data: Dict) -> Dict:
        """Prédictions fallback si modèles indisponibles"""
        home_stats = prediction_data['home_stats']
        away_stats = prediction_data['away_stats']
        
        # Calculs simples basés sur stats
        home_strength = home_stats['win_rate'] * 1.2  # Avantage domicile
        away_strength = away_stats['win_rate']
        
        total_strength = home_strength + away_strength
        home_win_prob = home_strength / total_strength if total_strength > 0 else 0.5
        
        return {
            'next_match_result': {
                'prediction': home_win_prob,
                'confidence': 0.65,
                'model_used': 'fallback_statistical'
            },
            'goals_scored': {
                'prediction': (home_stats['avg_goals_for'] + away_stats['avg_goals_for']) / 2,
                'confidence': 0.60,
                'model_used': 'fallback_statistical'
            },
            'both_teams_score': {
                'prediction': min(0.85, (home_stats['avg_goals_for'] + away_stats['avg_goals_for']) / 4),
                'confidence': 0.55,
                'model_used': 'fallback_statistical'
            }
        }
    
    def format_prediction_output(self, prediction_result: Dict) -> None:
        """Afficher prédictions formatées"""
        match = prediction_result['match']
        
        print(f"\n{'='*60}")
        print(f"PREDICTION: {match['home_team']} vs {match['away_team']}")
        print(f"{'='*60}")
        print(f"Ligue: {match['league']}")
        print(f"Heure Paris: {match['paris_time']}")
        print(f"Temps restant: {match['hours_until']:.1f}h")
        print(f"Etape: {prediction_result['stage']} ({prediction_result['data_completeness']*100:.0f}% données)")
        
        # Afficher les données pré-match utilisées
        if 'used_data_sources' in prediction_result:
            self._display_used_data_sources(prediction_result['used_data_sources'])
        
        print(f"\nPREDICTIONS:")
        print("-" * 40)
        
        predictions = prediction_result['predictions']
        
        # Résultat du match
        if 'next_match_result' in predictions:
            result_pred = predictions['next_match_result']
            home_prob = result_pred['prediction']
            
            if home_prob > 0.6:
                outcome = f"VICTOIRE {match['home_team']}"
                prob_pct = home_prob * 100
            elif home_prob < 0.4:
                outcome = f"VICTOIRE {match['away_team']}"
                prob_pct = (1 - home_prob) * 100
            else:
                outcome = "MATCH EQUILIBRE"
                prob_pct = 50
            
            print(f"Résultat: {outcome} ({prob_pct:.0f}%)")
            print(f"Confiance: {result_pred['confidence']*100:.0f}%")
        
        # Buts
        if 'goals_scored' in predictions:
            goals_pred = predictions['goals_scored']
            total_goals = goals_pred['prediction']
            
            over_under = "PLUS" if total_goals > 2.5 else "MOINS"
            print(f"Buts totaux: {total_goals:.1f} -> {over_under} de 2.5 buts")
        
        # Les deux équipes marquent
        if 'both_teams_score' in predictions:
            bts_pred = predictions['both_teams_score']
            bts_prob = bts_pred['prediction']
            
            bts_outcome = "OUI" if bts_prob > 0.5 else "NON"
            print(f"Both Teams Score: {bts_outcome} ({bts_prob*100:.0f}%)")
        
        print("\nModèles utilisés:")
        for pred_type, pred_data in predictions.items():
            print(f"  {pred_type}: {pred_data['model_used']}")
    
    def _display_used_data_sources(self, used_data: Dict) -> None:
        """Afficher les sources de données utilisées"""
        print(f"\nDONNEES PRE-MATCH UTILISEES:")
        print("-" * 30)
        
        data_labels = {
            'team_stats': 'Stats équipes (saison actuelle)',
            'formations': 'Formations probables',
            'lineups': 'Compositions officielles',
            'weather': 'Conditions météorologiques',
            'injuries': 'Blessures et suspensions',
            'referee': 'Informations arbitre',
            'live_events': 'Événements temps réel'
        }
        
        used_count = 0
        for data_type, is_used in used_data.items():
            status = "[UTILISE]" if is_used else "[Non disponible]"
            label = data_labels.get(data_type, data_type)
            print(f"  {label}: {status}")
            if is_used:
                used_count += 1
        
        print(f"\nTotal données utilisées: {used_count}/{len(used_data)} sources")
    
    async def display_detailed_match_data(self, prediction_data: Dict) -> None:
        """Afficher les données détaillées récupérées pour le match"""
        match_info = prediction_data['match_info']
        
        print(f"\n{'='*60}")
        print(f"DONNEES DETAILLEES RECUPEREES")
        print(f"{'='*60}")
        print(f"Match: {match_info['home_team']} vs {match_info['away_team']}")
        print(f"Etape: {prediction_data['prediction_stage']}")
        
        # Stats équipes
        print(f"\n--- STATS EQUIPES ---")
        home_stats = prediction_data.get('home_stats', {})
        away_stats = prediction_data.get('away_stats', {})
        
        print(f"{match_info['home_team']}:")
        print(f"  Matchs joués: {home_stats.get('matches_played', 'N/A')}")
        print(f"  Victoires: {home_stats.get('wins', 'N/A')}")
        print(f"  Buts/match: {home_stats.get('avg_goals_for', 'N/A'):.2f}")
        print(f"  Taux victoire: {home_stats.get('win_rate', 0)*100:.1f}%")
        
        print(f"\n{match_info['away_team']}:")
        print(f"  Matchs joués: {away_stats.get('matches_played', 'N/A')}")
        print(f"  Victoires: {away_stats.get('wins', 'N/A')}")
        print(f"  Buts/match: {away_stats.get('avg_goals_for', 'N/A'):.2f}")
        print(f"  Taux victoire: {away_stats.get('win_rate', 0)*100:.1f}%")
        
        # Formations (si étape MID ou LATE) - API FOOTBALL REELLE
        if prediction_data['prediction_stage'] in ['MID', 'LATE']:
            print(f"\n--- FORMATIONS ET LINEUPS (API FOOTBALL) ---")
            await self._display_real_formations_lineups(match_info)
        
        # Blessures (si étape MID ou LATE) - API FOOTBALL REELLE
        if prediction_data['prediction_stage'] in ['MID', 'LATE']:
            print(f"\n--- BLESSURES ET SUSPENSIONS (API FOOTBALL) ---")
            await self._display_real_injuries(match_info)
    
    async def _display_real_formations_lineups(self, match_info: Dict):
        """Afficher formations et lineups REELS depuis API Football"""
        fixture_id = match_info['fixture_id']
        
        async with aiohttp.ClientSession() as session:
            # 1. Lineups officiels (si disponibles)
            url = f"{self.base_url}/fixtures/lineups"
            params = {'fixture': fixture_id}
            
            async with session.get(url, headers=self.headers, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('response'):
                        lineups = data['response']
                        
                        for team_lineup in lineups:
                            team_name = team_lineup['team']['name']
                            formation = team_lineup['formation']
                            startXI = team_lineup['startXI']
                            
                            print(f"\n{team_name} - Formation: {formation}")
                            
                            if startXI:
                                print("  Composition officielle:")
                                for player in startXI:
                                    player_info = player['player']
                                    print(f"    {player_info['number']:2d}. {player_info['name']} ({player_info['pos']})")
                            else:
                                print("    Composition pas encore publiée")
                    else:
                        print("Compositions officielles pas encore disponibles")
                        await self._display_probable_formations(match_info)
                else:
                    print(f"Erreur récupération lineups: {resp.status}")
                    await self._display_probable_formations(match_info)
    
    async def _display_probable_formations(self, match_info: Dict):
        """Afficher formations probables basées sur matchs récents"""
        home_id = match_info['home_id']
        away_id = match_info['away_id']
        
        async with aiohttp.ClientSession() as session:
            # Récupérer derniers matchs pour déduire formation
            for team_id, team_name in [(home_id, match_info['home_team']), (away_id, match_info['away_team'])]:
                url = f"{self.base_url}/fixtures"
                params = {
                    'team': team_id,
                    'last': 3  # 3 derniers matchs
                }
                
                async with session.get(url, headers=self.headers, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('response'):
                            recent_fixtures = data['response']
                            
                            print(f"\n{team_name} - Formation probable (basée sur matchs récents):")
                            
                            formations_used = []
                            for fixture in recent_fixtures[:3]:
                                fixture_id = fixture['fixture']['id']
                                
                                # Récupérer lineup de ce match passé
                                lineup_url = f"{self.base_url}/fixtures/lineups"
                                lineup_params = {'fixture': fixture_id}
                                
                                async with session.get(lineup_url, headers=self.headers, params=lineup_params) as lineup_resp:
                                    if lineup_resp.status == 200:
                                        lineup_data = await lineup_resp.json()
                                        if lineup_data.get('response'):
                                            for team_lineup in lineup_data['response']:
                                                if team_lineup['team']['id'] == team_id:
                                                    formations_used.append(team_lineup['formation'])
                                                    break
                            
                            if formations_used:
                                # Formation la plus utilisée
                                most_common = max(set(formations_used), key=formations_used.count)
                                confidence = formations_used.count(most_common) / len(formations_used) * 100
                                print(f"  Formation probable: {most_common} (utilisée {confidence:.0f}% des derniers matchs)")
                            else:
                                print(f"  Formation: Données non disponibles")
                        
                        await asyncio.sleep(0.5)  # Rate limiting
    
    async def _display_real_injuries(self, match_info: Dict):
        """Afficher blessures REELLES depuis API Football"""
        home_id = match_info['home_id']
        away_id = match_info['away_id']
        league_id = match_info['league_id']
        
        async with aiohttp.ClientSession() as session:
            for team_id, team_name in [(home_id, match_info['home_team']), (away_id, match_info['away_team'])]:
                url = f"{self.base_url}/injuries"
                params = {
                    'team': team_id,
                    'league': league_id,
                    'season': Config.CURRENT_SEASON
                }
                
                async with session.get(url, headers=self.headers, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('response'):
                            injuries = data['response']
                            
                            active_injuries = [inj for inj in injuries if inj['player']['id'] is not None]
                            
                            if active_injuries:
                                print(f"\n{team_name} - Blessures actuelles:")
                                for injury in active_injuries:
                                    player = injury['player']['name']
                                    injury_type = injury.get('type', 'Injury')
                                    reason = injury.get('reason', 'Non spécifié')
                                    print(f"  • {player}: {injury_type} - {reason}")
                            else:
                                print(f"\n{team_name}: Aucune blessure reportée")
                        else:
                            print(f"\n{team_name}: Aucune blessure reportée")
                    else:
                        print(f"\n{team_name}: Erreur récupération blessures (status {resp.status})")
                
                await asyncio.sleep(0.5)  # Rate limiting
    
    async def run_daily_predictions(self):
        """Exécuter prédictions pour tous les matchs du jour"""
        print(f"PREDICTIONS MATCHS DU JOUR - HEURE PARIS")
        print(f"Exécution: {self.get_paris_time_now().strftime('%d/%m/%Y %H:%M:%S %Z')}")
        print("=" * 70)
        
        # Trouver matchs d'aujourd'hui
        matches = await self.find_matches_today()
        
        if not matches:
            print("Aucun match trouvé aujourd'hui")
            return
        
        print(f"\n{len(matches)} match(s) trouvé(s) - Génération prédictions...")
        
        # Prédire chaque match
        for i, match in enumerate(matches, 1):
            print(f"\n[{i}/{len(matches)}] Traitement match...")
            
            try:
                prediction = await self.predict_match(match)
                self.format_prediction_output(prediction)
                
            except Exception as e:
                print(f"Erreur prédiction: {e}")
            
            # Délai entre prédictions
            await asyncio.sleep(1)
        
        print(f"\n{'='*70}")
        print("PREDICTIONS TERMINEES")
        print(f"Heure fin: {self.get_paris_time_now().strftime('%H:%M:%S')}")


# Fonction principale
async def main():
    """Fonction principale"""
    system = FootballPredictionSystem()
    await system.run_daily_predictions()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nArrêt utilisateur")
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()