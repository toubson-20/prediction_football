"""
COLLECTEUR COMPLET API-FOOTBALL
Collecte TOUTES les donnees disponibles : matchs, equipes, joueurs, evenements, contexte
"""

import requests
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time
import sys
from collections import defaultdict

sys.path.append('src')
from config import API_FOOTBALL_CONFIG

class CompleteDataCollector:
    """Collecteur exhaustif de toutes les donnees API-Football"""
    
    def __init__(self):
        self.api_config = API_FOOTBALL_CONFIG
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {"X-RapidAPI-Key": self.api_config["api_key"]}
        
        # Dossiers de stockage
        self.data_dir = Path("data/complete_collection")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Sous-dossiers par type de donnees
        self.matches_dir = self.data_dir / "matches"
        self.teams_dir = self.data_dir / "teams" 
        self.players_dir = self.data_dir / "players"
        self.leagues_dir = self.data_dir / "leagues"
        self.seasons_dir = self.data_dir / "seasons"
        self.venues_dir = self.data_dir / "venues"
        self.statistics_dir = self.data_dir / "statistics"
        
        for dir_path in [self.matches_dir, self.teams_dir, self.players_dir, 
                        self.leagues_dir, self.seasons_dir, self.venues_dir, self.statistics_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Configuration des ligues principales
        self.main_leagues = {
            39: "Premier League",
            140: "La Liga", 
            61: "Ligue 1",
            78: "Bundesliga",
            135: "Serie A",
            2: "Champions League",
            3: "Europa League"
        }
        
        self.season = 2025
        
        # Compteurs et logs
        self.collection_stats = defaultdict(int)
        self.api_calls_made = 0
        self.rate_limit_delay = 1.2  # Securite
        
    def make_api_call(self, endpoint, params=None):
        """Faire un appel API avec gestion des erreurs et rate limiting"""
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            time.sleep(self.rate_limit_delay)
            response = requests.get(url, headers=self.headers, params=params or {})
            self.api_calls_made += 1
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"ERREUR API {endpoint}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"ERREUR appel {endpoint}: {e}")
            return None
    
    def collect_leagues_complete_info(self):
        """Collecter informations completes sur toutes les ligues"""
        
        print(f"\n=== COLLECTE INFORMATIONS LIGUES COMPLETES ===")
        
        for league_id, league_name in self.main_leagues.items():
            try:
                safe_league_name = league_name.encode('ascii', 'ignore').decode('ascii')
                print(f"Collecte ligue {safe_league_name}...")
            except:
                print(f"Collecte ligue ID {league_id}...")
            
            # Informations generales ligue (avec sauvegarde intelligente)
            league_file = self.leagues_dir / f"league_{league_id}_info.json"
            if not league_file.exists():
                league_data = self.make_api_call("leagues", {"id": league_id})
                if league_data:
                    with open(league_file, 'w') as f:
                        json.dump(league_data, f, indent=2)
                    self.collection_stats['leagues_info'] += 1
            else:
                print(f"    -> Deja collecte, passage...")
                self.collection_stats['leagues_info'] += 1
            
            # Saisons disponibles
            seasons_data = self.make_api_call("leagues/seasons")
            if seasons_data:
                seasons_file = self.seasons_dir / f"seasons_available.json"
                with open(seasons_file, 'w') as f:
                    json.dump(seasons_data, f, indent=2)
                self.collection_stats['seasons_data'] += 1
        
        print(f"  Ligues info collectees: {self.collection_stats['leagues_info']}")
    
    def collect_teams_complete_data(self):
        """Collecter donnees completes sur toutes les equipes"""
        
        print(f"\n=== COLLECTE DONNEES COMPLETES EQUIPES ===")
        
        for league_id, league_name in self.main_leagues.items():
            try:
                safe_league_name = league_name.encode('ascii', 'ignore').decode('ascii')
                print(f"Collecte equipes {safe_league_name}...")
            except:
                print(f"Collecte equipes ligue {league_id}...")
            
            # Liste des equipes de la ligue
            teams_data = self.make_api_call("teams", {
                "league": league_id, 
                "season": self.season
            })
            
            if teams_data and teams_data.get('response'):
                teams_file = self.teams_dir / f"teams_{league_id}_{self.season}.json"
                with open(teams_file, 'w') as f:
                    json.dump(teams_data, f, indent=2)
                
                # Pour chaque equipe, collecter donnees detaillees
                for team_info in teams_data['response']:
                    team_id = team_info['team']['id']
                    team_name = team_info['team']['name']
                    
                    try:
                        print(f"  Collecte {team_name} (ID: {team_id})...")
                    except UnicodeEncodeError:
                        # Gestion des caracteres non-ASCII pour Windows
                        safe_team_name = team_name.encode('ascii', 'ignore').decode('ascii')
                        print(f"  Collecte {safe_team_name} (ID: {team_id})...")
                    
                    # Statistiques equipe detaillees
                    team_stats = self.make_api_call("teams/statistics", {
                        "league": league_id,
                        "season": self.season, 
                        "team": team_id
                    })
                    
                    if team_stats:
                        stats_file = self.statistics_dir / f"team_stats_{team_id}_{league_id}_{self.season}.json"
                        with open(stats_file, 'w') as f:
                            json.dump(team_stats, f, indent=2)
                        self.collection_stats['team_statistics'] += 1
                    
                    # Informations sur le stade
                    venue_data = team_info.get('venue')
                    if venue_data:
                        venue_file = self.venues_dir / f"venue_{venue_data.get('id', team_id)}.json"
                        with open(venue_file, 'w') as f:
                            json.dump(venue_data, f, indent=2)
                        self.collection_stats['venues'] += 1
                
                self.collection_stats['teams'] += len(teams_data['response'])
        
        print(f"  Equipes collectees: {self.collection_stats['teams']}")
        print(f"  Statistiques equipes: {self.collection_stats['team_statistics']}")
    
    def collect_players_complete_data(self):
        """Collecter donnees completes sur tous les joueurs"""
        
        print(f"\n=== COLLECTE DONNEES COMPLETES JOUEURS ===")
        
        for league_id, league_name in self.main_leagues.items():
            try:
                safe_league_name = league_name.encode('ascii', 'ignore').decode('ascii')
                print(f"Collecte joueurs {safe_league_name}...")
            except:
                print(f"Collecte joueurs ligue {league_id}...")
            
            # Charger equipes de cette ligue
            teams_file = self.teams_dir / f"teams_{league_id}_{self.season}.json"
            if not teams_file.exists():
                continue
            
            with open(teams_file, 'r') as f:
                teams_data = json.load(f)
            
            for team_info in teams_data.get('response', []):
                team_id = team_info['team']['id']
                team_name = team_info['team']['name']
                
                try:
                    safe_team_name = team_name.encode('ascii', 'ignore').decode('ascii')
                    print(f"  Collecte joueurs {safe_team_name}...")
                except:
                    print(f"  Collecte joueurs equipe {team_id}...")
                
                # Joueurs de l'equipe
                players_data = self.make_api_call("players", {
                    "team": team_id,
                    "season": self.season
                })
                
                if players_data and players_data.get('response'):
                    players_file = self.players_dir / f"players_team_{team_id}_{self.season}.json"
                    with open(players_file, 'w') as f:
                        json.dump(players_data, f, indent=2)
                    
                    self.collection_stats['players'] += len(players_data['response'])
                    
                    # Statistiques detaillees pour chaque joueur vedette (top 5 par equipe)
                    top_players = players_data['response'][:5]  # Limiter pour eviter trop d'appels API
                    
                    for player_info in top_players:
                        player_id = player_info['player']['id']
                        player_name = player_info['player']['name']
                        
                        # Statistiques joueur detaillees
                        player_stats = self.make_api_call("players", {
                            "id": player_id,
                            "season": self.season
                        })
                        
                        if player_stats:
                            player_stats_file = self.players_dir / f"player_stats_{player_id}_{self.season}.json"
                            with open(player_stats_file, 'w') as f:
                                json.dump(player_stats, f, indent=2)
                            self.collection_stats['player_statistics'] += 1
        
        print(f"  Joueurs collectes: {self.collection_stats['players']}")
        print(f"  Statistiques joueurs: {self.collection_stats['player_statistics']}")
    
    def collect_matches_complete_data(self, days_back=30):
        """Collecter donnees completes sur tous les matchs"""
        
        print(f"\n=== COLLECTE DONNEES COMPLETES MATCHS ({days_back} derniers jours) ===")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        for league_id, league_name in self.main_leagues.items():
            try:
                safe_league_name = league_name.encode('ascii', 'ignore').decode('ascii')
                print(f"Collecte matchs {safe_league_name}...")
            except:
                print(f"Collecte matchs ligue {league_id}...")
            
            # Matchs de la periode
            fixtures_data = self.make_api_call("fixtures", {
                "league": league_id,
                "season": self.season,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d")
            })
            
            if fixtures_data and fixtures_data.get('response'):
                fixtures_file = self.matches_dir / f"fixtures_{league_id}_{self.season}.json"
                with open(fixtures_file, 'w') as f:
                    json.dump(fixtures_data, f, indent=2)
                
                # Pour chaque match, collecter donnees detaillees
                for fixture in fixtures_data['response']:
                    fixture_id = fixture['fixture']['id']
                    
                    if fixture['fixture']['status']['short'] == 'FT':  # Match termine
                        print(f"  Collecte match {fixture_id}...")
                        
                        # Statistiques detaillees du match
                        match_stats = self.make_api_call("fixtures/statistics", {
                            "fixture": fixture_id
                        })
                        
                        if match_stats:
                            stats_file = self.statistics_dir / f"match_stats_{fixture_id}.json"
                            with open(stats_file, 'w') as f:
                                json.dump(match_stats, f, indent=2)
                            self.collection_stats['match_statistics'] += 1
                        
                        # Evenements du match (timeline)
                        match_events = self.make_api_call("fixtures/events", {
                            "fixture": fixture_id
                        })
                        
                        if match_events:
                            events_file = self.matches_dir / f"match_events_{fixture_id}.json"
                            with open(events_file, 'w') as f:
                                json.dump(match_events, f, indent=2)
                            self.collection_stats['match_events'] += 1
                        
                        # Compositions d'equipes
                        lineups = self.make_api_call("fixtures/lineups", {
                            "fixture": fixture_id
                        })
                        
                        if lineups:
                            lineups_file = self.matches_dir / f"match_lineups_{fixture_id}.json"
                            with open(lineups_file, 'w') as f:
                                json.dump(lineups, f, indent=2)
                            self.collection_stats['match_lineups'] += 1
                        
                        # Statistiques joueurs du match
                        players_stats = self.make_api_call("fixtures/players", {
                            "fixture": fixture_id
                        })
                        
                        if players_stats:
                            players_stats_file = self.matches_dir / f"match_players_{fixture_id}.json"
                            with open(players_stats_file, 'w') as f:
                                json.dump(players_stats, f, indent=2)
                            self.collection_stats['match_player_stats'] += 1
                        
                        self.collection_stats['complete_matches'] += 1
        
        print(f"  Matchs complets collectes: {self.collection_stats['complete_matches']}")
        print(f"  Statistiques matchs: {self.collection_stats['match_statistics']}")
        print(f"  Evenements matchs: {self.collection_stats['match_events']}")
        print(f"  Lineups matchs: {self.collection_stats['match_lineups']}")
    
    def collect_additional_contextual_data(self):
        """Collecter donnees contextuelles additionnelles"""
        
        print(f"\n=== COLLECTE DONNEES CONTEXTUELLES ===")
        
        # Cotes/Odds disponibles
        for league_id, league_name in self.main_leagues.items():
            try:
                safe_league_name = league_name.encode('ascii', 'ignore').decode('ascii')
                print(f"Collecte cotes {safe_league_name}...")
            except:
                print(f"Collecte cotes ligue {league_id}...")
            
            # Bookmakers et cotes
            odds_data = self.make_api_call("odds", {
                "league": league_id,
                "season": self.season
            })
            
            if odds_data:
                odds_file = self.data_dir / f"odds_{league_id}_{self.season}.json"
                with open(odds_file, 'w') as f:
                    json.dump(odds_data, f, indent=2)
                self.collection_stats['odds'] += 1
        
        # Predictions API (si disponible)
        predictions_data = self.make_api_call("predictions", {
            "league": 39,  # Test avec Premier League
            "season": self.season
        })
        
        if predictions_data:
            pred_file = self.data_dir / f"api_predictions_{self.season}.json"
            with open(pred_file, 'w') as f:
                json.dump(predictions_data, f, indent=2)
            self.collection_stats['predictions'] += 1
        
        # Informations sur les arbitres
        timezone_data = self.make_api_call("timezone")
        if timezone_data:
            timezone_file = self.data_dir / f"timezones.json"
            with open(timezone_file, 'w') as f:
                json.dump(timezone_data, f, indent=2)
            self.collection_stats['timezone'] += 1
        
        print(f"  Cotes collectees: {self.collection_stats['odds']}")
        print(f"  Predictions API: {self.collection_stats['predictions']}")
    
    def comprehensive_collection(self):
        """Collecte comprehensive complete"""
        
        print(f"{'='*70}")
        print("COLLECTE COMPREHENSIVE COMPLETE API-FOOTBALL")
        print(f"{'='*70}")
        print(f"Debut: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = datetime.now()
        
        # 1. Informations ligues
        self.collect_leagues_complete_info()
        
        # 2. Donnees equipes completes
        self.collect_teams_complete_data()
        
        # 3. Donnees joueurs completes
        self.collect_players_complete_data()
        
        # 4. Donnees matchs completes
        self.collect_matches_complete_data(days_back=60)  # 2 mois de matchs
        
        # 5. Donnees contextuelles
        self.collect_additional_contextual_data()
        
        # 6. Resume final
        duration = datetime.now() - start_time
        
        print(f"\n{'='*70}")
        print("COLLECTE COMPREHENSIVE TERMINEE")
        print(f"{'='*70}")
        
        print(f"Duree totale: {duration.total_seconds() / 60:.1f} minutes")
        print(f"Appels API effectues: {self.api_calls_made}")
        print(f"Dossier donnees: {self.data_dir}")
        
        print(f"\nSTATISTIQUES COLLECTION:")
        for data_type, count in self.collection_stats.items():
            print(f"  {data_type}: {count}")
        
        # Generer rapport de collection
        collection_report = {
            'collection_timestamp': datetime.now().isoformat(),
            'duration_minutes': duration.total_seconds() / 60,
            'api_calls_made': self.api_calls_made,
            'data_directory': str(self.data_dir),
            'collection_stats': dict(self.collection_stats),
            'leagues_processed': list(self.main_leagues.keys()),
            'season': self.season
        }
        
        report_file = self.data_dir / "collection_report.json"
        with open(report_file, 'w') as f:
            json.dump(collection_report, f, indent=2)
        
        print(f"\nRapport sauve: {report_file}")
        
        return collection_report

def main():
    """Fonction principale de collecte comprehensive"""
    
    print("COLLECTEUR COMPLET API-FOOTBALL")
    print("Collecte EXHAUSTIVE de toutes les donnees disponibles")
    print()
    
    try:
        collector = CompleteDataCollector()
        
        # Collecte complete
        report = collector.comprehensive_collection()
        
        print(f"\nCOLLECTE COMPREHENSIVE REUSSIE!")
        print("Toutes les donnees API-Football collectees pour ML avance")
        
        return report
        
    except Exception as e:
        print(f"ERREUR COLLECTE: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()