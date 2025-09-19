#!/usr/bin/env python3
"""
Collecteur de donnees actuelles COMPLET - Saison 2025-2026
Recupere TOUTES les donnees recentes pour toutes les competitions
"""

import requests
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
from config import Config

class CompleteCurrentDataCollector:
    def __init__(self):
        self.api_key = Config.FOOTBALL_API_KEY
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            'x-apisports-key': self.api_key
        }

        # Dossier pour donnees actuelles
        self.current_dir = Path("data/current_2025")
        self.current_dir.mkdir(parents=True, exist_ok=True)

        # Toutes les competitions principales
        self.competitions = {
            'premier_league': 39,
            'la_liga': 140,
            'bundesliga': 78,
            'ligue_1': 61,
            'serie_a': 135,
            'champions_league': 2,
            'europa_league': 3,
            'conference_league': 848,
            'nations_league': 5,
            'world_cup': 1,
            'euros': 4
        }

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def api_request(self, endpoint, params=None):
        """Faire une requete API avec gestion d'erreurs"""
        url = f"{self.base_url}/{endpoint}"

        try:
            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code == 200:
                data = response.json()
                self.logger.info(f"API Success: {endpoint} - {data.get('results', 0)} resultats")
                return data
            else:
                self.logger.error(f"API Error {response.status_code}: {endpoint}")
                return None

        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            return None

        finally:
            # Respecter le rate limiting
            time.sleep(1)

    def get_league_current_standings(self, league_id, league_name):
        """Recuperer le classement actuel d'une ligue"""
        self.logger.info(f"Collecte classement actuel - {league_name}")

        data = self.api_request('standings', {
            'league': league_id,
            'season': 2025
        })

        if data and data.get('response'):
            output_file = self.current_dir / f"standings_{league_name}_{league_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Sauve: {output_file}")
            return data['response']

        return []

    def get_league_topscorers(self, league_id, league_name):
        """Recuperer les meilleurs buteurs actuels"""
        self.logger.info(f"Collecte meilleurs buteurs - {league_name}")

        data = self.api_request('players/topscorers', {
            'league': league_id,
            'season': 2025
        })

        if data and data.get('response'):
            output_file = self.current_dir / f"topscorers_{league_name}_{league_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Sauve: {output_file} ({len(data['response'])} joueurs)")
            return data['response']

        return []

    def get_league_topassists(self, league_id, league_name):
        """Recuperer les meilleurs passeurs actuels"""
        self.logger.info(f"Collecte meilleurs passeurs - {league_name}")

        data = self.api_request('players/topassists', {
            'league': league_id,
            'season': 2025
        })

        if data and data.get('response'):
            output_file = self.current_dir / f"topassists_{league_name}_{league_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Sauve: {output_file} ({len(data['response'])} joueurs)")
            return data['response']

        return []

    def get_recent_matches_all(self, league_id, league_name, last_days=14):
        """Recuperer tous les matchs recents d'une ligue"""
        self.logger.info(f"Collecte matchs recents - {league_name}")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=last_days)

        data = self.api_request('fixtures', {
            'league': league_id,
            'season': 2025,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d')
        })

        if data and data.get('response'):
            output_file = self.current_dir / f"recent_matches_{league_name}_{league_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Sauve: {output_file} ({len(data['response'])} matchs)")
            return data['response']

        return []

    def get_upcoming_matches_all(self, league_id, league_name, next_days=14):
        """Recuperer tous les prochains matchs d'une ligue"""
        self.logger.info(f"Collecte prochains matchs - {league_name}")

        start_date = datetime.now()
        end_date = start_date + timedelta(days=next_days)

        data = self.api_request('fixtures', {
            'league': league_id,
            'season': 2025,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d')
        })

        if data and data.get('response'):
            output_file = self.current_dir / f"upcoming_matches_{league_name}_{league_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Sauve: {output_file} ({len(data['response'])} matchs)")
            return data['response']

        return []

    def get_league_teams(self, league_id, league_name):
        """Recuperer toutes les equipes d'une ligue"""
        self.logger.info(f"Collecte equipes - {league_name}")

        data = self.api_request('teams', {
            'league': league_id,
            'season': 2025
        })

        if data and data.get('response'):
            output_file = self.current_dir / f"teams_{league_name}_{league_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Sauve: {output_file} ({len(data['response'])} equipes)")
            return data['response']

        return []

    def collect_league_complete_data(self, league_id, league_name):
        """Collecte complete pour une ligue"""
        self.logger.info(f"=== DEBUT COLLECTE COMPLETE: {league_name.upper()} ===")

        results = {
            'league_name': league_name,
            'league_id': league_id,
            'teams_count': 0,
            'recent_matches_count': 0,
            'upcoming_matches_count': 0,
            'topscorers_count': 0,
            'topassists_count': 0,
            'standings_available': False
        }

        try:
            # Equipes
            teams = self.get_league_teams(league_id, league_name)
            results['teams_count'] = len(teams)

            # Classement
            standings = self.get_league_current_standings(league_id, league_name)
            results['standings_available'] = len(standings) > 0

            # Matchs recents
            recent_matches = self.get_recent_matches_all(league_id, league_name)
            results['recent_matches_count'] = len(recent_matches)

            # Prochains matchs
            upcoming_matches = self.get_upcoming_matches_all(league_id, league_name)
            results['upcoming_matches_count'] = len(upcoming_matches)

            # Meilleurs buteurs
            topscorers = self.get_league_topscorers(league_id, league_name)
            results['topscorers_count'] = len(topscorers)

            # Meilleurs passeurs
            topassists = self.get_league_topassists(league_id, league_name)
            results['topassists_count'] = len(topassists)

            self.logger.info(f"=== COLLECTE TERMINEE: {league_name.upper()} ===")

        except Exception as e:
            self.logger.error(f"Erreur collecte {league_name}: {e}")

        return results

    def collect_all_competitions_data(self):
        """Collecte complete pour toutes les competitions"""
        self.logger.info("DEBUT COLLECTE GLOBALE - TOUTES COMPETITIONS")

        all_results = {}

        for league_name, league_id in self.competitions.items():
            try:
                result = self.collect_league_complete_data(league_id, league_name)
                all_results[league_name] = result

                # Petit delai entre ligues
                time.sleep(2)

            except Exception as e:
                self.logger.error(f"Erreur ligue {league_name}: {e}")
                all_results[league_name] = {
                    'error': str(e),
                    'league_name': league_name,
                    'league_id': league_id
                }

        # Sauvegarder resume global
        summary_file = self.current_dir / "collection_summary.json"
        summary_data = {
            'collection_date': datetime.now().isoformat(),
            'total_competitions': len(self.competitions),
            'results': all_results
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Resume global sauve: {summary_file}")

        return all_results

def main():
    print("COLLECTE DONNEES ACTUELLES - TOUTES COMPETITIONS SAISON 2025-2026")
    print("=" * 70)

    collector = CompleteCurrentDataCollector()

    try:
        results = collector.collect_all_competitions_data()

        print("\nRESUME COLLECTE GLOBALE:")
        print("=" * 50)

        total_teams = 0
        total_recent_matches = 0
        total_upcoming_matches = 0
        total_players = 0

        for league_name, result in results.items():
            if 'error' not in result:
                print(f"\n{league_name.upper()}:")
                print(f"  Equipes: {result['teams_count']}")
                print(f"  Matchs recents: {result['recent_matches_count']}")
                print(f"  Prochains matchs: {result['upcoming_matches_count']}")
                print(f"  Buteurs: {result['topscorers_count']}")
                print(f"  Passeurs: {result['topassists_count']}")
                print(f"  Classement: {'Oui' if result['standings_available'] else 'Non'}")

                total_teams += result['teams_count']
                total_recent_matches += result['recent_matches_count']
                total_upcoming_matches += result['upcoming_matches_count']
                total_players += result['topscorers_count'] + result['topassists_count']
            else:
                print(f"\n{league_name.upper()}: ERREUR - {result['error']}")

        print(f"\nTOTAUX:")
        print(f"Equipes: {total_teams}")
        print(f"Matchs recents: {total_recent_matches}")
        print(f"Prochains matchs: {total_upcoming_matches}")
        print(f"Donnees joueurs: {total_players}")

        print(f"\nDonnees sauvees dans: data/current_2025/")

    except Exception as e:
        print(f"ERREUR GLOBALE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()