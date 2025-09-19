"""
COLLECTEUR PRE-MATCH PAR GROUPES HORAIRES
Récupère données spécifiques <1h avant les matchs, groupés par heures similaires
"""

import asyncio
import aiohttp
import pytz
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import defaultdict

from config import Config

class PreMatchCollector:
    """Collecteur de données pré-match intelligent par groupes horaires"""
    
    def __init__(self):
        self.api_key = Config.FOOTBALL_API_KEY
        self.base_url = Config.FOOTBALL_API_BASE_URL
        self.headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }
        
        self.paris_tz = pytz.timezone(Config.TIMEZONE_PARIS)
        self.leagues = Config.TARGET_LEAGUES
        
        # Configuration groupement
        self.max_time_gap_minutes = 60  # Grouper matchs à moins d'1h d'écart
        
        # Dossiers de données
        self.pre_match_data_dir = Path("data/pre_match")
        self.lineups_dir = Path("data/lineups") 
        self.pre_match_data_dir.mkdir(parents=True, exist_ok=True)
        self.lineups_dir.mkdir(parents=True, exist_ok=True)
        
        # État des collectes
        self.scheduled_collections = {}  # {group_id: {matches: [], collection_time: datetime}}
        
        # Logging
        self.setup_logging()
    
    def setup_logging(self):
        """Configuration logging"""
        log_file = Path("logs/pre_match_collector.log")
        log_file.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def group_matches_by_time(self, matches: List[Dict]) -> Dict[str, List[Dict]]:
        """Grouper les matchs par heures similaires (<1h d'écart)"""
        if not matches:
            return {}
        
        # Trier matchs par heure
        sorted_matches = sorted(matches, key=lambda m: m['paris_time'])
        
        groups = {}
        group_id = 0
        
        for match in sorted_matches:
            match_time = match['paris_time']
            assigned_group = None
            
            # Chercher groupe existant compatible
            for gid, group_matches in groups.items():
                group_time = group_matches[0]['paris_time']
                time_diff = abs((match_time - group_time).total_seconds() / 60)
                
                if time_diff <= self.max_time_gap_minutes:
                    assigned_group = gid
                    break
            
            # Assigner à groupe existant ou créer nouveau
            if assigned_group is not None:
                groups[assigned_group].append(match)
            else:
                groups[f"group_{group_id}"] = [match]
                group_id += 1
        
        return groups
    
    async def find_todays_matches_for_pre_collection(self) -> Dict[str, List[Dict]]:
        """Trouver matchs d'aujourd'hui et les grouper par heures"""
        self.logger.info("Recherche matchs d'aujourd'hui pour collecte pré-match")
        
        # Importer le système principal pour récupérer matchs
        from football_prediction_system import FootballPredictionSystem
        prediction_system = FootballPredictionSystem()
        
        # Récupérer tous les matchs d'aujourd'hui
        all_matches = await prediction_system.find_matches_today()
        
        if not all_matches:
            self.logger.info("Aucun match aujourd'hui")
            return {}
        
        # Filtrer matchs dans les prochaines 24h
        now = datetime.now(self.paris_tz)
        upcoming_matches = []
        
        for match in all_matches:
            time_until = match['paris_time'] - now
            hours_until = time_until.total_seconds() / 3600
            
            if 0 < hours_until <= 24:  # Matchs dans les 24 prochaines heures
                upcoming_matches.append(match)
        
        self.logger.info(f"{len(upcoming_matches)} match(s) dans les 24h")
        
        # Grouper par heures similaires
        groups = self.group_matches_by_time(upcoming_matches)
        
        self.logger.info(f"Matchs groupés en {len(groups)} groupe(s):")
        for group_id, group_matches in groups.items():
            times = [m['paris_time'].strftime('%H:%M') for m in group_matches]
            teams = [f"{m['home_team']} vs {m['away_team']}" for m in group_matches]
            self.logger.info(f"  {group_id}: {len(group_matches)} matchs ({min(times)}-{max(times)})")
            for team in teams:
                self.logger.info(f"    - {team}")
        
        return groups
    
    async def schedule_pre_match_collections(self, match_groups: Dict[str, List[Dict]]):
        """Planifier collectes pré-match pour chaque groupe"""
        self.logger.info("Planification collectes pré-match par groupe")
        
        for group_id, matches in match_groups.items():
            # Calculer temps de collecte : 50 minutes avant le premier match du groupe
            first_match_time = min(match['paris_time'] for match in matches)
            collection_time = first_match_time - timedelta(minutes=50)
            
            self.scheduled_collections[group_id] = {
                'matches': matches,
                'collection_time': collection_time,
                'status': 'scheduled'
            }
            
            time_until_collection = collection_time - datetime.now(self.paris_tz)
            
            self.logger.info(f"Groupe {group_id}:")
            self.logger.info(f"  Premier match: {first_match_time.strftime('%H:%M')}")
            self.logger.info(f"  Collecte prévue: {collection_time.strftime('%H:%M')}")
            self.logger.info(f"  Dans: {time_until_collection}")
    
    async def collect_group_pre_match_data(self, group_id: str, matches: List[Dict]):
        """Collecter données pré-match pour un groupe de matchs"""
        self.logger.info(f"=== COLLECTE PRE-MATCH GROUPE {group_id} ===")
        
        group_data = {
            'group_id': group_id,
            'collection_time': datetime.now(self.paris_tz).isoformat(),
            'matches_count': len(matches),
            'matches_data': {}
        }
        
        async with aiohttp.ClientSession() as session:
            for match in matches:
                fixture_id = match['fixture_id']
                match_key = f"{match['home_team']}_vs_{match['away_team']}"
                
                self.logger.info(f"Collecte: {match_key}")
                
                match_pre_data = {
                    'fixture_id': fixture_id,
                    'match_time': match['paris_time'].isoformat(),
                    'lineups': await self._get_real_lineups(session, fixture_id),
                    'injuries': await self._get_real_injuries(session, match['home_id'], match['away_id'], match['league_id']),
                    'head_to_head': await self._get_real_h2h(session, match['home_id'], match['away_id']),
                    'weather': await self._get_weather_data(session, match),
                    'referee': await self._get_referee_info(session, fixture_id)
                }
                
                group_data['matches_data'][match_key] = match_pre_data
                
                # Délai entre matchs
                await asyncio.sleep(1)
        
        # Sauvegarder données du groupe
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        group_file = self.pre_match_data_dir / f"group_{group_id}_{timestamp}.json"
        
        with open(group_file, 'w', encoding='utf-8') as f:
            json.dump(group_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Données groupe sauvées: {group_file}")
        return group_data
    
    async def _get_real_lineups(self, session: aiohttp.ClientSession, fixture_id: int) -> Dict:
        """Récupérer compositions officielles réelles"""
        url = f"{self.base_url}/fixtures/lineups"
        params = {'fixture': fixture_id}
        
        async with session.get(url, headers=self.headers, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get('response'):
                    return {
                        'available': True,
                        'data': data['response']
                    }
            
            return {'available': False, 'reason': f'Status {resp.status}'}
    
    async def _get_real_injuries(self, session: aiohttp.ClientSession, 
                               home_id: int, away_id: int, league_id: int) -> Dict:
        """Récupérer blessures réelles des deux équipes"""
        injuries_data = {'home': [], 'away': []}
        
        for team_id, team_key in [(home_id, 'home'), (away_id, 'away')]:
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
                        injuries_data[team_key] = data['response']
                
                await asyncio.sleep(0.5)
        
        return injuries_data
    
    async def _get_real_h2h(self, session: aiohttp.ClientSession, 
                          home_id: int, away_id: int) -> Dict:
        """Récupérer historique H2H réel"""
        url = f"{self.base_url}/fixtures/headtohead"
        params = {'h2h': f"{home_id}-{away_id}"}
        
        async with session.get(url, headers=self.headers, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get('response'):
                    return {
                        'available': True,
                        'matches': data['response'][:10]  # 10 derniers H2H
                    }
            
            return {'available': False}
    
    async def _get_weather_data(self, session: aiohttp.ClientSession, match: Dict) -> Dict:
        """Récupérer données météo (placeholder - intégration météo API)"""
        # Pour l'instant, placeholder - à intégrer avec API météo
        return {
            'available': False,
            'reason': 'Weather API not integrated yet',
            'placeholder': {
                'temperature': 20,
                'conditions': 'clear',
                'wind_speed': 10
            }
        }
    
    async def _get_referee_info(self, session: aiohttp.ClientSession, fixture_id: int) -> Dict:
        """Récupérer infos arbitre"""
        # L'API Football peut avoir des infos arbitre dans les fixtures
        url = f"{self.base_url}/fixtures"
        params = {'id': fixture_id}
        
        async with session.get(url, headers=self.headers, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get('response'):
                    fixture_data = data['response'][0]
                    referee = fixture_data['fixture'].get('referee')
                    
                    if referee:
                        return {
                            'available': True,
                            'name': referee,
                            'nationality': 'Unknown'  # API Football ne fournit pas toujours
                        }
            
            return {'available': False}
    
    async def run_pre_match_monitoring(self):
        """Exécuter monitoring continu des collectes pré-match"""
        self.logger.info("=== DEMARRAGE MONITORING PRE-MATCH ===")
        
        # Planifier collectes initiales
        match_groups = await self.find_todays_matches_for_pre_collection()
        await self.schedule_pre_match_collections(match_groups)
        
        if not self.scheduled_collections:
            self.logger.info("Aucune collecte pré-match à planifier aujourd'hui")
            return
        
        self.logger.info(f"{len(self.scheduled_collections)} groupe(s) de collecte planifié(s)")
        
        # Boucle de monitoring
        try:
            while self.scheduled_collections:
                now = datetime.now(self.paris_tz)
                
                # Vérifier chaque groupe planifié
                completed_groups = []
                
                for group_id, group_info in self.scheduled_collections.items():
                    collection_time = group_info['collection_time']
                    
                    if now >= collection_time and group_info['status'] == 'scheduled':
                        self.logger.info(f"Déclenchement collecte groupe {group_id}")
                        
                        # Exécuter collecte
                        try:
                            group_info['status'] = 'collecting'
                            await self.collect_group_pre_match_data(group_id, group_info['matches'])
                            group_info['status'] = 'completed'
                            completed_groups.append(group_id)
                            
                        except Exception as e:
                            self.logger.error(f"Erreur collecte groupe {group_id}: {e}")
                            group_info['status'] = 'failed'
                
                # Nettoyer groupes terminés
                for group_id in completed_groups:
                    del self.scheduled_collections[group_id]
                
                # Vérifier toutes les 5 minutes
                await asyncio.sleep(300)
                
        except KeyboardInterrupt:
            self.logger.info("Arrêt monitoring demandé")
        except Exception as e:
            self.logger.error(f"Erreur monitoring: {e}")
            raise
    
    def get_collection_status(self) -> Dict:
        """Obtenir statut des collectes planifiées"""
        now = datetime.now(self.paris_tz)
        
        status = {
            'current_time': now.isoformat(),
            'scheduled_groups': len(self.scheduled_collections),
            'groups_detail': {}
        }
        
        for group_id, group_info in self.scheduled_collections.items():
            collection_time = group_info['collection_time']
            time_until = collection_time - now
            
            status['groups_detail'][group_id] = {
                'matches_count': len(group_info['matches']),
                'collection_time': collection_time.strftime('%H:%M'),
                'time_until_minutes': int(time_until.total_seconds() / 60),
                'status': group_info['status'],
                'matches': [f"{m['home_team']} vs {m['away_team']}" for m in group_info['matches']]
            }
        
        return status
    
    async def emergency_collect_now(self, fixture_ids: List[int] = None):
        """Collecte d'urgence immédiate (pour tests ou situations spéciales)"""
        self.logger.info("=== COLLECTE D'URGENCE ===")
        
        if fixture_ids:
            # Collecte pour fixtures spécifiques
            self.logger.info(f"Collecte pour fixtures: {fixture_ids}")
            
            async with aiohttp.ClientSession() as session:
                for fixture_id in fixture_ids:
                    self.logger.info(f"Collecte urgente fixture {fixture_id}")
                    
                    lineups = await self._get_real_lineups(session, fixture_id)
                    
                    if lineups['available']:
                        self.logger.info("  Lineups disponibles!")
                        for team_lineup in lineups['data']:
                            team_name = team_lineup['team']['name']
                            formation = team_lineup['formation']
                            players_count = len(team_lineup['startXI'])
                            self.logger.info(f"    {team_name}: {formation} ({players_count} joueurs)")
                    else:
                        self.logger.info(f"  Lineups non disponibles: {lineups.get('reason', 'Inconnu')}")
        else:
            # Collecte pour tous les matchs actuels
            match_groups = await self.find_todays_matches_for_pre_collection()
            
            for group_id, matches in match_groups.items():
                await self.collect_group_pre_match_data(group_id, matches)


async def main():
    """Fonction principale"""
    collector = PreMatchCollector()
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'emergency':
        # Mode collecte d'urgence
        await collector.emergency_collect_now()
    else:
        # Mode monitoring normal
        await collector.run_pre_match_monitoring()


if __name__ == "__main__":
    asyncio.run(main())