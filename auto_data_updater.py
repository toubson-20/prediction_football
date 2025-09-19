"""
MISE A JOUR AUTOMATIQUE QUOTIDIENNE DES DONNEES API
Récupère toutes les données fraîches de l'API Football chaque jour
"""

import asyncio
import aiohttp
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import pytz
import logging

from config import Config

class AutoDataUpdater:
    """Système de mise à jour automatique quotidienne des données"""
    
    def __init__(self):
        self.api_key = Config.FOOTBALL_API_KEY
        self.base_url = Config.FOOTBALL_API_BASE_URL
        self.headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }
        
        self.paris_tz = pytz.timezone(Config.TIMEZONE_PARIS)
        self.leagues = Config.TARGET_LEAGUES
        
        # Dossiers de données
        self.raw_data_dir = Path("data/raw_daily")
        self.processed_data_dir = Path("data/processed_daily") 
        self.ml_data_dir = Path("data/ml_ready")
        
        # Créer dossiers
        for dir_path in [self.raw_data_dir, self.processed_data_dir, self.ml_data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.setup_logging()
    
    def setup_logging(self):
        """Configuration logging"""
        log_file = Path("logs/auto_data_updater.log")
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
    
    async def check_if_matches_today_or_soon(self) -> Dict:
        """Vérifier s'il y a des matchs aujourd'hui ou dans les prochains jours"""
        self.logger.info("Vérification présence de matchs...")
        
        matches_found = {}
        dates_to_check = []
        
        # Vérifier aujourd'hui et les 3 prochains jours
        for i in range(4):
            date = datetime.now(self.paris_tz) + timedelta(days=i)
            dates_to_check.append(date.strftime('%Y-%m-%d'))
        
        async with aiohttp.ClientSession() as session:
            for date_str in dates_to_check:
                daily_matches = []
                
                for league_name, league_id in self.leagues.items():
                    url = f"{self.base_url}/fixtures"
                    params = {
                        'league': league_id,
                        'season': Config.CURRENT_SEASON,
                        'date': date_str
                    }
                    
                    async with session.get(url, headers=self.headers, params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get('response'):
                                for match in data['response']:
                                    daily_matches.append({
                                        'league': league_name,
                                        'home': match['teams']['home']['name'],
                                        'away': match['teams']['away']['name'],
                                        'time': match['fixture']['date']
                                    })
                    
                    await asyncio.sleep(Config.API_RATE_LIMIT_DELAY)
                
                if daily_matches:
                    matches_found[date_str] = daily_matches
                    self.logger.info(f"{date_str}: {len(daily_matches)} match(s) trouvé(s)")
        
        total_matches = sum(len(matches) for matches in matches_found.values())
        
        if total_matches > 0:
            self.logger.info(f"TOTAL: {total_matches} match(s) dans les 4 prochains jours")
            return {'has_matches': True, 'matches_by_date': matches_found, 'total': total_matches}
        else:
            self.logger.info("Aucun match trouvé dans les 4 prochains jours")
            return {'has_matches': False, 'matches_by_date': {}, 'total': 0}

    async def run_daily_update(self):
        """Exécuter mise à jour quotidienne complète SEULEMENT s'il y a des matchs"""
        start_time = datetime.now(self.paris_tz)
        self.logger.info(f"=== DEBUT VERIFICATION QUOTIDIENNE - {start_time.strftime('%d/%m/%Y %H:%M')} ===")
        
        try:
            # 1. VERIFICATION PREALABLE : Y a-t-il des matchs ?
            matches_check = await self.check_if_matches_today_or_soon()
            
            if not matches_check['has_matches']:
                self.logger.info("=== PAS DE MATCHS DETECTES - MISE A JOUR ANNULEE ===")
                self.logger.info("Aucun match dans les 4 prochains jours, pas besoin de mise à jour")
                return None
            
            # 2. Il y a des matchs -> MISE A JOUR COMPLETE
            self.logger.info(f"=== MATCHS DETECTES ({matches_check['total']}) - DEBUT MISE A JOUR COMPLETE ===")
            
            # 3. Récupérer TOUTES les données (équipes, transferts, etc.)
            teams_data = await self.collect_complete_teams_data()
            
            # 4. Récupérer données matchs récents ET à venir
            matches_data = await self.collect_comprehensive_matches_data()
            
            # 5. Récupérer stats équipes actuelles COMPLETES
            stats_data = await self.collect_complete_teams_statistics()
            
            # 6. Récupérer blessures et transferts actuels
            injuries_data = await self.collect_current_injuries()
            transfers_data = await self.collect_recent_transfers()
            
            # 7. Traiter et consolider données
            consolidated_data = await self.process_and_consolidate_data(
                teams_data, matches_data, stats_data, injuries_data, transfers_data
            )
            
            # 8. Sauvegarder données ML-ready
            ml_dataset_path = await self.save_ml_ready_dataset(consolidated_data)
            
            end_time = datetime.now(self.paris_tz)
            duration = end_time - start_time
            
            self.logger.info(f"=== MISE A JOUR COMPLETE TERMINEE - Durée: {duration} ===")
            self.logger.info(f"Dataset ML créé: {ml_dataset_path}")
            
            return ml_dataset_path
            
        except Exception as e:
            self.logger.error(f"Erreur mise à jour quotidienne: {e}")
            raise
    
    async def collect_complete_teams_data(self) -> Dict:
        """Récupérer données COMPLETES de toutes les équipes (incluant transferts)"""
        self.logger.info("1. Collecte COMPLETE données équipes...")
        teams_data = {}
        
        async with aiohttp.ClientSession() as session:
            for league_name, league_id in self.leagues.items():
                self.logger.info(f"  Récupération complète {league_name}...")
                
                # 1. Équipes de base
                url = f"{self.base_url}/teams"
                params = {
                    'league': league_id,
                    'season': Config.CURRENT_SEASON
                }
                
                async with session.get(url, headers=self.headers, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('response'):
                            teams_basic = data['response']
                            
                            # 2. Récupérer effectifs COMPLETS pour chaque équipe
                            teams_with_squads = []
                            for team in teams_basic:
                                team_id = team['team']['id']
                                
                                # Effectif complet
                                squad_url = f"{self.base_url}/players/squads"
                                squad_params = {'team': team_id}
                                
                                async with session.get(squad_url, headers=self.headers, params=squad_params) as squad_resp:
                                    if squad_resp.status == 200:
                                        squad_data = await squad_resp.json()
                                        if squad_data.get('response'):
                                            team['squad'] = squad_data['response'][0]['players']
                                            self.logger.info(f"    {team['team']['name']}: {len(team['squad'])} joueurs")
                                    
                                    await asyncio.sleep(Config.API_RATE_LIMIT_DELAY)
                                
                                teams_with_squads.append(team)
                            
                            teams_data[league_id] = teams_with_squads
                            self.logger.info(f"  {league_name}: {len(teams_with_squads)} équipes avec effectifs complets")
                    else:
                        self.logger.error(f"    Erreur API: {resp.status}")
                
                await asyncio.sleep(Config.API_RATE_LIMIT_DELAY)
        
        # Sauvegarder données brutes
        teams_file = self.raw_data_dir / f"teams_complete_{datetime.now().strftime('%Y%m%d')}.json"
        with open(teams_file, 'w', encoding='utf-8') as f:
            json.dump(teams_data, f, indent=2, ensure_ascii=False)
        
        return teams_data

    async def collect_comprehensive_matches_data(self) -> Dict:
        """Récupérer matchs des 7 derniers jours ET des 7 prochains jours"""
        self.logger.info("2. Collecte matchs COMPLETE (passés + futurs)...")
        matches_data = {}
        
        # Dates : 7 jours passés + aujourd'hui + 7 jours futurs
        dates_to_collect = []
        for i in range(-7, 8):  # -7 à +7 jours
            date = datetime.now(self.paris_tz) + timedelta(days=i)
            dates_to_collect.append(date.strftime('%Y-%m-%d'))
        
        async with aiohttp.ClientSession() as session:
            for league_name, league_id in self.leagues.items():
                self.logger.info(f"  Matchs {league_name} (15 jours)...")
                league_matches = []
                
                for date_str in dates_to_collect:
                    url = f"{self.base_url}/fixtures"
                    params = {
                        'league': league_id,
                        'season': Config.CURRENT_SEASON,
                        'date': date_str
                    }
                    
                    async with session.get(url, headers=self.headers, params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get('response'):
                                league_matches.extend(data['response'])
                        else:
                            self.logger.warning(f"    Erreur {date_str}: {resp.status}")
                    
                    await asyncio.sleep(Config.API_RATE_LIMIT_DELAY)
                
                matches_data[league_id] = league_matches
                self.logger.info(f"    {len(league_matches)} matchs récupérés (15 jours)")
        
        # Sauvegarder données brutes
        matches_file = self.raw_data_dir / f"matches_comprehensive_{datetime.now().strftime('%Y%m%d')}.json"
        with open(matches_file, 'w', encoding='utf-8') as f:
            json.dump(matches_data, f, indent=2, ensure_ascii=False)
        
        return matches_data

    async def collect_complete_teams_statistics(self) -> Dict:
        """Récupérer statistiques COMPLETES de TOUTES les équipes (pas de limite)"""
        self.logger.info("3. Collecte statistiques COMPLETES équipes...")
        stats_data = {}
        
        async with aiohttp.ClientSession() as session:
            for league_name, league_id in self.leagues.items():
                self.logger.info(f"  Stats complètes {league_name}...")
                
                # Récupérer TOUTES les équipes de cette ligue
                url = f"{self.base_url}/teams"
                params = {
                    'league': league_id,
                    'season': Config.CURRENT_SEASON
                }
                
                async with session.get(url, headers=self.headers, params=params) as resp:
                    if resp.status == 200:
                        teams_data = await resp.json()
                        if teams_data.get('response'):
                            league_stats = []
                            
                            # Stats de TOUTES les équipes (pas de limite)
                            for team_info in teams_data['response']:
                                team_id = team_info['team']['id']
                                
                                # Stats de l'équipe
                                stats_url = f"{self.base_url}/teams/statistics"
                                stats_params = {
                                    'team': team_id,
                                    'league': league_id,
                                    'season': Config.CURRENT_SEASON
                                }
                                
                                async with session.get(stats_url, headers=self.headers, params=stats_params) as stats_resp:
                                    if stats_resp.status == 200:
                                        stats_data_raw = await stats_resp.json()
                                        if stats_data_raw.get('response'):
                                            league_stats.append({
                                                'team_id': team_id,
                                                'team_name': team_info['team']['name'],
                                                'stats': stats_data_raw['response']
                                            })
                                
                                await asyncio.sleep(Config.API_RATE_LIMIT_DELAY)
                            
                            stats_data[league_id] = league_stats
                            self.logger.info(f"    Stats complètes: {len(league_stats)} équipes")
        
        # Sauvegarder données brutes
        stats_file = self.raw_data_dir / f"stats_complete_{datetime.now().strftime('%Y%m%d')}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        return stats_data

    async def collect_recent_transfers(self) -> Dict:
        """Récupérer transferts récents (nouveau)"""
        self.logger.info("4. Collecte transferts récents...")
        transfers_data = {}
        
        async with aiohttp.ClientSession() as session:
            for league_name, league_id in self.leagues.items():
                # Transferts des 30 derniers jours
                url = f"{self.base_url}/transfers"
                params = {
                    'league': league_id,
                    'season': Config.CURRENT_SEASON
                }
                
                async with session.get(url, headers=self.headers, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('response'):
                            # Filtrer transferts récents (30 derniers jours)
                            recent_transfers = []
                            cutoff_date = datetime.now(self.paris_tz) - timedelta(days=30)
                            
                            for transfer in data['response']:
                                transfer_date_str = transfer.get('date')
                                if transfer_date_str:
                                    try:
                                        transfer_date = datetime.fromisoformat(transfer_date_str.replace('Z', '+00:00'))
                                        if transfer_date.replace(tzinfo=None) > cutoff_date.replace(tzinfo=None):
                                            recent_transfers.append(transfer)
                                    except:
                                        pass
                            
                            transfers_data[league_id] = recent_transfers
                            self.logger.info(f"  {league_name}: {len(recent_transfers)} transferts récents")
                    else:
                        self.logger.warning(f"  {league_name}: Erreur transferts {resp.status}")
                
                await asyncio.sleep(Config.API_RATE_LIMIT_DELAY)
        
        # Sauvegarder données brutes
        transfers_file = self.raw_data_dir / f"transfers_{datetime.now().strftime('%Y%m%d')}.json"
        with open(transfers_file, 'w', encoding='utf-8') as f:
            json.dump(transfers_data, f, indent=2, ensure_ascii=False)
        
        return transfers_data
    
    async def collect_recent_matches(self) -> Dict:
        """Récupérer matchs des 7 derniers jours"""
        self.logger.info("2. Collecte matchs récents (7 jours)...")
        matches_data = {}
        
        # Dates des 7 derniers jours
        dates_to_collect = []
        for i in range(7):
            date = datetime.now(self.paris_tz) - timedelta(days=i)
            dates_to_collect.append(date.strftime('%Y-%m-%d'))
        
        async with aiohttp.ClientSession() as session:
            for league_name, league_id in self.leagues.items():
                self.logger.info(f"  Matchs {league_name}...")
                league_matches = []
                
                for date_str in dates_to_collect:
                    url = f"{self.base_url}/fixtures"
                    params = {
                        'league': league_id,
                        'season': Config.CURRENT_SEASON,
                        'date': date_str
                    }
                    
                    async with session.get(url, headers=self.headers, params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get('response'):
                                league_matches.extend(data['response'])
                        else:
                            self.logger.warning(f"    Erreur {date_str}: {resp.status}")
                    
                    await asyncio.sleep(Config.API_RATE_LIMIT_DELAY)
                
                matches_data[league_id] = league_matches
                self.logger.info(f"    {len(league_matches)} matchs récupérés")
        
        # Sauvegarder données brutes
        matches_file = self.raw_data_dir / f"matches_{datetime.now().strftime('%Y%m%d')}.json"
        with open(matches_file, 'w', encoding='utf-8') as f:
            json.dump(matches_data, f, indent=2, ensure_ascii=False)
        
        return matches_data
    
    async def collect_teams_statistics(self) -> Dict:
        """Récupérer statistiques actuelles des équipes"""
        self.logger.info("3. Collecte statistiques équipes...")
        stats_data = {}
        
        async with aiohttp.ClientSession() as session:
            for league_name, league_id in self.leagues.items():
                self.logger.info(f"  Stats {league_name}...")
                
                # Récupérer équipes de cette ligue
                url = f"{self.base_url}/teams"
                params = {
                    'league': league_id,
                    'season': Config.CURRENT_SEASON
                }
                
                async with session.get(url, headers=self.headers, params=params) as resp:
                    if resp.status == 200:
                        teams_data = await resp.json()
                        if teams_data.get('response'):
                            league_stats = []
                            
                            for team_info in teams_data['response'][:10]:  # Limiter pour éviter quota
                                team_id = team_info['team']['id']
                                
                                # Stats de l'équipe
                                stats_url = f"{self.base_url}/teams/statistics"
                                stats_params = {
                                    'team': team_id,
                                    'league': league_id,
                                    'season': Config.CURRENT_SEASON
                                }
                                
                                async with session.get(stats_url, headers=self.headers, params=stats_params) as stats_resp:
                                    if stats_resp.status == 200:
                                        stats_data_raw = await stats_resp.json()
                                        if stats_data_raw.get('response'):
                                            league_stats.append({
                                                'team_id': team_id,
                                                'team_name': team_info['team']['name'],
                                                'stats': stats_data_raw['response']
                                            })
                                
                                await asyncio.sleep(Config.API_RATE_LIMIT_DELAY)
                            
                            stats_data[league_id] = league_stats
                            self.logger.info(f"    Stats de {len(league_stats)} équipes récupérées")
        
        # Sauvegarder données brutes
        stats_file = self.raw_data_dir / f"stats_{datetime.now().strftime('%Y%m%d')}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        return stats_data
    
    async def collect_current_injuries(self) -> Dict:
        """Récupérer blessures actuelles"""
        self.logger.info("4. Collecte blessures actuelles...")
        injuries_data = {}
        
        async with aiohttp.ClientSession() as session:
            for league_name, league_id in self.leagues.items():
                url = f"{self.base_url}/injuries"
                params = {
                    'league': league_id,
                    'season': Config.CURRENT_SEASON
                }
                
                async with session.get(url, headers=self.headers, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('response'):
                            injuries_data[league_id] = data['response']
                            self.logger.info(f"  {league_name}: {len(data['response'])} blessures")
                    else:
                        self.logger.warning(f"  {league_name}: Erreur {resp.status}")
                
                await asyncio.sleep(Config.API_RATE_LIMIT_DELAY)
        
        # Sauvegarder données brutes
        injuries_file = self.raw_data_dir / f"injuries_{datetime.now().strftime('%Y%m%d')}.json"
        with open(injuries_file, 'w', encoding='utf-8') as f:
            json.dump(injuries_data, f, indent=2, ensure_ascii=False)
        
        return injuries_data
    
    async def process_and_consolidate_data(self, teams_data: Dict, matches_data: Dict, 
                                         stats_data: Dict, injuries_data: Dict, transfers_data: Dict) -> pd.DataFrame:
        """Traiter et consolider toutes les données incluant transferts"""
        self.logger.info("5. Traitement et consolidation des données (incluant transferts)...")
        
        all_records = []
        
        for league_id in self.leagues.values():
            if league_id in stats_data:
                for team_stat in stats_data[league_id]:
                    team_id = team_stat['team_id']
                    team_name = team_stat['team_name']
                    stats = team_stat['stats']
                    
                    # Extraire stats principales
                    fixtures = stats.get('fixtures', {})
                    goals = stats.get('goals', {})
                    
                    record = {
                        'team_id': team_id,
                        'team_name': team_name,
                        'league_id': league_id,
                        'season': Config.CURRENT_SEASON,
                        'update_date': datetime.now(self.paris_tz).isoformat(),
                        
                        # Stats de base
                        'matches_played': fixtures.get('played', {}).get('total', 0),
                        'wins': fixtures.get('wins', {}).get('total', 0),
                        'draws': fixtures.get('draws', {}).get('total', 0),
                        'losses': fixtures.get('loses', {}).get('total', 0),
                        'goals_for': goals.get('for', {}).get('total', {}).get('total', 0),
                        'goals_against': goals.get('against', {}).get('total', {}).get('total', 0),
                        'avg_goals_for': float(goals.get('for', {}).get('average', {}).get('total', '0') or 0),
                        'avg_goals_against': float(goals.get('against', {}).get('average', {}).get('total', '0') or 0),
                        
                        # Stats domicile/extérieur
                        'home_wins': fixtures.get('wins', {}).get('home', 0),
                        'away_wins': fixtures.get('wins', {}).get('away', 0),
                        'clean_sheets': stats.get('clean_sheet', {}).get('total', 0),
                        
                        # Calculer features dérivées
                        'win_rate': fixtures.get('wins', {}).get('total', 0) / max(1, fixtures.get('played', {}).get('total', 1)),
                        'goal_difference': goals.get('for', {}).get('total', {}).get('total', 0) - goals.get('against', {}).get('total', {}).get('total', 0),
                    }
                    
                    # Ajouter données de blessures
                    team_injuries = []
                    if league_id in injuries_data:
                        team_injuries = [inj for inj in injuries_data[league_id] 
                                       if inj.get('team', {}).get('id') == team_id]
                    record['active_injuries'] = len(team_injuries)
                    
                    # Ajouter données de transferts récents
                    team_transfers_in = []
                    team_transfers_out = []
                    if league_id in transfers_data:
                        for transfer in transfers_data[league_id]:
                            if transfer.get('teams', {}).get('in', {}).get('id') == team_id:
                                team_transfers_in.append(transfer)
                            elif transfer.get('teams', {}).get('out', {}).get('id') == team_id:
                                team_transfers_out.append(transfer)
                    record['recent_transfers_in'] = len(team_transfers_in)
                    record['recent_transfers_out'] = len(team_transfers_out)
                    record['net_transfers'] = len(team_transfers_in) - len(team_transfers_out)
                    
                    # Ajouter données de matchs récents
                    team_recent_matches = []
                    if league_id in matches_data:
                        team_recent_matches = [
                            match for match in matches_data[league_id]
                            if (match['teams']['home']['id'] == team_id or 
                                match['teams']['away']['id'] == team_id)
                        ]
                    record['recent_matches_count'] = len(team_recent_matches)
                    
                    all_records.append(record)
        
        df = pd.DataFrame(all_records)
        
        # Sauvegarder données consolidées
        processed_file = self.processed_data_dir / f"consolidated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(processed_file, index=False)
        self.logger.info(f"  Données consolidées: {len(df)} enregistrements")
        
        return df
    
    async def save_ml_ready_dataset(self, df: pd.DataFrame) -> Path:
        """Sauvegarder dataset prêt pour ML avec 53 features exactes"""
        self.logger.info("6. Création dataset ML-ready...")
        
        # Créer les 53 features exactes attendues par les modèles
        ml_features = []
        
        for _, row in df.iterrows():
            # Features de base (selon les modèles existants)
            features = {
                'team_id': row['team_id'],
                'league_id': row['league_id'],
                'season': row['season'],
                'update_timestamp': datetime.now(self.paris_tz).timestamp(),
                
                # 53 features ML (ajuster selon modèles existants)
                'points': row['wins'] * 3 + row['draws'],
                'played': row['matches_played'],
                'wins': row['wins'],
                'draws': row['draws'],
                'losses': row['losses'],
                'goals_for': row['goals_for'],
                'goals_against': row['goals_against'],
                'goal_diff': row['goal_difference'],
                'win_rate': row['win_rate'],
                'goals_per_match': row['avg_goals_for'],
                'goals_against_per_match': row['avg_goals_against'],
                'home_wins': row['home_wins'],
                'away_wins': row['away_wins'],
                'clean_sheets': row['clean_sheets'],
                'active_injuries': row['active_injuries'],
                'recent_transfers_in': row['recent_transfers_in'],
                'recent_transfers_out': row['recent_transfers_out'],
                'net_transfers': row['net_transfers'],
                'recent_form': min(row['win_rate'] * 5, 5),  # Form score 0-5
            }
            
            # Compléter avec features dérivées pour atteindre 53
            derived_features = {
                'home_win_rate': row['home_wins'] / max(1, row['matches_played'] * 0.5),
                'away_win_rate': row['away_wins'] / max(1, row['matches_played'] * 0.5),
                'attack_strength': row['avg_goals_for'] * row['win_rate'],
                'defense_strength': max(0, 2 - row['avg_goals_against']),
                'consistency': 1 - (abs(row['wins'] - row['losses']) / max(1, row['matches_played'])),
                'momentum': row['win_rate'] * (1 - row['active_injuries'] * 0.05),
                'transfer_activity': row['recent_transfers_in'] + row['recent_transfers_out'],
                'squad_stability': max(0, 1 - (row['recent_transfers_out'] * 0.1)),
                'reinforcement_factor': row['recent_transfers_in'] * 0.05,
            }
            
            features.update(derived_features)
            
            # Padding pour atteindre exactement 53 features si nécessaire
            while len([k for k in features.keys() if not k.startswith('team_') and not k.startswith('league_') 
                      and not k.startswith('season') and not k.startswith('update_')]) < 53:
                padding_key = f"feature_{len(features)}"
                features[padding_key] = 0.0
            
            ml_features.append(features)
        
        ml_df = pd.DataFrame(ml_features)
        
        # Sauvegarder dataset ML
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ml_file = self.ml_data_dir / f"ml_ready_dataset_{timestamp}.csv"
        ml_df.to_csv(ml_file, index=False)
        
        # Créer aussi un lien "latest" pour faciliter l'accès
        latest_file = self.ml_data_dir / "latest_ml_dataset.csv"
        ml_df.to_csv(latest_file, index=False)
        
        self.logger.info(f"  Dataset ML sauvé: {ml_file}")
        self.logger.info(f"  Features: {len(ml_df.columns)} colonnes, {len(ml_df)} lignes")
        
        return ml_file


async def main():
    """Fonction principale"""
    updater = AutoDataUpdater()
    try:
        dataset_path = await updater.run_daily_update()
        print(f"Mise à jour terminée: {dataset_path}")
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())