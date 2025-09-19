"""
PLANIFICATEUR AUTOMATIQUE DE DONNEES PRE-MATCH
Collecte automatique des données et ré-entraînement des modèles selon timing des matchs
"""

import asyncio
import schedule
import time
import pytz
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from pathlib import Path

from football_prediction_system import FootballPredictionSystem
from config import Config

class AutoDataScheduler:
    """Planificateur automatique pour données pré-match"""
    
    def __init__(self):
        self.prediction_system = FootballPredictionSystem()
        self.paris_tz = pytz.timezone(Config.TIMEZONE_PARIS)
        
        # État des tâches
        self.scheduled_matches = {}  # {fixture_id: {stages_completed: [], next_stage: timestamp}}
        self.retrain_queue = set()   # Modèles à ré-entraîner
        
        # Configuration logging
        self.setup_logging()
        
    def setup_logging(self):
        """Configuration du logging"""
        log_file = Path("logs/auto_scheduler.log")
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
    
    def get_paris_now(self) -> datetime:
        """Heure actuelle Paris"""
        return datetime.now(self.paris_tz)
    
    async def daily_match_scan(self):
        """Scan quotidien des matchs - 06h00 Paris"""
        self.logger.info("=== SCAN QUOTIDIEN DES MATCHS ===")
        current_time = self.get_paris_now()
        
        try:
            # Chercher matchs d'aujourd'hui et demain
            today_matches = await self.prediction_system.find_matches_today()
            
            # Scanner aussi les matchs de demain pour préparation
            tomorrow_matches = []
            try:
                # Modifier temporairement la date pour demain
                tomorrow_date = (current_time + timedelta(days=1)).strftime('%Y-%m-%d')
                # Code pour chercher matchs de demain (similaire à find_matches_today)
                self.logger.info(f"Scan également prévu pour matchs du {tomorrow_date}")
            except Exception as e:
                self.logger.warning(f"Erreur scan matchs demain: {e}")
            
            all_matches = today_matches + tomorrow_matches
            
            if not all_matches:
                self.logger.info("Aucun match trouvé aujourd'hui/demain")
                return
            
            self.logger.info(f"{len(all_matches)} match(s) détecté(s)")
            
            # Planifier collecte données pour chaque match
            for match in all_matches:
                await self.schedule_match_data_collection(match)
                
        except Exception as e:
            self.logger.error(f"Erreur scan quotidien: {e}")
    
    async def schedule_match_data_collection(self, match_info: Dict):
        """Planifier collecte données pour un match"""
        fixture_id = match_info['fixture_id']
        match_time = match_info['paris_time']
        
        self.logger.info(f"Planification: {match_info['home_team']} vs {match_info['away_team']} ({match_time.strftime('%H:%M')})")
        
        # Calculer moments de collecte
        stages = {
            'EARLY': match_time - timedelta(hours=48),   # 48h avant - stats générales
            'MID': match_time - timedelta(hours=24),     # 24h avant - formations probables  
            'LATE': match_time - timedelta(hours=2),     # 2h avant - lineups officiels
            'FINAL': match_time - timedelta(minutes=30)  # 30min avant - données finales
        }
        
        self.scheduled_matches[fixture_id] = {
            'match_info': match_info,
            'stages': stages,
            'stages_completed': [],
            'data_collected': {}
        }
        
        # Planifier chaque étape si elle n'est pas déjà passée
        now = self.get_paris_now()
        for stage_name, stage_time in stages.items():
            if stage_time > now:
                delay_seconds = (stage_time - now).total_seconds()
                self.logger.info(f"  {stage_name}: dans {delay_seconds/3600:.1f}h ({stage_time.strftime('%H:%M')})")
    
    async def collect_match_data_stage(self, fixture_id: str, stage: str):
        """Collecter données pour étape spécifique d'un match"""
        if fixture_id not in self.scheduled_matches:
            self.logger.error(f"Match {fixture_id} non trouvé dans planning")
            return
        
        match_data = self.scheduled_matches[fixture_id]
        match_info = match_data['match_info']
        
        self.logger.info(f"=== COLLECTE {stage}: {match_info['home_team']} vs {match_info['away_team']} ===")
        
        try:
            # Collecter données selon l'étape
            collected_data = await self._collect_stage_specific_data(match_info, stage)
            
            # Sauvegarder données
            match_data['data_collected'][stage] = {
                'timestamp': self.get_paris_now(),
                'data': collected_data
            }
            match_data['stages_completed'].append(stage)
            
            # Vérifier si ré-entraînement nécessaire
            if self._should_retrain_models(collected_data, stage):
                self.retrain_queue.add(match_info['league_id'])
                self.logger.info(f"Ré-entraînement programmé pour ligue {match_info['league_id']}")
            
            self.logger.info(f"Collecte {stage} terminée avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur collecte {stage} pour {fixture_id}: {e}")
    
    async def _collect_stage_specific_data(self, match_info: Dict, stage: str) -> Dict:
        """Collecter données spécifiques à l'étape"""
        
        if stage == "EARLY":
            # Stats équipes, historique, classement
            home_stats = await self.prediction_system.get_team_current_stats(
                match_info['home_id'], match_info['league_id']
            )
            away_stats = await self.prediction_system.get_team_current_stats(
                match_info['away_id'], match_info['league_id'] 
            )
            
            return {
                'home_stats': home_stats,
                'away_stats': away_stats,
                'stage_quality': 'basic'
            }
            
        elif stage == "MID":
            # Formations probables, blessures, suspensions
            return {
                'probable_formations': await self._get_probable_formations(match_info),
                'injury_reports': await self._get_injury_reports(match_info),
                'motivation_factors': await self._analyze_motivation(match_info),
                'stage_quality': 'enhanced'
            }
            
        elif stage == "LATE":
            # Lineups officiels, météo, arbitre
            return {
                'official_lineups': await self._get_official_lineups(match_info['fixture_id']),
                'weather_conditions': await self._get_weather_data(match_info),
                'referee_info': await self._get_referee_info(match_info['fixture_id']),
                'stage_quality': 'complete'
            }
            
        elif stage == "FINAL":
            # Dernières informations, ajustements
            return {
                'final_adjustments': await self._get_final_adjustments(match_info),
                'betting_odds': await self._get_current_odds(match_info['fixture_id']),
                'stage_quality': 'premium'
            }
        
        return {}
    
    async def _get_probable_formations(self, match_info: Dict) -> Dict:
        """Récupérer formations probables"""
        # Intégration avec API ou sources spécialisées
        return {
            'home': {'formation': '4-3-3', 'confidence': 0.8},
            'away': {'formation': '4-4-2', 'confidence': 0.7}
        }
    
    async def _get_injury_reports(self, match_info: Dict) -> Dict:
        """Récupérer rapports blessures"""
        # Intégration avec API Football pour blessures
        return {
            'home': [],
            'away': []
        }
    
    async def _get_official_lineups(self, fixture_id: str) -> Dict:
        """Récupérer compositions officielles"""
        # API Football - lineups endpoint
        return {
            'home': {'formation': '4-3-3', 'players': []},
            'away': {'formation': '4-4-2', 'players': []}
        }
    
    async def _get_weather_data(self, match_info: Dict) -> Dict:
        """Récupérer données météo"""
        # Intégration API météo (OpenWeather, etc.)
        return {
            'temperature': 18,
            'rain_probability': 10,
            'wind_speed': 8
        }
    
    async def _get_referee_info(self, fixture_id: str) -> Dict:
        """Récupérer infos arbitre"""
        return {
            'name': 'Unknown',
            'nationality': 'Unknown',
            'strictness': 0.5
        }
    
    async def _analyze_motivation(self, match_info: Dict) -> Dict:
        """Analyser facteurs motivation"""
        return {
            'importance_level': 'normal',
            'rivalry': False,
            'stakes': 'league_points'
        }
    
    async def _get_final_adjustments(self, match_info: Dict) -> Dict:
        """Ajustements finaux"""
        return {
            'last_minute_news': [],
            'lineup_changes': []
        }
    
    async def _get_current_odds(self, fixture_id: str) -> Dict:
        """Cotes actuelles"""
        return {
            'home_win': 2.10,
            'draw': 3.20,
            'away_win': 3.50
        }
    
    def _should_retrain_models(self, collected_data: Dict, stage: str) -> bool:
        """Déterminer si ré-entraînement nécessaire"""
        # Critères pour déclencher ré-entraînement
        if stage == "LATE" and collected_data.get('stage_quality') == 'complete':
            # Si on a des données complètes, on peut envisager ré-entraînement
            return True
        return False
    
    async def retrain_models_if_needed(self):
        """Ré-entraîner modèles si nécessaire"""
        if not self.retrain_queue:
            return
        
        self.logger.info(f"=== RE-ENTRAINEMENT MODELES ===")
        self.logger.info(f"Ligues à traiter: {list(self.retrain_queue)}")
        
        for league_id in list(self.retrain_queue):
            try:
                await self._retrain_league_models(league_id)
                self.retrain_queue.remove(league_id)
                self.logger.info(f"Ré-entraînement ligue {league_id} terminé")
                
            except Exception as e:
                self.logger.error(f"Erreur ré-entraînement ligue {league_id}: {e}")
    
    async def _retrain_league_models(self, league_id: int):
        """Ré-entraîner modèles d'une ligue"""
        self.logger.info(f"Ré-entraînement modèles ligue {league_id}")
        
        # Ici on intégrerait le code de ré-entraînement
        # - Charger nouvelles données
        # - Re-entraîner modèles
        # - Sauvegarder nouveaux modèles
        # - Recharger dans le système de prédiction
        
        # Simulation pour le moment
        await asyncio.sleep(2)
        
        # Recharger modèles dans le système
        self.prediction_system.load_existing_models()
    
    def save_state(self):
        """Sauvegarder état du planificateur"""
        state_file = Path("data/scheduler_state.json")
        state_file.parent.mkdir(exist_ok=True)
        
        # Préparer données à sauvegarder
        serializable_state = {}
        for fixture_id, match_data in self.scheduled_matches.items():
            serializable_state[fixture_id] = {
                'stages_completed': match_data['stages_completed'],
                'match_info': {
                    'home_team': match_data['match_info']['home_team'],
                    'away_team': match_data['match_info']['away_team'],
                    'league_name': match_data['match_info']['league_name'],
                    'paris_time': match_data['match_info']['paris_time'].isoformat()
                }
            }
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump({
                'scheduled_matches': serializable_state,
                'retrain_queue': list(self.retrain_queue),
                'last_save': self.get_paris_now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"État sauvegardé: {state_file}")
    
    def load_state(self):
        """Charger état du planificateur"""
        state_file = Path("data/scheduler_state.json")
        
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.retrain_queue = set(state.get('retrain_queue', []))
            self.logger.info(f"État chargé depuis {state_file}")
            
        except Exception as e:
            self.logger.error(f"Erreur chargement état: {e}")
    
    def setup_daily_schedule(self):
        """Configuration planning quotidien"""
        # Scan quotidien à 06h00 Paris
        schedule.every().day.at("06:00").do(lambda: asyncio.run(self.daily_match_scan()))
        
        # Ré-entraînement si nécessaire à 03h00 Paris  
        schedule.every().day.at("03:00").do(lambda: asyncio.run(self.retrain_models_if_needed()))
        
        # Sauvegarde état toutes les heures
        schedule.every().hour.do(self.save_state)
        
        self.logger.info("Planning quotidien configuré:")
        self.logger.info("  06:00 - Scan matchs du jour")
        self.logger.info("  03:00 - Ré-entraînement modèles")
        self.logger.info("  Toutes les heures - Sauvegarde état")
    
    async def run_scheduler(self):
        """Exécuter le planificateur en continu"""
        self.logger.info("=== DEMARRAGE AUTO-PLANIFICATEUR ===")
        
        # Charger état précédent
        self.load_state()
        
        # Configurer planning
        self.setup_daily_schedule()
        
        # Exécution immédiate du scan pour test
        await self.daily_match_scan()
        
        # Boucle principale
        self.logger.info("Planificateur démarré - CTRL+C pour arrêter")
        
        try:
            while True:
                schedule.run_pending()
                await asyncio.sleep(60)  # Vérifier toutes les minutes
                
        except KeyboardInterrupt:
            self.logger.info("Arrêt planificateur demandé")
            self.save_state()
        except Exception as e:
            self.logger.error(f"Erreur planificateur: {e}")
            self.save_state()
            raise


# Point d'entrée
async def main():
    """Fonction principale"""
    scheduler = AutoDataScheduler()
    await scheduler.run_scheduler()


if __name__ == "__main__":
    # Créer dossiers nécessaires
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nArrêt utilisateur")
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()