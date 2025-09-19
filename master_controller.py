"""
CONTROLEUR PRINCIPAL DU SYSTEME DE PREDICTION FOOTBALL
Orchestre tous les composants : mise à jour, ré-entraînement, collecte pré-match
"""

import asyncio
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
import pytz
import logging

from config import Config
from auto_data_updater import AutoDataUpdater
from auto_model_trainer import AutoModelTrainer  
from pre_match_collector import PreMatchCollector
from football_prediction_system import FootballPredictionSystem

class MasterController:
    """Contrôleur principal orchestrant tous les systèmes"""
    
    def __init__(self):
        self.paris_tz = pytz.timezone(Config.TIMEZONE_PARIS)
        
        # Composants
        self.data_updater = AutoDataUpdater()
        self.model_trainer = AutoModelTrainer()
        self.pre_match_collector = PreMatchCollector()
        self.prediction_system = FootballPredictionSystem()
        
        # État du système
        self.system_state = {
            'last_data_update': None,
            'last_model_retrain': None,
            'active_pre_match_collections': {},
            'system_health': 'unknown'
        }
        
        # Logging
        self.setup_logging()
        self.load_system_state()
    
    def setup_logging(self):
        """Configuration logging principal"""
        log_file = Path("logs/master_controller.log")
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
    
    def load_system_state(self):
        """Charger état système précédent"""
        state_file = Path("data/system_state.json")
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    self.system_state.update(json.load(f))
                self.logger.info(f"État système chargé depuis {state_file}")
            except Exception as e:
                self.logger.warning(f"Erreur chargement état: {e}")
    
    def save_system_state(self):
        """Sauvegarder état système"""
        state_file = Path("data/system_state.json")
        state_file.parent.mkdir(exist_ok=True)
        
        # Préparer état sérialisable
        serializable_state = self.system_state.copy()
        serializable_state['last_save'] = datetime.now(self.paris_tz).isoformat()
        
        with open(state_file, 'w') as f:
            json.dump(serializable_state, f, indent=2)
        
        self.logger.info(f"État système sauvé: {state_file}")
    
    def check_system_health(self) -> str:
        """Vérifier santé générale du système"""
        health_issues = []
        
        # Vérifier API key
        if not Config.FOOTBALL_API_KEY or Config.FOOTBALL_API_KEY == "demo_key":
            health_issues.append("API key Football manquante")
        
        # Vérifier modèles
        models_dir = Path("models/complete_models")
        if not models_dir.exists() or len(list(models_dir.glob("*.joblib"))) < 10:
            health_issues.append("Modèles ML insuffisants")
        
        # Vérifier dernière mise à jour données
        if self.system_state.get('last_data_update'):
            last_update = datetime.fromisoformat(self.system_state['last_data_update'])
            if datetime.now(self.paris_tz) - last_update > timedelta(days=2):
                health_issues.append("Données anciennes (>2 jours)")
        
        # Vérifier dossiers requis
        required_dirs = ["data", "logs", "models"]
        for dir_name in required_dirs:
            if not Path(dir_name).exists():
                health_issues.append(f"Dossier {dir_name} manquant")
        
        if health_issues:
            health = "warning"
            self.logger.warning(f"Problèmes système: {health_issues}")
        else:
            health = "healthy"
            self.logger.info("Système en bonne santé")
        
        self.system_state['system_health'] = health
        self.system_state['health_issues'] = health_issues
        self.system_state['last_health_check'] = datetime.now(self.paris_tz).isoformat()
        
        return health
    
    async def run_complete_daily_cycle(self):
        """Exécuter cycle quotidien complet"""
        cycle_start = datetime.now(self.paris_tz)
        self.logger.info(f"=== DEBUT CYCLE QUOTIDIEN COMPLET - {cycle_start.strftime('%d/%m/%Y %H:%M')} ===")
        
        try:
            # 1. Vérifier santé système
            health = self.check_system_health()
            if health == "warning":
                self.logger.warning("Système en état d'alerte - continuer quand même")
            
            # 2. Mise à jour données API
            self.logger.info("ETAPE 1: Mise à jour données API")
            dataset_path = await self.data_updater.run_daily_update()
            self.system_state['last_data_update'] = datetime.now(self.paris_tz).isoformat()
            self.system_state['last_dataset_path'] = str(dataset_path)
            
            # 3. Ré-entraînement modèles
            self.logger.info("ETAPE 2: Ré-entraînement modèles ML")
            retrain_success = await self.model_trainer.run_full_retraining()
            if retrain_success:
                self.system_state['last_model_retrain'] = datetime.now(self.paris_tz).isoformat()
                
                # Recharger modèles dans système prédiction
                self.prediction_system.load_existing_models()
                self.logger.info("Modèles rechargés dans système prédiction")
            
            # 4. Démarrer monitoring pré-match
            self.logger.info("ETAPE 3: Démarrage monitoring pré-match")
            # Ne pas attendre - lancer en arrière-plan
            asyncio.create_task(self.pre_match_collector.run_pre_match_monitoring())
            
            cycle_end = datetime.now(self.paris_tz)
            duration = cycle_end - cycle_start
            
            self.system_state['last_complete_cycle'] = cycle_start.isoformat()
            self.system_state['last_cycle_duration'] = str(duration)
            
            self.logger.info(f"=== CYCLE QUOTIDIEN TERMINE - Durée: {duration} ===")
            
            # Sauvegarder état
            self.save_system_state()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur cycle quotidien: {e}")
            self.system_state['last_error'] = str(e)
            self.system_state['last_error_time'] = datetime.now(self.paris_tz).isoformat()
            self.save_system_state()
            raise
    
    async def run_predictions_only(self):
        """Exécuter seulement les prédictions avec modèles actuels"""
        self.logger.info("=== PREDICTIONS AVEC MODELES ACTUELS ===")
        
        try:
            await self.prediction_system.run_daily_predictions()
            
        except Exception as e:
            self.logger.error(f"Erreur prédictions: {e}")
            raise
    
    async def emergency_update_and_predict(self):
        """Mise à jour d'urgence et prédictions immédiates"""
        self.logger.info("=== MODE URGENCE: MISE A JOUR ET PREDICTION ===")
        
        try:
            # 1. Mise à jour rapide données
            self.logger.info("Mise à jour données urgente...")
            await self.data_updater.run_daily_update()
            
            # 2. Ré-entraînement rapide (modèles critiques seulement)
            self.logger.info("Ré-entraînement modèles critiques...")
            await self.model_trainer.run_full_retraining()
            
            # 3. Recharger et prédire
            self.logger.info("Prédictions avec modèles frais...")
            self.prediction_system.load_existing_models()
            await self.prediction_system.run_daily_predictions()
            
            # 4. Collecte pré-match immédiate
            self.logger.info("Collecte pré-match d'urgence...")
            await self.pre_match_collector.emergency_collect_now()
            
            self.logger.info("=== MODE URGENCE TERMINE ===")
            
        except Exception as e:
            self.logger.error(f"Erreur mode urgence: {e}")
            raise
    
    def display_system_status(self):
        """Afficher statut complet du système"""
        now = datetime.now(self.paris_tz)
        
        print("="*60)
        print("STATUT SYSTEME PREDICTION FOOTBALL")
        print("="*60)
        print(f"Heure actuelle Paris: {now.strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"Santé système: {self.system_state.get('system_health', 'unknown').upper()}")
        
        if self.system_state.get('health_issues'):
            print(f"Problèmes: {', '.join(self.system_state['health_issues'])}")
        
        print(f"\nDERNIERES OPERATIONS:")
        
        if self.system_state.get('last_data_update'):
            last_update = datetime.fromisoformat(self.system_state['last_data_update'])
            ago = now - last_update
            print(f"  Données API: {last_update.strftime('%d/%m %H:%M')} (il y a {ago})")
        else:
            print("  Données API: Jamais")
        
        if self.system_state.get('last_model_retrain'):
            last_retrain = datetime.fromisoformat(self.system_state['last_model_retrain'])
            ago = now - last_retrain
            print(f"  Modèles ML: {last_retrain.strftime('%d/%m %H:%M')} (il y a {ago})")
        else:
            print("  Modèles ML: Jamais")
        
        if self.system_state.get('last_complete_cycle'):
            last_cycle = datetime.fromisoformat(self.system_state['last_complete_cycle'])
            duration = self.system_state.get('last_cycle_duration', 'Inconnu')
            ago = now - last_cycle
            print(f"  Cycle complet: {last_cycle.strftime('%d/%m %H:%M')} (durée: {duration}, il y a {ago})")
        
        # Afficher statut pré-match
        pre_match_status = self.pre_match_collector.get_collection_status()
        if pre_match_status['scheduled_groups'] > 0:
            print(f"\nCOLLECTES PRE-MATCH PREVUES:")
            for group_id, group_info in pre_match_status['groups_detail'].items():
                print(f"  {group_id}: {group_info['matches_count']} matchs à {group_info['collection_time']}")
                print(f"    Dans {group_info['time_until_minutes']} min - {group_info['status']}")
        else:
            print(f"\nPas de collectes pré-match planifiées")


async def main():
    """Fonction principale"""
    controller = MasterController()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'full':
            # Cycle complet quotidien
            await controller.run_complete_daily_cycle()
            
        elif command == 'predict':
            # Prédictions seulement
            await controller.run_predictions_only()
            
        elif command == 'emergency':
            # Mode urgence
            await controller.emergency_update_and_predict()
            
        elif command == 'status':
            # Statut système
            controller.display_system_status()
            
        else:
            print("Commandes: full, predict, emergency, status")
    else:
        print("=== CONTROLEUR PRINCIPAL FOOTBALL PREDICTION ===")
        print("Usage:")
        print(f"  {sys.argv[0]} full      - Cycle quotidien complet")
        print(f"  {sys.argv[0]} predict   - Prédictions seulement")
        print(f"  {sys.argv[0]} emergency - Mise à jour urgente + prédictions")
        print(f"  {sys.argv[0]} status    - Statut système")
        print()
        print("AUTOMATISATION WINDOWS:")
        print("  python windows_scheduler.py setup")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nArrêt utilisateur")
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()