"""
CONFIGURATEUR WINDOWS TASK SCHEDULER
Configure automatiquement les tâches Windows pour le système de prédiction football
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict
import json

class WindowsTaskScheduler:
    """Configurateur de tâches Windows automatisées"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent.absolute()
        self.python_exe = sys.executable
        
        # Configuration des tâches
        self.tasks_config = {
            'FootballDataUpdate': {
                'script': 'auto_data_updater.py',
                'schedule': 'DAILY',
                'time': '06:00',
                'description': 'Mise à jour quotidienne données API Football'
            },
            'FootballModelRetrain': {
                'script': 'auto_model_trainer.py', 
                'schedule': 'DAILY',
                'time': '07:00',
                'description': 'Ré-entraînement quotidien modèles ML'
            },
            'FootballPreMatchMonitor': {
                'script': 'pre_match_collector.py',
                'schedule': 'DAILY',
                'time': '08:00',
                'description': 'Monitoring collecte données pré-match'
            }
        }
    
    def create_batch_files(self):
        """Créer fichiers .bat pour exécution Windows"""
        batch_files = {}
        
        for task_name, config in self.tasks_config.items():
            script_name = config['script']
            batch_content = f"""@echo off
cd /d "{self.project_dir}"
"{self.python_exe}" "{script_name}" >> "logs\\{task_name.lower()}.log" 2>&1
"""
            
            batch_file = self.project_dir / f"{task_name.lower()}.bat"
            with open(batch_file, 'w') as f:
                f.write(batch_content)
            
            batch_files[task_name] = batch_file
            print(f"Fichier batch créé: {batch_file}")
        
        return batch_files
    
    def create_windows_tasks(self, batch_files: Dict[str, Path]):
        """Créer tâches Windows Task Scheduler"""
        print("Configuration Windows Task Scheduler...")
        
        for task_name, config in self.tasks_config.items():
            batch_file = batch_files[task_name]
            
            # Commande schtasks pour créer la tâche
            cmd = [
                'schtasks', '/create',
                '/tn', f"FootballPrediction\\{task_name}",
                '/tr', str(batch_file),
                '/sc', config['schedule'],
                '/st', config['time'],
                '/f',  # Force overwrite si existe
                '/rl', 'HIGHEST',  # Run with highest privileges
                '/ru', 'SYSTEM',   # Run as SYSTEM account
                '/desc', f'"{config["description"]}"'
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                
                if result.returncode == 0:
                    print(f"[OK] Tâche créée: {task_name} - {config['time']} quotidien")
                else:
                    print(f"[ERREUR] Création {task_name}: {result.stderr}")
                    
            except Exception as e:
                print(f"[ERREUR] Commande {task_name}: {e}")
    
    def create_emergency_task(self):
        """Créer tâche d'urgence pour collecte immédiate"""
        emergency_batch = f"""@echo off
cd /d "{self.project_dir}"
"{self.python_exe}" "pre_match_collector.py" emergency >> "logs\\emergency_collection.log" 2>&1
"""
        
        emergency_file = self.project_dir / "emergency_collection.bat"
        with open(emergency_file, 'w') as f:
            f.write(emergency_batch)
        
        print(f"Fichier urgence créé: {emergency_file}")
        return emergency_file
    
    def list_existing_tasks(self):
        """Lister tâches existantes"""
        print("Tâches FootballPrediction existantes:")
        
        try:
            result = subprocess.run([
                'schtasks', '/query', '/fo', 'csv', '/tn', 'FootballPrediction\\*'
            ], capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Header + data
                    for line in lines[1:]:  # Skip header
                        parts = line.split(',')
                        if len(parts) >= 4:
                            task_name = parts[0].strip('"')
                            next_run = parts[1].strip('"')
                            status = parts[2].strip('"')
                            print(f"  {task_name}: {status} - Prochaine: {next_run}")
                else:
                    print("  Aucune tâche trouvée")
            else:
                print("  Erreur lecture tâches existantes")
                
        except Exception as e:
            print(f"  Erreur: {e}")
    
    def delete_all_tasks(self):
        """Supprimer toutes les tâches FootballPrediction"""
        print("Suppression tâches existantes...")
        
        for task_name in self.tasks_config.keys():
            try:
                result = subprocess.run([
                    'schtasks', '/delete', '/tn', f"FootballPrediction\\{task_name}", '/f'
                ], capture_output=True, text=True, check=False)
                
                if result.returncode == 0:
                    print(f"  Supprimé: {task_name}")
                else:
                    print(f"  Non trouvé: {task_name}")
                    
            except Exception as e:
                print(f"  Erreur suppression {task_name}: {e}")
    
    def setup_complete_automation(self):
        """Configuration complète de l'automatisation Windows"""
        print("=== CONFIGURATION AUTOMATION WINDOWS ===")
        print(f"Répertoire projet: {self.project_dir}")
        print(f"Python: {self.python_exe}")
        
        # 1. Créer dossiers logs
        logs_dir = self.project_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # 2. Supprimer anciennes tâches
        self.delete_all_tasks()
        
        # 3. Créer fichiers batch
        batch_files = self.create_batch_files()
        
        # 4. Créer tâches Windows
        self.create_windows_tasks(batch_files)
        
        # 5. Créer tâche d'urgence
        emergency_file = self.create_emergency_task()
        
        # 6. Lister tâches créées
        print("\n" + "="*50)
        self.list_existing_tasks()
        
        print(f"\n=== CONFIGURATION TERMINEE ===")
        print(f"Planning automatique:")
        print(f"  06:00 - Mise à jour données API")
        print(f"  07:00 - Ré-entraînement modèles")  
        print(f"  08:00 - Monitoring pré-match")
        print(f"\nFichier urgence: {emergency_file}")
        print(f"Logs: {logs_dir}")
        
        # Sauvegarder configuration
        config_file = self.project_dir / "windows_automation_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'setup_date': datetime.now().isoformat(),
                'project_dir': str(self.project_dir),
                'python_exe': str(self.python_exe),
                'tasks': self.tasks_config,
                'batch_files': {k: str(v) for k, v in batch_files.items()},
                'emergency_file': str(emergency_file)
            }, f, indent=2)
        
        print(f"Configuration sauvée: {config_file}")


def main():
    """Fonction principale"""
    scheduler = WindowsTaskScheduler()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'setup':
            scheduler.setup_complete_automation()
        elif command == 'list':
            scheduler.list_existing_tasks()
        elif command == 'delete':
            scheduler.delete_all_tasks()
        else:
            print("Commandes: setup, list, delete")
    else:
        print("=== CONFIGURATEUR WINDOWS TASK SCHEDULER ===")
        print("Usage:")
        print(f"  {sys.argv[0]} setup   - Configurer automation complète")
        print(f"  {sys.argv[0]} list    - Lister tâches existantes")
        print(f"  {sys.argv[0]} delete  - Supprimer toutes tâches")


if __name__ == "__main__":
    main()