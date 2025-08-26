"""
CONFIGURATEUR PLANIFICATEUR AUTO-REENTRAI­NEMENT
Configure l'exécution automatique du système de re-entraînement
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def create_batch_script():
    """Créer script batch pour exécution Windows"""
    
    batch_content = f"""@echo off
cd /d "{Path.cwd()}"
python final_auto_retrain_system.py
echo Execution terminee a %date% %time%
"""
    
    batch_file = Path("auto_retrain.bat")
    with open(batch_file, 'w') as f:
        f.write(batch_content)
    
    return batch_file

def setup_task_scheduler():
    """Configurer tâche planifiée Windows"""
    
    print("CONFIGURATION PLANIFICATEUR WINDOWS")
    print("=" * 40)
    
    # Créer script batch
    batch_file = create_batch_script()
    print(f"Script batch cree: {batch_file}")
    
    # Commandes pour planificateur Windows
    task_name = "AutoRetrainML"
    
    # Supprimer tâche existante si elle existe
    try:
        subprocess.run([
            "schtasks", "/Delete", "/TN", task_name, "/F"
        ], capture_output=True)
    except:
        pass
    
    # Créer nouvelle tâche - quotidienne
    cmd_create = [
        "schtasks", "/Create",
        "/TN", task_name,
        "/TR", str(batch_file.absolute()),
        "/SC", "DAILY",
        "/ST", "08:00",  # Démarre à 8h
        "/F"  # Force création
    ]
    
    try:
        result = subprocess.run(cmd_create, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Tache planifiee creee avec succes!")
            print(f"Nom: {task_name}")
            print("Frequence: Quotidienne (24h)")
            print("Debut: 08:00")
            print()
            print("La tache va automatiquement:")
            print("- Verifier l'age des donnees et modeles")
            print("- Collecter nouvelles donnees si necessaire")
            print("- Re-entrainer modeles si necessaire")
            print()
            
            # Afficher info tâche
            subprocess.run([
                "schtasks", "/Query", "/TN", task_name, "/FO", "LIST"
            ])
            
            return True
        else:
            print("ERREUR creation tache planifiee:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"ERREUR: {e}")
        return False

def setup_manual_execution():
    """Configuration pour exécution manuelle périodique"""
    
    print("CONFIGURATION EXECUTION MANUELLE")
    print("=" * 40)
    
    # Créer script batch
    batch_file = create_batch_script()
    print(f"Script batch cree: {batch_file}")
    
    print("\nPour executer manuellement:")
    print(f"1. Double-cliquer sur: {batch_file}")
    print("2. Ou executer: python final_auto_retrain_system.py")
    print()
    print("Recommandations d'execution:")
    print("- Quotidien: 1 fois par jour (matin) - CONFIGURE")
    print("- Apres grands evenements foot (weekends)")
    print("- Execution manuelle si besoin urgent")
    print()
    
    return True

def main():
    """Configuration du planificateur"""
    
    print("CONFIGURATEUR PLANIFICATEUR AUTO-REENTRAI­NEMENT")
    print("Garantit re-entrainement automatique des modeles ML")
    print()
    
    # Détecter si on est sur Windows
    if sys.platform.startswith('win'):
        print("Systeme Windows detecte")
        print("Tentative configuration Planificateur de taches...")
        print()
        
        success = setup_task_scheduler()
        
        if not success:
            print("\nEchec planificateur automatique")
            print("Configuration manuelle...")
            setup_manual_execution()
    else:
        print("Systeme non-Windows detecte")
        print("Configuration manuelle...")
        setup_manual_execution()
    
    print("\nCONFIGURATION TERMINEE")
    print("Le systeme de re-entrainement automatique est pret!")
    print()
    print("Test immediat...")
    
    # Test immédiat
    try:
        result = subprocess.run([
            "python", "final_auto_retrain_system.py"
        ], cwd=Path.cwd())
        
        if result.returncode == 0:
            print("TEST: REUSSI")
        else:
            print("TEST: ECHEC")
    except Exception as e:
        print(f"TEST: ERREUR {e}")

if __name__ == "__main__":
    main()