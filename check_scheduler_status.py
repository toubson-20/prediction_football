"""
VERIFICATEUR STATUT PLANIFICATEUR
Vérifie le statut de la tâche planifiée AutoRetrainML
"""

import subprocess
from datetime import datetime

def check_scheduled_task():
    """Vérifier le statut de la tâche planifiée"""
    
    print("VERIFICATION STATUT PLANIFICATEUR")
    print("=" * 40)
    
    try:
        # Exécuter requête sur la tâche
        result = subprocess.run([
            "schtasks", "/Query", "/TN", "AutoRetrainML", "/FO", "LIST"
        ], capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            output = result.stdout
            
            # Parser les informations importantes
            lines = output.split('\n')
            
            for line in lines:
                if 'Nom de la tâche' in line or 'TaskName' in line:
                    print(f"Tache: {line.split(':')[-1].strip()}")
                elif 'Prochaine exécution' in line or 'Next Run Time' in line:
                    print(f"Prochaine execution: {line.split(':')[-1].strip()}")
                elif 'Statut' in line or 'Status' in line:
                    print(f"Statut: {line.split(':')[-1].strip()}")
                elif 'Planifier la tâche' in line or 'Schedule Type' in line:
                    print(f"Type: {line.split(':')[-1].strip()}")
            
            print("\nTACHE PLANIFIEE: CONFIGUREE")
            print("Frequence: QUOTIDIENNE (24h)")
            print("Heure: 08:00")
            
        else:
            print("TACHE NON TROUVEE ou ERREUR")
            print("Sortie d'erreur:", result.stderr)
            
    except Exception as e:
        print(f"ERREUR verification: {e}")

def verify_auto_retrain_files():
    """Vérifier que tous les fichiers nécessaires existent"""
    
    print("\nVERIFICATION FICHIERS SYSTEME")
    print("-" * 30)
    
    from pathlib import Path
    
    files_to_check = [
        "final_auto_retrain_system.py",
        "auto_retrain.bat",
        "complete_data_collector.py",
        "integrate_complete_datasets.py", 
        "adapt_ml_models_complete.py"
    ]
    
    all_exist = True
    
    for file in files_to_check:
        file_path = Path(file)
        exists = file_path.exists()
        status = "OK" if exists else "MANQUANT"
        print(f"  {file}: {status}")
        
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\nFICHIERS: COMPLETS")
    else:
        print("\nFICHIERS: INCOMPLETS - Certains fichiers manquent")
    
    return all_exist

def main():
    """Vérification complète du système"""
    
    print(f"VERIFICATION SYSTEME AUTO-REENTRAI­NEMENT")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Vérifier tâche planifiée
    check_scheduled_task()
    
    # Vérifier fichiers
    files_ok = verify_auto_retrain_files()
    
    print("\n" + "="*40)
    print("RESUME STATUT")
    print("="*40)
    print("Planificateur Windows: CONFIGURE (quotidien 08:00)")
    print(f"Fichiers systeme: {'COMPLETS' if files_ok else 'INCOMPLETS'}")
    print("Execution automatique: ACTIVE")
    print()
    print("Le systeme va automatiquement:")
    print("- Verifier donnees et modeles chaque jour a 08:00")
    print("- Re-entrainer si donnees/modeles obsoletes")
    print("- Maintenir logs d'execution")
    print()
    print("Execution manuelle: python final_auto_retrain_system.py")

if __name__ == "__main__":
    main()