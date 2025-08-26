"""
SYSTEME FINAL AUTO-REENTRAI­NEMENT
Version finale qui garantit re-entraînement après nouveaux matchs
"""

import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
import json

def check_dataset_age():
    """Vérifier l'âge du dataset le plus récent"""
    
    data_dir = Path("data/ultra_processed")
    dataset_files = list(data_dir.glob("complete_ml_dataset_*.csv"))
    
    if not dataset_files:
        return 999, None  # Très vieux si pas de dataset
    
    latest_file = max(dataset_files, key=lambda f: f.stat().st_mtime)
    last_update = datetime.fromtimestamp(latest_file.stat().st_mtime)
    age_hours = (datetime.now() - last_update).total_seconds() / 3600
    
    return age_hours, latest_file

def check_models_age():
    """Vérifier l'âge des modèles les plus récents"""
    
    models_dir = Path("models/complete_models")
    
    if not models_dir.exists():
        return 999
    
    model_files = list(models_dir.glob("complete_*.joblib"))
    
    if not model_files:
        return 999
    
    latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
    last_training = datetime.fromtimestamp(latest_model.stat().st_mtime)
    age_hours = (datetime.now() - last_training).total_seconds() / 3600
    
    return age_hours

def force_retrain_cycle():
    """Forcer un cycle complet de re-entraînement"""
    
    print("FORCE CYCLE COMPLET RE-ENTRAI­NEMENT")
    print("=" * 50)
    
    start_time = datetime.now()
    
    # Étape 1: Collecte (si nécessaire)
    dataset_age, _ = check_dataset_age()
    
    if dataset_age > 4:  # Plus de 4h
        print("1. COLLECTE DONNEES (dataset obsolete)...")
        
        try:
            result1 = subprocess.run([
                "python", "complete_data_collector.py"
            ], cwd=Path.cwd(), timeout=7200)
            
            if result1.returncode != 0:
                print("   ECHEC collecte")
                return False
            
            print("   COLLECTE: REUSSIE")
            
        except subprocess.TimeoutExpired:
            print("   COLLECTE: TIMEOUT")
            return False
        except Exception as e:
            print(f"   COLLECTE: ERREUR {e}")
            return False
        
        # Étape 2: Intégration
        print("2. INTEGRATION DATASET...")
        
        try:
            result2 = subprocess.run([
                "python", "integrate_complete_datasets.py"
            ], cwd=Path.cwd(), timeout=1800)
            
            if result2.returncode != 0:
                print("   ECHEC integration")
                return False
            
            print("   INTEGRATION: REUSSIE")
            
        except subprocess.TimeoutExpired:
            print("   INTEGRATION: TIMEOUT")
            return False
        except Exception as e:
            print(f"   INTEGRATION: ERREUR {e}")
            return False
    else:
        print("1. COLLECTE: IGNOREE (dataset recent)")
        print("2. INTEGRATION: IGNOREE (dataset recent)")
    
    # Étape 3: Re-entraînement (TOUJOURS)
    print("3. RE-ENTRAI­NEMENT MODELES (FORCE)...")
    
    try:
        result3 = subprocess.run([
            "python", "adapt_ml_models_complete.py"
        ], cwd=Path.cwd(), timeout=3600)
        
        if result3.returncode != 0:
            print("   ECHEC re-entrainement")
            return False
        
        print("   RE-ENTRAI­NEMENT: REUSSI")
        
    except subprocess.TimeoutExpired:
        print("   RE-ENTRAI­NEMENT: TIMEOUT")
        return False
    except Exception as e:
        print(f"   RE-ENTRAI­NEMENT: ERREUR {e}")
        return False
    
    # Succès
    duration = datetime.now() - start_time
    
    print()
    print("=" * 50)
    print("CYCLE FORCE TERMINE AVEC SUCCES!")
    print("=" * 50)
    print(f"Duree: {duration.total_seconds():.1f}s")
    print("Modeles re-entraines avec donnees les plus recentes")
    
    # Sauvegarder rapport
    report = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": duration.total_seconds(),
        "dataset_age_hours": dataset_age,
        "forced_retrain": True,
        "success": True
    }
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    report_file = logs_dir / f"forced_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Rapport: {report_file}")
    
    return True

def smart_retrain_check():
    """Vérification intelligente et re-entraînement si nécessaire"""
    
    print(f"\nVERIFICATION SYSTEME - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    # Vérifier âges
    dataset_age, latest_dataset = check_dataset_age()
    models_age = check_models_age()
    
    print(f"Dataset age: {dataset_age:.1f}h")
    print(f"Models age: {models_age:.1f}h")
    
    # Logique de décision
    need_retrain = False
    reasons = []
    
    # Critère 1: Modèles plus anciens que dataset
    if models_age > dataset_age + 0.5:  # 30min de marge
        need_retrain = True
        reasons.append("Modeles obsoletes par rapport au dataset")
    
    # Critère 2: Dataset très ancien (>6h)
    if dataset_age > 6:
        need_retrain = True
        reasons.append("Dataset tres ancien (>6h)")
    
    # Critère 3: Modèles très anciens (>12h)
    if models_age > 12:
        need_retrain = True
        reasons.append("Modeles tres anciens (>12h)")
    
    print(f"Re-entrainement requis: {'OUI' if need_retrain else 'NON'}")
    
    if need_retrain:
        for reason in reasons:
            print(f"  - {reason}")
        
        # Lancer re-entraînement
        return force_retrain_cycle()
    else:
        print("Systeme a jour - aucune action requise")
        return True

def main():
    """Fonction principale - Une exécution"""
    
    print("SYSTEME FINAL AUTO-REENTRAI­NEMENT")
    print("Garantit que les modeles sont re-entraines apres nouveaux matchs")
    print()
    
    try:
        success = smart_retrain_check()
        
        if success:
            print("\nSYSTEME AUTO-REENTRAI­NEMENT: OPERATIONNEL")
            print("Les modeles ML sont garantis a jour!")
        else:
            print("\nSYSTEME AUTO-REENTRAI­NEMENT: ECHEC")
        
        return success
        
    except Exception as e:
        print(f"ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()