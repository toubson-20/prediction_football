#!/usr/bin/env python3
"""
Moniteur d'Entraînement Enrichi
Surveille le progrès de l'intégration des features avancées
"""

import time
import json
from pathlib import Path
from datetime import datetime
import logging

class EnhancedTrainingMonitor:
    def __init__(self):
        self.models_path = Path("models/enhanced_models")
        self.data_path = Path("data/ultra_processed")

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def check_training_progress(self):
        """Vérifier le progrès de l'entraînement"""

        # Vérifier fichier dataset enrichi
        enhanced_datasets = list(self.data_path.glob("enhanced_ml_dataset_*.csv"))

        if enhanced_datasets:
            latest_dataset = max(enhanced_datasets, key=lambda x: x.stat().st_mtime)
            dataset_time = datetime.fromtimestamp(latest_dataset.stat().st_mtime)
            self.logger.info(f"Dataset enrichi détecté: {latest_dataset.name}")
            self.logger.info(f"Créé à: {dataset_time}")
        else:
            self.logger.info("Aucun dataset enrichi trouvé encore")

        # Vérifier modèles enrichis
        if self.models_path.exists():
            model_files = list(self.models_path.glob("enhanced_*.joblib"))
            metrics_files = list(self.models_path.glob("enhanced_metrics_*.json"))

            self.logger.info(f"Modèles enrichis: {len(model_files)}")
            self.logger.info(f"Métriques disponibles: {len(metrics_files)}")

            if metrics_files:
                self.logger.info("Modèles entraînés:")
                for metrics_file in sorted(metrics_files):
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)

                        model_name = metrics_file.stem.replace('enhanced_metrics_', '')
                        r2_score = metrics.get('r2', 'N/A')
                        features_count = metrics.get('features_count', 'N/A')

                        self.logger.info(f"  {model_name}: R²={r2_score:.3f} ({features_count} features)")

                    except Exception as e:
                        self.logger.warning(f"Erreur lecture {metrics_file}: {e}")
        else:
            self.logger.info("Répertoire modèles enrichis pas encore créé")

    def wait_for_completion(self, check_interval=30, max_wait=3600):
        """Attendre la fin de l'entraînement"""

        self.logger.info(f"Surveillance entraînement (vérification toutes les {check_interval}s)")

        start_time = datetime.now()

        while (datetime.now() - start_time).seconds < max_wait:
            self.check_training_progress()

            # Vérifier si entraînement terminé
            if self.models_path.exists():
                model_files = list(self.models_path.glob("enhanced_*.joblib"))
                if len(model_files) > 10:  # Au moins quelques modèles entraînés
                    self.logger.info("Entraînement semble terminé!")
                    return True

            self.logger.info(f"Attente... (prochaine vérification dans {check_interval}s)")
            time.sleep(check_interval)

        self.logger.warning("Timeout atteint pour la surveillance")
        return False

if __name__ == "__main__":
    print("=== MONITEUR ENTRAINEMENT ENRICHI ===")

    monitor = EnhancedTrainingMonitor()
    monitor.check_training_progress()

    print("\nPour surveiller en continu:")
    print("monitor.wait_for_completion()")