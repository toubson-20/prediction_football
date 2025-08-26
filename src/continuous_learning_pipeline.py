"""
🔄 CONTINUOUS LEARNING PIPELINE - PHASE 4
Système d'apprentissage continu auto-adaptatif pour l'amélioration permanente des modèles ML
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pickle
import os
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Métriques de performance pour un modèle"""
    model_id: str
    prediction_type: str
    league: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    log_loss: float
    roi: float  # Return on Investment
    profit_loss: float
    confidence_calibration: float
    sample_size: int
    period_start: datetime
    period_end: datetime
    
    def to_dict(self) -> Dict:
        """Conversion en dictionnaire"""
        data = asdict(self)
        data['period_start'] = self.period_start.isoformat()
        data['period_end'] = self.period_end.isoformat()
        return data

@dataclass
class RetrainingTrigger:
    """Configuration des déclencheurs de réentraînement"""
    performance_threshold: float = 0.05  # Dégradation de 5%
    sample_size_threshold: int = 20  # Minimum 20 matchs
    time_threshold_days: int = 7  # Maximum 7 jours
    confidence_drift_threshold: float = 0.1  # Dérive de confiance 10%
    force_retrain_days: int = 30  # Réentraînement forcé tous les 30 jours

class PerformanceTracker:
    """Suivi des performances des modèles"""
    
    def __init__(self, storage_path: str = "data/performance_tracking"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metrics_history: Dict[str, List[PerformanceMetrics]] = {}
        self.load_history()
    
    def load_history(self):
        """Charger l'historique des performances"""
        try:
            history_file = self.storage_path / "metrics_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for model_id, metrics_list in data.items():
                        self.metrics_history[model_id] = [
                            PerformanceMetrics(
                                model_id=m['model_id'],
                                prediction_type=m['prediction_type'],
                                league=m['league'],
                                accuracy=m['accuracy'],
                                precision=m['precision'],
                                recall=m['recall'],
                                f1_score=m['f1_score'],
                                roc_auc=m['roc_auc'],
                                log_loss=m['log_loss'],
                                roi=m['roi'],
                                profit_loss=m['profit_loss'],
                                confidence_calibration=m['confidence_calibration'],
                                sample_size=m['sample_size'],
                                period_start=datetime.fromisoformat(m['period_start']),
                                period_end=datetime.fromisoformat(m['period_end'])
                            ) for m in metrics_list
                        ]
        except Exception as e:
            logger.warning(f"Impossible de charger l'historique: {e}")
    
    def save_history(self):
        """Sauvegarder l'historique"""
        try:
            history_file = self.storage_path / "metrics_history.json"
            data = {
                model_id: [m.to_dict() for m in metrics_list]
                for model_id, metrics_list in self.metrics_history.items()
            }
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur sauvegarde historique: {e}")
    
    def add_performance_metrics(self, metrics: PerformanceMetrics):
        """Ajouter de nouvelles métriques"""
        if metrics.model_id not in self.metrics_history:
            self.metrics_history[metrics.model_id] = []
        
        self.metrics_history[metrics.model_id].append(metrics)
        
        # Garder seulement les 100 dernières métriques par modèle
        self.metrics_history[metrics.model_id] = self.metrics_history[metrics.model_id][-100:]
        
        self.save_history()
        logger.info(f"Métriques ajoutées pour {metrics.model_id}")
    
    def get_recent_performance(self, model_id: str, days: int = 7) -> List[PerformanceMetrics]:
        """Obtenir les performances récentes"""
        if model_id not in self.metrics_history:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            m for m in self.metrics_history[model_id]
            if m.period_end >= cutoff_date
        ]
    
    def calculate_performance_drift(self, model_id: str, baseline_days: int = 30, 
                                  recent_days: int = 7) -> Dict[str, float]:
        """Calculer la dérive de performance"""
        baseline_metrics = self.get_recent_performance(model_id, baseline_days)
        recent_metrics = self.get_recent_performance(model_id, recent_days)
        
        if len(baseline_metrics) < 5 or len(recent_metrics) < 3:
            return {'drift': 0.0, 'confidence': 0.0}
        
        # Calcul de la dérive moyenne
        baseline_accuracy = np.mean([m.accuracy for m in baseline_metrics])
        recent_accuracy = np.mean([m.accuracy for m in recent_metrics])
        
        baseline_roi = np.mean([m.roi for m in baseline_metrics])
        recent_roi = np.mean([m.roi for m in recent_metrics])
        
        accuracy_drift = (baseline_accuracy - recent_accuracy) / baseline_accuracy
        roi_drift = (baseline_roi - recent_roi) / abs(baseline_roi) if baseline_roi != 0 else 0
        
        overall_drift = (accuracy_drift + roi_drift) / 2
        
        return {
            'accuracy_drift': accuracy_drift,
            'roi_drift': roi_drift, 
            'overall_drift': overall_drift,
            'baseline_accuracy': baseline_accuracy,
            'recent_accuracy': recent_accuracy,
            'baseline_roi': baseline_roi,
            'recent_roi': recent_roi,
            'confidence': min(len(recent_metrics) / 10, 1.0)
        }

class RetrainingDecisionEngine:
    """Moteur de décision pour le réentraînement"""
    
    def __init__(self, config: RetrainingTrigger = RetrainingTrigger()):
        self.config = config
        self.last_retrain_dates: Dict[str, datetime] = {}
    
    def should_retrain(self, model_id: str, performance_tracker: PerformanceTracker) -> Dict[str, Any]:
        """Décider si un modèle doit être réentraîné"""
        decision_factors = {
            'should_retrain': False,
            'reasons': [],
            'urgency': 'low',  # low, medium, high, critical
            'estimated_improvement': 0.0
        }
        
        # Vérifier la dérive de performance
        drift_analysis = performance_tracker.calculate_performance_drift(model_id)
        
        if drift_analysis['overall_drift'] > self.config.performance_threshold:
            decision_factors['should_retrain'] = True
            decision_factors['reasons'].append(
                f"Dégradation performance: {drift_analysis['overall_drift']:.2%}"
            )
            decision_factors['urgency'] = 'high'
        
        # Vérifier la taille de l'échantillon récent
        recent_metrics = performance_tracker.get_recent_performance(model_id, 7)
        total_samples = sum(m.sample_size for m in recent_metrics)
        
        if total_samples >= self.config.sample_size_threshold:
            if not decision_factors['should_retrain']:
                decision_factors['should_retrain'] = True
                decision_factors['reasons'].append(f"Échantillon suffisant: {total_samples} matchs")
        
        # Réentraînement forcé périodique
        last_retrain = self.last_retrain_dates.get(model_id, 
                                                  datetime.now() - timedelta(days=365))
        days_since_retrain = (datetime.now() - last_retrain).days
        
        if days_since_retrain >= self.config.force_retrain_days:
            decision_factors['should_retrain'] = True
            decision_factors['reasons'].append(
                f"Réentraînement périodique: {days_since_retrain} jours"
            )
            if days_since_retrain > self.config.force_retrain_days * 2:
                decision_factors['urgency'] = 'critical'
        
        # Estimer l'amélioration potentielle
        if decision_factors['should_retrain']:
            decision_factors['estimated_improvement'] = min(
                abs(drift_analysis.get('overall_drift', 0)) * 1.5, 0.3
            )
        
        return decision_factors

class ModelRetrainer:
    """Gestionnaire du réentraînement des modèles"""
    
    def __init__(self, models_path: str = "data/models"):
        self.models_path = Path(models_path)
        self.retrain_queue: List[Dict] = []
        self.active_retrains: Dict[str, bool] = {}
    
    async def retrain_model(self, model_id: str, retrain_type: str = "incremental") -> Dict[str, Any]:
        """Réentraîner un modèle"""
        if model_id in self.active_retrains:
            return {'status': 'error', 'message': 'Réentraînement déjà en cours'}
        
        self.active_retrains[model_id] = True
        retrain_start = datetime.now()
        
        try:
            logger.info(f"Début réentraînement {retrain_type} pour {model_id}")
            
            # Simuler le réentraînement (remplacer par la vraie logique)
            if retrain_type == "incremental":
                await self._incremental_retrain(model_id)
            else:
                await self._full_retrain(model_id)
            
            retrain_duration = datetime.now() - retrain_start
            
            result = {
                'status': 'success',
                'model_id': model_id,
                'retrain_type': retrain_type,
                'duration_seconds': retrain_duration.total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'improvement_estimated': np.random.uniform(0.02, 0.15)  # Simulation
            }
            
            logger.info(f"Réentraînement terminé pour {model_id}")
            return result
            
        except Exception as e:
            logger.error(f"Erreur réentraînement {model_id}: {e}")
            return {
                'status': 'error',
                'model_id': model_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        finally:
            self.active_retrains.pop(model_id, None)
    
    async def _incremental_retrain(self, model_id: str):
        """Réentraînement incrémental"""
        # Simuler le processus
        await asyncio.sleep(np.random.uniform(10, 30))
        logger.info(f"Réentraînement incrémental simulé pour {model_id}")
    
    async def _full_retrain(self, model_id: str):
        """Réentraînement complet"""
        # Simuler le processus plus long
        await asyncio.sleep(np.random.uniform(60, 180))
        logger.info(f"Réentraînement complet simulé pour {model_id}")

class ABTestManager:
    """Gestionnaire des tests A/B pour nouvelles fonctionnalités"""
    
    def __init__(self):
        self.active_tests: Dict[str, Dict] = {}
        self.test_history: List[Dict] = []
    
    def create_ab_test(self, test_name: str, model_a_id: str, model_b_id: str, 
                      traffic_split: float = 0.5, duration_days: int = 7) -> str:
        """Créer un nouveau test A/B"""
        test_id = f"ab_{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.active_tests[test_id] = {
            'test_name': test_name,
            'model_a': model_a_id,
            'model_b': model_b_id,
            'traffic_split': traffic_split,
            'start_date': datetime.now(),
            'end_date': datetime.now() + timedelta(days=duration_days),
            'results_a': [],
            'results_b': []
        }
        
        logger.info(f"Test A/B créé: {test_id}")
        return test_id
    
    def assign_model(self, test_id: str, user_context: Dict) -> str:
        """Assigner un modèle pour un utilisateur"""
        if test_id not in self.active_tests:
            return self.active_tests[test_id]['model_a']
        
        test = self.active_tests[test_id]
        
        # Hash de l'utilisateur pour assignment consistante
        user_hash = hash(str(user_context.get('user_id', 'anonymous'))) % 100
        
        if user_hash < test['traffic_split'] * 100:
            return test['model_a']
        else:
            return test['model_b']
    
    def record_result(self, test_id: str, model_id: str, prediction_result: Dict):
        """Enregistrer un résultat de prédiction"""
        if test_id not in self.active_tests:
            return
        
        test = self.active_tests[test_id]
        result_data = {
            'timestamp': datetime.now(),
            'prediction_result': prediction_result
        }
        
        if model_id == test['model_a']:
            test['results_a'].append(result_data)
        elif model_id == test['model_b']:
            test['results_b'].append(result_data)
    
    def analyze_test_results(self, test_id: str) -> Dict[str, Any]:
        """Analyser les résultats d'un test A/B"""
        if test_id not in self.active_tests:
            return {'error': 'Test non trouvé'}
        
        test = self.active_tests[test_id]
        results_a = test['results_a']
        results_b = test['results_b']
        
        if len(results_a) < 10 or len(results_b) < 10:
            return {
                'status': 'insufficient_data',
                'samples_a': len(results_a),
                'samples_b': len(results_b)
            }
        
        # Calculs statistiques simulés
        accuracy_a = np.random.uniform(0.6, 0.8)
        accuracy_b = np.random.uniform(0.6, 0.8)
        roi_a = np.random.uniform(0.05, 0.25)
        roi_b = np.random.uniform(0.05, 0.25)
        
        statistical_significance = abs(accuracy_a - accuracy_b) > 0.02
        
        return {
            'test_id': test_id,
            'model_a': test['model_a'],
            'model_b': test['model_b'],
            'samples_a': len(results_a),
            'samples_b': len(results_b),
            'accuracy_a': accuracy_a,
            'accuracy_b': accuracy_b,
            'roi_a': roi_a,
            'roi_b': roi_b,
            'winner': 'model_a' if accuracy_a > accuracy_b else 'model_b',
            'improvement': abs(accuracy_a - accuracy_b),
            'statistical_significance': statistical_significance,
            'recommendation': 'deploy' if statistical_significance else 'continue_test'
        }

class ContinuousLearningPipeline:
    """Pipeline principal d'apprentissage continu"""
    
    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.decision_engine = RetrainingDecisionEngine()
        self.model_retrainer = ModelRetrainer()
        self.ab_test_manager = ABTestManager()
        self.pipeline_active = False
        self.monitoring_interval = 3600  # 1 heure
    
    async def start_pipeline(self):
        """Démarrer le pipeline d'apprentissage continu"""
        if self.pipeline_active:
            logger.warning("Pipeline déjà actif")
            return
        
        self.pipeline_active = True
        logger.info("🔄 Pipeline d'apprentissage continu démarré")
        
        try:
            while self.pipeline_active:
                await self._run_monitoring_cycle()
                await asyncio.sleep(self.monitoring_interval)
        except Exception as e:
            logger.error(f"Erreur pipeline: {e}")
            self.pipeline_active = False
    
    def stop_pipeline(self):
        """Arrêter le pipeline"""
        self.pipeline_active = False
        logger.info("Pipeline d'apprentissage continu arrêté")
    
    async def _run_monitoring_cycle(self):
        """Exécuter un cycle de monitoring"""
        logger.info("🔍 Début cycle de monitoring")
        
        # Simuler la collecte de nouvelles métriques
        await self._collect_new_performance_data()
        
        # Analyser les besoins de réentraînement
        await self._analyze_retraining_needs()
        
        # Gérer les tests A/B actifs
        await self._manage_ab_tests()
        
        logger.info("✅ Cycle de monitoring terminé")
    
    async def _collect_new_performance_data(self):
        """Collecter les nouvelles données de performance"""
        # Simuler la collecte de données pour différents modèles
        model_types = ['match_result', 'total_goals', 'both_teams_scored']
        leagues = ['premier_league', 'la_liga', 'bundesliga']
        
        for model_type in model_types:
            for league in leagues:
                model_id = f"{league}_{model_type}_xgb"
                
                # Simuler des métriques
                metrics = PerformanceMetrics(
                    model_id=model_id,
                    prediction_type=model_type,
                    league=league,
                    accuracy=np.random.uniform(0.6, 0.85),
                    precision=np.random.uniform(0.55, 0.8),
                    recall=np.random.uniform(0.5, 0.75),
                    f1_score=np.random.uniform(0.52, 0.77),
                    roc_auc=np.random.uniform(0.65, 0.9),
                    log_loss=np.random.uniform(0.3, 0.8),
                    roi=np.random.uniform(-0.1, 0.3),
                    profit_loss=np.random.uniform(-50, 150),
                    confidence_calibration=np.random.uniform(0.7, 0.95),
                    sample_size=np.random.randint(5, 25),
                    period_start=datetime.now() - timedelta(hours=24),
                    period_end=datetime.now()
                )
                
                self.performance_tracker.add_performance_metrics(metrics)
    
    async def _analyze_retraining_needs(self):
        """Analyser les besoins de réentraînement"""
        for model_id in list(self.performance_tracker.metrics_history.keys()):
            decision = self.decision_engine.should_retrain(model_id, self.performance_tracker)
            
            if decision['should_retrain']:
                logger.info(f"Réentraînement recommandé pour {model_id}: {decision['reasons']}")
                
                # Décider du type de réentraînement
                retrain_type = "full" if decision['urgency'] in ['high', 'critical'] else "incremental"
                
                # Lancer le réentraînement en arrière-plan
                asyncio.create_task(self.model_retrainer.retrain_model(model_id, retrain_type))
    
    async def _manage_ab_tests(self):
        """Gérer les tests A/B actifs"""
        completed_tests = []
        
        for test_id, test_info in self.ab_test_manager.active_tests.items():
            if datetime.now() >= test_info['end_date']:
                # Analyser les résultats
                results = self.ab_test_manager.analyze_test_results(test_id)
                
                if results.get('recommendation') == 'deploy':
                    logger.info(f"Test A/B {test_id} terminé - Déploiement recommandé du {results['winner']}")
                
                completed_tests.append(test_id)
        
        # Nettoyer les tests terminés
        for test_id in completed_tests:
            self.ab_test_manager.test_history.append(
                self.ab_test_manager.active_tests.pop(test_id)
            )
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Obtenir le statut du pipeline"""
        return {
            'active': self.pipeline_active,
            'models_monitored': len(self.performance_tracker.metrics_history),
            'active_ab_tests': len(self.ab_test_manager.active_tests),
            'active_retrains': len(self.model_retrainer.active_retrains),
            'monitoring_interval_hours': self.monitoring_interval / 3600,
            'last_cycle': datetime.now().isoformat()
        }
    
    def trigger_manual_retrain(self, model_id: str, retrain_type: str = "full") -> Dict[str, Any]:
        """Déclencher un réentraînement manuel"""
        return asyncio.create_task(
            self.model_retrainer.retrain_model(model_id, retrain_type)
        )

# Fonction principale pour tests
async def test_continuous_learning_pipeline():
    """Tester le pipeline d'apprentissage continu"""
    print("🔄 Test Pipeline d'Apprentissage Continu")
    
    pipeline = ContinuousLearningPipeline()
    
    # Test du tracking de performance
    print("\n📊 Test Performance Tracking...")
    test_metrics = PerformanceMetrics(
        model_id="test_model",
        prediction_type="match_result", 
        league="premier_league",
        accuracy=0.75,
        precision=0.73,
        recall=0.71,
        f1_score=0.72,
        roc_auc=0.82,
        log_loss=0.45,
        roi=0.15,
        profit_loss=75.5,
        confidence_calibration=0.85,
        sample_size=20,
        period_start=datetime.now() - timedelta(hours=24),
        period_end=datetime.now()
    )
    
    pipeline.performance_tracker.add_performance_metrics(test_metrics)
    print("✅ Métriques ajoutées")
    
    # Test décision de réentraînement
    print("\n🎯 Test Décision Réentraînement...")
    decision = pipeline.decision_engine.should_retrain("test_model", pipeline.performance_tracker)
    print(f"Décision: {decision}")
    
    # Test A/B Testing
    print("\n🧪 Test A/B Testing...")
    test_id = pipeline.ab_test_manager.create_ab_test(
        "new_feature_test", "model_v1", "model_v2"
    )
    print(f"Test A/B créé: {test_id}")
    
    # Simuler quelques résultats
    for i in range(15):
        user_context = {'user_id': f'user_{i}'}
        assigned_model = pipeline.ab_test_manager.assign_model(test_id, user_context)
        
        # Simuler un résultat
        result = {
            'accuracy': np.random.uniform(0.6, 0.8),
            'roi': np.random.uniform(0.05, 0.25)
        }
        
        pipeline.ab_test_manager.record_result(test_id, assigned_model, result)
    
    # Analyser les résultats
    analysis = pipeline.ab_test_manager.analyze_test_results(test_id)
    print(f"Analyse A/B: {analysis}")
    
    print("\n🎉 Tests terminés avec succès!")

if __name__ == "__main__":
    # Exécuter les tests
    asyncio.run(test_continuous_learning_pipeline())