"""
📊 REALTIME PERFORMANCE MONITOR - PHASE 4
Système de monitoring temps réel des performances avec alertes et dashboards
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from pathlib import Path
import sqlite3
from collections import deque, defaultdict

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MetricType(str, Enum):
    ACCURACY = "accuracy"
    ROI = "roi"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CONFIDENCE_DRIFT = "confidence_drift"

@dataclass
class PerformanceMetric:
    """Métrique de performance temps réel"""
    timestamp: datetime
    metric_type: MetricType
    value: float
    model_id: str
    league: str
    additional_data: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class Alert:
    """Alerte de performance"""
    alert_id: str
    level: AlertLevel
    metric_type: MetricType
    message: str
    model_id: str
    threshold_value: float
    actual_value: float
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class MetricsDatabase:
    """Base de données des métriques temps réel"""
    
    def __init__(self, db_path: str = "data/realtime_metrics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._lock = threading.Lock()
    
    def _init_database(self):
        """Initialiser la base de données"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    model_id TEXT NOT NULL,
                    league TEXT NOT NULL,
                    additional_data TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL UNIQUE,
                    level TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    threshold_value REAL NOT NULL,
                    actual_value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    acknowledged INTEGER DEFAULT 0,
                    resolved INTEGER DEFAULT 0
                )
            """)
            
            # Index pour les performances
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_model ON metrics(model_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_level ON alerts(level)")
    
    def insert_metric(self, metric: PerformanceMetric):
        """Insérer une métrique"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO metrics (timestamp, metric_type, value, model_id, league, additional_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metric.timestamp.isoformat(),
                    metric.metric_type.value,
                    metric.value,
                    metric.model_id,
                    metric.league,
                    json.dumps(metric.additional_data) if metric.additional_data else None
                ))
    
    def insert_alert(self, alert: Alert):
        """Insérer une alerte"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO alerts 
                    (alert_id, level, metric_type, message, model_id, threshold_value, actual_value, timestamp, acknowledged, resolved)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id,
                    alert.level.value,
                    alert.metric_type.value,
                    alert.message,
                    alert.model_id,
                    alert.threshold_value,
                    alert.actual_value,
                    alert.timestamp.isoformat(),
                    int(alert.acknowledged),
                    int(alert.resolved)
                ))
    
    def get_recent_metrics(self, model_id: Optional[str] = None, 
                          metric_type: Optional[MetricType] = None,
                          hours: int = 24, limit: int = 1000) -> List[Dict]:
        """Récupérer les métriques récentes"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = """
                SELECT * FROM metrics 
                WHERE timestamp > ?
            """
            params = [(datetime.now() - timedelta(hours=hours)).isoformat()]
            
            if model_id:
                query += " AND model_id = ?"
                params.append(model_id)
            
            if metric_type:
                query += " AND metric_type = ?"
                params.append(metric_type.value)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Dict]:
        """Récupérer les alertes actives"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM alerts WHERE resolved = 0"
            params = []
            
            if level:
                query += " AND level = ?"
                params.append(level.value)
            
            query += " ORDER BY timestamp DESC"
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

class ThresholdManager:
    """Gestionnaire des seuils d'alertes"""
    
    def __init__(self):
        self.thresholds = {
            MetricType.ACCURACY: {
                AlertLevel.WARNING: 0.05,  # Baisse de 5%
                AlertLevel.CRITICAL: 0.10,  # Baisse de 10%
                AlertLevel.EMERGENCY: 0.20  # Baisse de 20%
            },
            MetricType.ROI: {
                AlertLevel.WARNING: -0.02,  # ROI négatif -2%
                AlertLevel.CRITICAL: -0.05,  # ROI négatif -5%
                AlertLevel.EMERGENCY: -0.10  # ROI négatif -10%
            },
            MetricType.LATENCY: {
                AlertLevel.WARNING: 500,   # 500ms
                AlertLevel.CRITICAL: 1000,  # 1s
                AlertLevel.EMERGENCY: 3000  # 3s
            },
            MetricType.ERROR_RATE: {
                AlertLevel.WARNING: 0.05,  # 5% erreurs
                AlertLevel.CRITICAL: 0.10,  # 10% erreurs
                AlertLevel.EMERGENCY: 0.25  # 25% erreurs
            },
            MetricType.CONFIDENCE_DRIFT: {
                AlertLevel.WARNING: 0.10,  # 10% dérive
                AlertLevel.CRITICAL: 0.20,  # 20% dérive
                AlertLevel.EMERGENCY: 0.35  # 35% dérive
            }
        }
        
        # Seuils par modèle (personnalisables)
        self.model_specific_thresholds: Dict[str, Dict] = {}
    
    def set_model_threshold(self, model_id: str, metric_type: MetricType, 
                          level: AlertLevel, value: float):
        """Définir un seuil spécifique pour un modèle"""
        if model_id not in self.model_specific_thresholds:
            self.model_specific_thresholds[model_id] = {}
        
        if metric_type not in self.model_specific_thresholds[model_id]:
            self.model_specific_thresholds[model_id][metric_type] = {}
        
        self.model_specific_thresholds[model_id][metric_type][level] = value
        logger.info(f"Seuil {level.value} défini pour {model_id}.{metric_type.value}: {value}")
    
    def get_threshold(self, model_id: str, metric_type: MetricType, level: AlertLevel) -> float:
        """Obtenir le seuil pour un modèle et métrique donnés"""
        # Vérifier si un seuil spécifique existe
        if (model_id in self.model_specific_thresholds and
            metric_type in self.model_specific_thresholds[model_id] and
            level in self.model_specific_thresholds[model_id][metric_type]):
            return self.model_specific_thresholds[model_id][metric_type][level]
        
        # Utiliser le seuil par défaut
        return self.thresholds.get(metric_type, {}).get(level, float('inf'))
    
    def check_threshold_violation(self, model_id: str, metric_type: MetricType, 
                                value: float, baseline: Optional[float] = None) -> Optional[AlertLevel]:
        """Vérifier si une valeur viole les seuils"""
        # Pour les métriques relatives (accuracy, ROI), comparer avec baseline
        if baseline is not None and metric_type in [MetricType.ACCURACY, MetricType.ROI]:
            drift = (baseline - value) / baseline if baseline != 0 else 0
            comparison_value = drift
        else:
            comparison_value = value
        
        # Vérifier dans l'ordre de gravité
        for level in [AlertLevel.EMERGENCY, AlertLevel.CRITICAL, AlertLevel.WARNING]:
            threshold = self.get_threshold(model_id, metric_type, level)
            
            if metric_type in [MetricType.LATENCY, MetricType.ERROR_RATE]:
                # Pour latence et taux d'erreur : alerte si valeur > seuil
                if comparison_value > threshold:
                    return level
            else:
                # Pour accuracy et ROI : alerte si dégradation > seuil
                if comparison_value > threshold:
                    return level
        
        return None

class AlertEngine:
    """Moteur d'alertes temps réel"""
    
    def __init__(self, db: MetricsDatabase, threshold_manager: ThresholdManager):
        self.db = db
        self.threshold_manager = threshold_manager
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.baselines: Dict[str, Dict[MetricType, float]] = {}
        self._lock = threading.Lock()
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Ajouter une fonction callback pour les alertes"""
        self.alert_callbacks.append(callback)
    
    def update_baseline(self, model_id: str, metric_type: MetricType, value: float):
        """Mettre à jour la baseline pour un modèle"""
        if model_id not in self.baselines:
            self.baselines[model_id] = {}
        self.baselines[model_id][metric_type] = value
    
    def process_metric(self, metric: PerformanceMetric):
        """Traiter une métrique et générer des alertes si nécessaire"""
        model_id = metric.model_id
        metric_type = metric.metric_type
        
        # Obtenir la baseline si disponible
        baseline = self.baselines.get(model_id, {}).get(metric_type)
        
        # Vérifier les seuils
        violation_level = self.threshold_manager.check_threshold_violation(
            model_id, metric_type, metric.value, baseline
        )
        
        if violation_level:
            self._generate_alert(metric, violation_level, baseline)
        else:
            # Résoudre les alertes existantes pour cette métrique si elle est revenue normale
            self._resolve_alerts(model_id, metric_type)
    
    def _generate_alert(self, metric: PerformanceMetric, level: AlertLevel, baseline: Optional[float]):
        """Générer une alerte"""
        alert_id = f"{metric.model_id}_{metric.metric_type.value}_{level.value}"
        
        # Éviter les alertes en doublon
        if alert_id in self.active_alerts and not self.active_alerts[alert_id].resolved:
            return
        
        # Construire le message d'alerte
        if baseline is not None:
            drift_percent = ((baseline - metric.value) / baseline) * 100 if baseline != 0 else 0
            message = f"{metric.metric_type.value} dégradé de {drift_percent:.1f}% pour {metric.model_id}"
        else:
            message = f"{metric.metric_type.value} = {metric.value:.3f} pour {metric.model_id}"
        
        threshold = self.threshold_manager.get_threshold(metric.model_id, metric.metric_type, level)
        
        alert = Alert(
            alert_id=alert_id,
            level=level,
            metric_type=metric.metric_type,
            message=message,
            model_id=metric.model_id,
            threshold_value=threshold,
            actual_value=metric.value,
            timestamp=metric.timestamp
        )
        
        with self._lock:
            self.active_alerts[alert_id] = alert
            self.db.insert_alert(alert)
        
        # Déclencher les callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Erreur callback alerte: {e}")
        
        logger.warning(f"ALERTE {level.value.upper()}: {message}")
    
    def _resolve_alerts(self, model_id: str, metric_type: MetricType):
        """Résoudre les alertes actives pour un modèle/métrique"""
        alerts_to_resolve = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if (alert.model_id == model_id and 
                alert.metric_type == metric_type and
                not alert.resolved)
        ]
        
        for alert_id in alerts_to_resolve:
            self.acknowledge_alert(alert_id, auto_resolve=True)
    
    def acknowledge_alert(self, alert_id: str, auto_resolve: bool = False):
        """Acquitter/résoudre une alerte"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            if auto_resolve:
                alert.resolved = True
            
            self.db.insert_alert(alert)  # Mettre à jour en DB
            logger.info(f"Alerte {alert_id} {'résolue' if auto_resolve else 'acquittée'}")

class MetricsCollector:
    """Collecteur de métriques temps réel"""
    
    def __init__(self, db: MetricsDatabase, alert_engine: AlertEngine):
        self.db = db
        self.alert_engine = alert_engine
        self.collection_active = False
        self.collection_interval = 30  # 30 secondes
        self.models_to_monitor = set()
        
        # Buffers pour calculs de moyennes mobiles
        self.metric_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def add_model_to_monitor(self, model_id: str):
        """Ajouter un modèle au monitoring"""
        self.models_to_monitor.add(model_id)
        logger.info(f"Modèle {model_id} ajouté au monitoring")
    
    def remove_model_from_monitor(self, model_id: str):
        """Retirer un modèle du monitoring"""
        self.models_to_monitor.discard(model_id)
        logger.info(f"Modèle {model_id} retiré du monitoring")
    
    async def start_collection(self):
        """Démarrer la collecte de métriques"""
        if self.collection_active:
            logger.warning("Collecte déjà active")
            return
        
        self.collection_active = True
        logger.info("🔄 Collecte de métriques temps réel démarrée")
        
        try:
            while self.collection_active:
                await self._collect_metrics_cycle()
                await asyncio.sleep(self.collection_interval)
        except Exception as e:
            logger.error(f"Erreur collecte métriques: {e}")
            self.collection_active = False
    
    def stop_collection(self):
        """Arrêter la collecte"""
        self.collection_active = False
        logger.info("Collecte de métriques arrêtée")
    
    async def _collect_metrics_cycle(self):
        """Exécuter un cycle de collecte"""
        logger.debug("Début cycle de collecte métriques")
        
        for model_id in self.models_to_monitor:
            try:
                # Collecter différents types de métriques
                await self._collect_accuracy_metrics(model_id)
                await self._collect_performance_metrics(model_id)
                await self._collect_system_metrics(model_id)
                
            except Exception as e:
                logger.error(f"Erreur collecte pour {model_id}: {e}")
        
        logger.debug("Cycle de collecte terminé")
    
    async def _collect_accuracy_metrics(self, model_id: str):
        """Collecter les métriques de précision"""
        # Simuler la collecte de métriques de précision
        current_accuracy = np.random.uniform(0.6, 0.85)
        current_roi = np.random.uniform(-0.1, 0.3)
        
        # Créer les métriques
        accuracy_metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.ACCURACY,
            value=current_accuracy,
            model_id=model_id,
            league="premier_league",  # Simulation
            additional_data={"sample_size": np.random.randint(10, 50)}
        )
        
        roi_metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.ROI,
            value=current_roi,
            model_id=model_id,
            league="premier_league",
            additional_data={"period_hours": 24}
        )
        
        # Sauvegarder et traiter
        self.db.insert_metric(accuracy_metric)
        self.db.insert_metric(roi_metric)
        
        self.alert_engine.process_metric(accuracy_metric)
        self.alert_engine.process_metric(roi_metric)
        
        # Mettre à jour les buffers pour moyennes mobiles
        self.metric_buffers[f"{model_id}_accuracy"].append(current_accuracy)
        self.metric_buffers[f"{model_id}_roi"].append(current_roi)
    
    async def _collect_performance_metrics(self, model_id: str):
        """Collecter les métriques de performance"""
        latency = np.random.uniform(50, 200)  # ms
        error_rate = np.random.uniform(0, 0.1)  # 0-10%
        
        latency_metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.LATENCY,
            value=latency,
            model_id=model_id,
            league="system",
            additional_data={"endpoint": "predict"}
        )
        
        error_metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.ERROR_RATE,
            value=error_rate,
            model_id=model_id,
            league="system",
            additional_data={"total_requests": np.random.randint(100, 1000)}
        )
        
        self.db.insert_metric(latency_metric)
        self.db.insert_metric(error_metric)
        
        self.alert_engine.process_metric(latency_metric)
        self.alert_engine.process_metric(error_metric)
    
    async def _collect_system_metrics(self, model_id: str):
        """Collecter les métriques système"""
        # Dérive de confiance simulée
        confidence_drift = np.random.uniform(0, 0.15)
        
        drift_metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.CONFIDENCE_DRIFT,
            value=confidence_drift,
            model_id=model_id,
            league="system",
            additional_data={"baseline_period_days": 30}
        )
        
        self.db.insert_metric(drift_metric)
        self.alert_engine.process_metric(drift_metric)

class DashboardGenerator:
    """Générateur de dashboards temps réel"""
    
    def __init__(self, db: MetricsDatabase):
        self.db = db
    
    def generate_dashboard_data(self, hours: int = 24) -> Dict[str, Any]:
        """Générer les données pour le dashboard"""
        try:
            # Récupérer les métriques récentes
            recent_metrics = self.db.get_recent_metrics(hours=hours)
            active_alerts = self.db.get_active_alerts()
            
            # Organiser les données par type et modèle
            dashboard_data = {
                "timestamp": datetime.now().isoformat(),
                "period_hours": hours,
                "summary": {
                    "total_metrics": len(recent_metrics),
                    "active_alerts": len(active_alerts),
                    "monitored_models": len(set(m['model_id'] for m in recent_metrics)),
                    "critical_alerts": len([a for a in active_alerts if a['level'] == 'critical'])
                },
                "metrics_by_type": self._group_metrics_by_type(recent_metrics),
                "model_performance": self._calculate_model_performance(recent_metrics),
                "alert_summary": self._summarize_alerts(active_alerts),
                "trends": self._calculate_trends(recent_metrics),
                "health_score": self._calculate_health_score(recent_metrics, active_alerts)
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Erreur génération dashboard: {e}")
            return {"error": str(e)}
    
    def _group_metrics_by_type(self, metrics: List[Dict]) -> Dict[str, List]:
        """Grouper les métriques par type"""
        grouped = defaultdict(list)
        for metric in metrics:
            grouped[metric['metric_type']].append({
                'timestamp': metric['timestamp'],
                'value': metric['value'],
                'model_id': metric['model_id']
            })
        
        # Calculer les statistiques pour chaque type
        result = {}
        for metric_type, values in grouped.items():
            result[metric_type] = {
                'data_points': values,
                'avg_value': np.mean([v['value'] for v in values]),
                'min_value': np.min([v['value'] for v in values]),
                'max_value': np.max([v['value'] for v in values]),
                'std_value': np.std([v['value'] for v in values])
            }
        
        return result
    
    def _calculate_model_performance(self, metrics: List[Dict]) -> Dict[str, Dict]:
        """Calculer les performances par modèle"""
        model_metrics = defaultdict(lambda: defaultdict(list))
        
        for metric in metrics:
            model_id = metric['model_id']
            metric_type = metric['metric_type']
            model_metrics[model_id][metric_type].append(metric['value'])
        
        result = {}
        for model_id, metrics_dict in model_metrics.items():
            result[model_id] = {}
            
            for metric_type, values in metrics_dict.items():
                if values:
                    result[model_id][metric_type] = {
                        'current': values[-1],  # Dernière valeur
                        'average': np.mean(values),
                        'trend': self._calculate_trend(values),
                        'stability': 1 - (np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0
                    }
        
        return result
    
    def _summarize_alerts(self, alerts: List[Dict]) -> Dict[str, Any]:
        """Résumé des alertes"""
        if not alerts:
            return {"total": 0, "by_level": {}, "by_model": {}}
        
        by_level = defaultdict(int)
        by_model = defaultdict(int)
        
        for alert in alerts:
            by_level[alert['level']] += 1
            by_model[alert['model_id']] += 1
        
        return {
            "total": len(alerts),
            "by_level": dict(by_level),
            "by_model": dict(by_model),
            "most_recent": max(alerts, key=lambda x: x['timestamp']),
            "oldest_unresolved": min([a for a in alerts if not a['resolved']], 
                                   key=lambda x: x['timestamp'], default=None)
        }
    
    def _calculate_trends(self, metrics: List[Dict]) -> Dict[str, str]:
        """Calculer les tendances générales"""
        trends = {}
        
        # Grouper par type de métrique et calculer la tendance
        by_type = defaultdict(list)
        for metric in sorted(metrics, key=lambda x: x['timestamp']):
            by_type[metric['metric_type']].append(metric['value'])
        
        for metric_type, values in by_type.items():
            if len(values) >= 3:
                trend = self._calculate_trend(values)
                if trend > 0.05:
                    trends[metric_type] = "amélioration"
                elif trend < -0.05:
                    trends[metric_type] = "dégradation"
                else:
                    trends[metric_type] = "stable"
            else:
                trends[metric_type] = "données insuffisantes"
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculer la tendance d'une série de valeurs"""
        if len(values) < 2:
            return 0.0
        
        # Calculer la pente de régression linéaire simple
        n = len(values)
        x = list(range(n))
        
        slope = (n * sum(i * v for i, v in enumerate(values)) - sum(x) * sum(values)) / \
               (n * sum(i ** 2 for i in x) - sum(x) ** 2)
        
        return slope
    
    def _calculate_health_score(self, metrics: List[Dict], alerts: List[Dict]) -> Dict[str, Any]:
        """Calculer un score de santé global"""
        if not metrics:
            return {"score": 0, "status": "unknown", "factors": []}
        
        # Facteurs de santé
        factors = []
        base_score = 100
        
        # Pénalité pour les alertes
        critical_alerts = len([a for a in alerts if a['level'] == 'critical'])
        warning_alerts = len([a for a in alerts if a['level'] == 'warning'])
        
        alert_penalty = critical_alerts * 20 + warning_alerts * 10
        base_score -= alert_penalty
        
        if critical_alerts > 0:
            factors.append(f"{critical_alerts} alertes critiques")
        if warning_alerts > 0:
            factors.append(f"{warning_alerts} alertes d'avertissement")
        
        # Analyse des métriques récentes
        accuracy_metrics = [m for m in metrics if m['metric_type'] == 'accuracy']
        if accuracy_metrics:
            avg_accuracy = np.mean([m['value'] for m in accuracy_metrics])
            if avg_accuracy < 0.7:
                base_score -= 15
                factors.append("précision globale faible")
        
        error_metrics = [m for m in metrics if m['metric_type'] == 'error_rate']
        if error_metrics:
            avg_error_rate = np.mean([m['value'] for m in error_metrics])
            if avg_error_rate > 0.05:
                base_score -= 10
                factors.append("taux d'erreur élevé")
        
        # Déterminer le statut
        if base_score >= 90:
            status = "excellent"
        elif base_score >= 75:
            status = "bon"
        elif base_score >= 60:
            status = "moyen"
        elif base_score >= 40:
            status = "préoccupant"
        else:
            status = "critique"
        
        return {
            "score": max(0, base_score),
            "status": status,
            "factors": factors,
            "recommendations": self._generate_recommendations(factors)
        }
    
    def _generate_recommendations(self, factors: List[str]) -> List[str]:
        """Générer des recommandations basées sur les facteurs"""
        recommendations = []
        
        if any("alerte" in f for f in factors):
            recommendations.append("Examiner et résoudre les alertes actives")
        
        if any("précision" in f for f in factors):
            recommendations.append("Envisager un réentraînement des modèles")
        
        if any("erreur" in f for f in factors):
            recommendations.append("Vérifier la stabilité du système et des données")
        
        if not recommendations:
            recommendations.append("Système opérationnel - continuer la surveillance")
        
        return recommendations

class RealtimePerformanceMonitor:
    """Moniteur principal de performance temps réel"""
    
    def __init__(self):
        self.db = MetricsDatabase()
        self.threshold_manager = ThresholdManager()
        self.alert_engine = AlertEngine(self.db, self.threshold_manager)
        self.metrics_collector = MetricsCollector(self.db, self.alert_engine)
        self.dashboard_generator = DashboardGenerator(self.db)
        
        self.monitoring_active = False
        self.alert_callbacks = []
        
        # Configurer les callbacks d'alerte par défaut
        self.alert_engine.add_alert_callback(self._default_alert_handler)
    
    def _default_alert_handler(self, alert: Alert):
        """Gestionnaire d'alerte par défaut"""
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️", 
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.EMERGENCY: "🔥"
        }
        
        emoji = level_emoji.get(alert.level, "❗")
        logger.info(f"{emoji} ALERTE {alert.level.value.upper()}: {alert.message}")
    
    async def start_monitoring(self):
        """Démarrer le monitoring complet"""
        if self.monitoring_active:
            logger.warning("Monitoring déjà actif")
            return
        
        self.monitoring_active = True
        logger.info("🔄 Monitoring temps réel démarré")
        
        # Ajouter quelques modèles pour les tests
        models_to_monitor = [
            "premier_league_match_result_xgb",
            "la_liga_total_goals_nn", 
            "bundesliga_both_teams_scored_ensemble"
        ]
        
        for model_id in models_to_monitor:
            self.metrics_collector.add_model_to_monitor(model_id)
        
        # Définir quelques baselines pour les alertes
        for model_id in models_to_monitor:
            self.alert_engine.update_baseline(model_id, MetricType.ACCURACY, 0.75)
            self.alert_engine.update_baseline(model_id, MetricType.ROI, 0.15)
        
        # Démarrer la collecte
        await self.metrics_collector.start_collection()
    
    def stop_monitoring(self):
        """Arrêter le monitoring"""
        self.monitoring_active = False
        self.metrics_collector.stop_collection()
        logger.info("Monitoring temps réel arrêté")
    
    def get_dashboard_data(self, hours: int = 24) -> Dict[str, Any]:
        """Obtenir les données du dashboard"""
        return self.dashboard_generator.generate_dashboard_data(hours)
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Dict]:
        """Obtenir les alertes actives"""
        return self.db.get_active_alerts(level)
    
    def acknowledge_alert(self, alert_id: str):
        """Acquitter une alerte"""
        self.alert_engine.acknowledge_alert(alert_id)
    
    def set_custom_threshold(self, model_id: str, metric_type: MetricType, 
                           level: AlertLevel, value: float):
        """Définir un seuil personnalisé"""
        self.threshold_manager.set_model_threshold(model_id, metric_type, level, value)
    
    def add_model_to_monitoring(self, model_id: str):
        """Ajouter un modèle au monitoring"""
        self.metrics_collector.add_model_to_monitor(model_id)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Obtenir le statut du monitoring"""
        return {
            "active": self.monitoring_active,
            "monitored_models": list(self.metrics_collector.models_to_monitor),
            "collection_interval_seconds": self.metrics_collector.collection_interval,
            "active_alerts_count": len(self.alert_engine.active_alerts),
            "thresholds_configured": len(self.threshold_manager.model_specific_thresholds),
            "last_collection": datetime.now().isoformat()
        }

# Fonction de test
async def test_realtime_monitor():
    """Tester le monitoring temps réel"""
    print("📊 Test Monitoring Temps Réel")
    
    monitor = RealtimePerformanceMonitor()
    
    # Test des seuils personnalisés
    print("\n⚙️ Configuration des seuils...")
    monitor.set_custom_threshold(
        "premier_league_match_result_xgb",
        MetricType.ACCURACY,
        AlertLevel.WARNING,
        0.03  # Alerte si baisse de 3%
    )
    
    print("\n🔄 Démarrage du monitoring...")
    # Démarrer le monitoring pour quelques secondes
    monitor_task = asyncio.create_task(monitor.start_monitoring())
    
    # Laisser tourner quelques cycles
    await asyncio.sleep(90)  # 90 secondes pour voir quelques cycles
    
    print("\n📊 Génération du dashboard...")
    dashboard = monitor.get_dashboard_data(hours=1)
    
    print(f"Résumé:")
    print(f"- Métriques collectées: {dashboard['summary']['total_metrics']}")
    print(f"- Alertes actives: {dashboard['summary']['active_alerts']}")
    print(f"- Score de santé: {dashboard['health_score']['score']}/100 ({dashboard['health_score']['status']})")
    
    if dashboard['alert_summary']['total'] > 0:
        print(f"\n🚨 Alertes par niveau:")
        for level, count in dashboard['alert_summary']['by_level'].items():
            print(f"  - {level}: {count}")
    
    print("\n🔄 Arrêt du monitoring...")
    monitor.stop_monitoring()
    monitor_task.cancel()
    
    print("✅ Test terminé avec succès!")

if __name__ == "__main__":
    # Exécuter le test
    asyncio.run(test_realtime_monitor())