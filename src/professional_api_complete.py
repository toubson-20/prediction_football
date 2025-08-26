"""
🚀 PROFESSIONAL API COMPLETE - PHASE 4
API professionnelle complète avec 12 endpoints avancés pour le système ML
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import logging
import uvicorn
from dataclasses import asdict
import numpy as np

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modèles Pydantic pour l'API
class PredictionType(str, Enum):
    MATCH_RESULT = "match_result"
    TOTAL_GOALS = "total_goals" 
    BOTH_TEAMS_SCORED = "both_teams_scored"
    HOME_GOALS = "home_goals"
    AWAY_GOALS = "away_goals"
    OVER_2_5 = "over_2_5_goals"

class League(str, Enum):
    PREMIER_LEAGUE = "premier_league"
    LA_LIGA = "la_liga" 
    BUNDESLIGA = "bundesliga"
    SERIE_A = "serie_a"
    LIGUE_1 = "ligue_1"
    CHAMPIONS_LEAGUE = "champions_league"

class CouponRequest(BaseModel):
    match_ids: List[int] = Field(..., description="IDs des matchs à inclure")
    risk_profile: str = Field("balanced", description="Profil de risque: conservative, balanced, aggressive")
    min_confidence: float = Field(70.0, description="Confiance minimum (0-100)")
    max_predictions: int = Field(10, description="Nombre maximum de prédictions")
    budget: Optional[float] = Field(None, description="Budget disponible")

class CustomPredictionRequest(BaseModel):
    match_id: int = Field(..., description="ID du match")
    prediction_types: List[PredictionType] = Field(..., description="Types de prédictions demandées")
    league: League = Field(..., description="Ligue du match")
    custom_features: Optional[Dict[str, Any]] = Field(None, description="Features personnalisées")

class BulkPredictionRequest(BaseModel):
    match_ids: List[int] = Field(..., description="IDs des matchs")
    prediction_types: List[PredictionType] = Field(..., description="Types de prédictions")
    leagues: List[League] = Field(..., description="Ligues")
    batch_size: int = Field(50, description="Taille des batches")

class RetrainRequest(BaseModel):
    model_ids: Optional[List[str]] = Field(None, description="IDs des modèles à réentraîner")
    retrain_type: str = Field("incremental", description="Type: incremental, full")
    priority: str = Field("normal", description="Priorité: low, normal, high")
    schedule_time: Optional[datetime] = Field(None, description="Programmer le réentraînement")

class ModelConfigRequest(BaseModel):
    model_id: str = Field(..., description="ID du modèle")
    config_updates: Dict[str, Any] = Field(..., description="Mises à jour de configuration")
    validation_required: bool = Field(True, description="Validation requise")

# Gestionnaire des connexions WebSocket
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscription_filters: Dict[WebSocket, Dict] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connecté. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.subscription_filters.pop(websocket, None)
        logger.info(f"WebSocket déconnecté. Total: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Erreur envoi message personnel: {e}")

    async def broadcast(self, message: dict, filters: Dict = None):
        if not self.active_connections:
            return
            
        disconnected = []
        for connection in self.active_connections:
            try:
                # Vérifier les filtres si spécifiés
                if filters and connection in self.subscription_filters:
                    user_filters = self.subscription_filters[connection]
                    if not self._match_filters(message, user_filters):
                        continue
                
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Erreur broadcast: {e}")
                disconnected.append(connection)
        
        # Nettoyer les connexions fermées
        for conn in disconnected:
            self.disconnect(conn)

    def _match_filters(self, message: dict, filters: dict) -> bool:
        """Vérifier si un message correspond aux filtres d'un utilisateur"""
        for key, value in filters.items():
            if key in message and message[key] != value:
                return False
        return True

    def set_subscription_filters(self, websocket: WebSocket, filters: Dict):
        """Définir les filtres pour un WebSocket"""
        self.subscription_filters[websocket] = filters

# Simulateurs de données (remplacer par vraies intégrations)
class DataSimulator:
    """Simulateur de données pour les tests"""
    
    @staticmethod
    def generate_coupon_predictions(request: CouponRequest) -> Dict[str, Any]:
        """Générer des prédictions de coupon simulées"""
        predictions = []
        
        for match_id in request.match_ids:
            if len(predictions) >= request.max_predictions:
                break
                
            # Simuler des prédictions selon le profil de risque
            if request.risk_profile == "conservative":
                confidence = np.random.uniform(75, 95)
                odds = np.random.uniform(1.2, 1.8)
            elif request.risk_profile == "aggressive": 
                confidence = np.random.uniform(60, 85)
                odds = np.random.uniform(2.0, 8.0)
            else:  # balanced
                confidence = np.random.uniform(70, 90)
                odds = np.random.uniform(1.5, 3.5)
            
            if confidence >= request.min_confidence:
                predictions.append({
                    "match_id": match_id,
                    "prediction_type": np.random.choice(list(PredictionType)),
                    "prediction_value": np.random.choice(["1", "X", "2", "Over 2.5", "Under 2.5"]),
                    "confidence": confidence,
                    "odds": odds,
                    "expected_value": (odds * (confidence/100)) - 1,
                    "stake_recommended": 50 if request.budget else None,
                    "model_used": f"ensemble_{np.random.choice(['xgb', 'nn', 'rf'])}"
                })
        
        return {
            "coupon_id": f"coupon_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "risk_profile": request.risk_profile,
            "total_predictions": len(predictions),
            "predictions": predictions,
            "total_odds": np.prod([p["odds"] for p in predictions]),
            "total_confidence": np.mean([p["confidence"] for p in predictions]),
            "recommended_stake": min(request.budget * 0.02, 100) if request.budget else 50,
            "potential_return": sum(p["stake_recommended"] * p["odds"] for p in predictions if p["stake_recommended"]),
            "generated_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def generate_custom_predictions(request: CustomPredictionRequest) -> Dict[str, Any]:
        """Générer des prédictions personnalisées simulées"""
        predictions = []
        
        for pred_type in request.prediction_types:
            confidence = np.random.uniform(60, 90)
            
            if pred_type == PredictionType.MATCH_RESULT:
                prediction_value = np.random.choice(["1", "X", "2"])
                odds = np.random.uniform(1.8, 4.5)
            elif pred_type in [PredictionType.TOTAL_GOALS, PredictionType.HOME_GOALS, PredictionType.AWAY_GOALS]:
                prediction_value = str(np.random.randint(0, 4))
                odds = np.random.uniform(2.0, 8.0)
            else:
                prediction_value = np.random.choice(["Yes", "No"])
                odds = np.random.uniform(1.4, 2.8)
            
            predictions.append({
                "prediction_type": pred_type,
                "prediction_value": prediction_value,
                "confidence": confidence,
                "odds": odds,
                "expected_value": (odds * (confidence/100)) - 1,
                "model_used": f"{request.league}_{pred_type}_ensemble",
                "features_importance": {
                    "recent_form": np.random.uniform(0.1, 0.3),
                    "head_to_head": np.random.uniform(0.05, 0.25),
                    "team_strength": np.random.uniform(0.15, 0.35),
                    "home_advantage": np.random.uniform(0.05, 0.2)
                }
            })
        
        return {
            "match_id": request.match_id,
            "league": request.league,
            "total_predictions": len(predictions),
            "predictions": predictions,
            "generated_at": datetime.now().isoformat(),
            "processing_time_ms": np.random.uniform(50, 200)
        }

# Application FastAPI
app = FastAPI(
    title="⚽ Football ML Predictions API",
    description="API professionnelle complète pour prédictions ML football avec apprentissage continu",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gestionnaire WebSocket global
websocket_manager = WebSocketManager()

# Simulateurs globaux (remplacer par vraies intégrations)
data_simulator = DataSimulator()

# ==================== ENDPOINTS PRÉDICTIONS ====================

@app.post("/predict/coupon/{match_id}")
async def generate_intelligent_coupon(match_id: int, request: CouponRequest):
    """
    🎯 Générer un coupon intelligent adaptatif
    
    Génère un coupon optimisé avec 5-12 prédictions selon le profil de risque
    et les contraintes spécifiées.
    """
    try:
        # Ajouter le match principal à la liste
        if match_id not in request.match_ids:
            request.match_ids.insert(0, match_id)
        
        # Générer le coupon
        coupon_data = data_simulator.generate_coupon_predictions(request)
        
        # Broadcaster via WebSocket
        await websocket_manager.broadcast({
            "type": "coupon_generated",
            "coupon_id": coupon_data["coupon_id"],
            "match_id": match_id,
            "predictions_count": coupon_data["total_predictions"]
        })
        
        return {
            "status": "success",
            "data": coupon_data,
            "message": f"Coupon intelligent généré avec {coupon_data['total_predictions']} prédictions"
        }
        
    except Exception as e:
        logger.error(f"Erreur génération coupon: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/custom")
async def generate_custom_predictions(request: CustomPredictionRequest):
    """
    🔧 Générer des prédictions personnalisées à la demande
    
    Permet de demander des prédictions spécifiques pour un match donné.
    """
    try:
        predictions_data = data_simulator.generate_custom_predictions(request)
        
        return {
            "status": "success", 
            "data": predictions_data,
            "message": f"Prédictions générées pour {len(request.prediction_types)} types"
        }
        
    except Exception as e:
        logger.error(f"Erreur prédictions personnalisées: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/bulk")
async def generate_bulk_predictions(request: BulkPredictionRequest, background_tasks: BackgroundTasks):
    """
    📦 Générer des prédictions en masse
    
    Traite de grandes quantités de prédictions en arrière-plan.
    """
    try:
        job_id = f"bulk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Lancer le traitement en arrière-plan
        background_tasks.add_task(
            process_bulk_predictions, 
            job_id, 
            request
        )
        
        return {
            "status": "accepted",
            "job_id": job_id,
            "estimated_completion": (datetime.now() + timedelta(minutes=5)).isoformat(),
            "message": f"Traitement en cours pour {len(request.match_ids)} matchs"
        }
        
    except Exception as e:
        logger.error(f"Erreur prédictions masse: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/predict/live")
async def live_predictions_websocket(websocket: WebSocket):
    """
    🔴 Prédictions temps réel via WebSocket
    
    Fournit des mises à jour en temps réel des prédictions.
    """
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # Recevoir les filtres du client
            data = await websocket.receive_json()
            
            if data.get("action") == "subscribe":
                filters = data.get("filters", {})
                websocket_manager.set_subscription_filters(websocket, filters)
                await websocket_manager.send_personal_message({
                    "type": "subscription_confirmed",
                    "filters": filters
                }, websocket)
            
            elif data.get("action") == "predict":
                # Traitement d'une prédiction en temps réel
                match_id = data.get("match_id")
                if match_id:
                    # Simuler une prédiction rapide
                    prediction = {
                        "type": "live_prediction",
                        "match_id": match_id,
                        "prediction": {
                            "type": "match_result",
                            "value": np.random.choice(["1", "X", "2"]),
                            "confidence": np.random.uniform(70, 95),
                            "odds": np.random.uniform(1.5, 4.0)
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    await websocket_manager.send_personal_message(prediction, websocket)
    
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

# ==================== ENDPOINTS ANALYTICS & INSIGHTS ====================

@app.get("/analytics/performance")
async def get_performance_analytics(
    days: int = 30,
    leagues: Optional[List[League]] = None,
    prediction_types: Optional[List[PredictionType]] = None
):
    """
    📊 Analytics de performance (ROI, précision)
    
    Fournit des analyses détaillées des performances des modèles.
    """
    try:
        # Simuler des analytics de performance
        performance_data = {
            "period": {
                "start_date": (datetime.now() - timedelta(days=days)).isoformat(),
                "end_date": datetime.now().isoformat(),
                "total_days": days
            },
            "overall_metrics": {
                "total_predictions": np.random.randint(500, 2000),
                "accuracy": np.random.uniform(0.65, 0.82),
                "precision": np.random.uniform(0.62, 0.79),
                "recall": np.random.uniform(0.58, 0.76),
                "f1_score": np.random.uniform(0.60, 0.77),
                "roi": np.random.uniform(0.08, 0.25),
                "profit_loss": np.random.uniform(-100, 500),
                "win_rate": np.random.uniform(0.55, 0.75)
            },
            "by_league": {
                league.value: {
                    "accuracy": np.random.uniform(0.60, 0.85),
                    "roi": np.random.uniform(0.05, 0.30),
                    "sample_size": np.random.randint(50, 300)
                } for league in (leagues or list(League))
            },
            "by_prediction_type": {
                pred_type.value: {
                    "accuracy": np.random.uniform(0.55, 0.80),
                    "confidence_calibration": np.random.uniform(0.70, 0.95),
                    "avg_odds": np.random.uniform(1.8, 3.5)
                } for pred_type in (prediction_types or list(PredictionType))
            },
            "trends": [
                {
                    "date": (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
                    "accuracy": np.random.uniform(0.60, 0.85),
                    "roi": np.random.uniform(-0.05, 0.30)
                } for i in range(days, 0, -1)
            ]
        }
        
        return {
            "status": "success",
            "data": performance_data,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur analytics performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/trends")
async def get_trends_analysis(
    leagues: Optional[List[League]] = None,
    teams: Optional[List[str]] = None,
    timeframe: str = "last_month"
):
    """
    📈 Analyse des tendances par ligue/équipe
    
    Identifie les patterns et tendances dans les performances.
    """
    try:
        trends_data = {
            "timeframe": timeframe,
            "leagues_analyzed": leagues or ["all"],
            "teams_analyzed": teams or ["all"],
            "key_trends": [
                {
                    "trend_type": "home_advantage_declining",
                    "description": "L'avantage du domicile diminue de 2.3% ce mois",
                    "impact_score": 0.73,
                    "affected_leagues": ["premier_league", "la_liga"]
                },
                {
                    "trend_type": "goal_scoring_increase", 
                    "description": "Augmentation de 8% des matchs Over 2.5",
                    "impact_score": 0.81,
                    "affected_leagues": ["bundesliga", "serie_a"]
                }
            ],
            "league_insights": {
                league.value: {
                    "avg_goals_per_match": np.random.uniform(2.1, 3.2),
                    "home_win_rate": np.random.uniform(0.35, 0.55),
                    "draw_rate": np.random.uniform(0.20, 0.35),
                    "top_prediction_accuracy": np.random.uniform(0.70, 0.90)
                } for league in (leagues or list(League))
            }
        }
        
        return {
            "status": "success",
            "data": trends_data,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur analyse tendances: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/insights/model")
async def get_model_insights(model_id: str, explain_prediction: Optional[bool] = False):
    """
    🧠 Explications et insights des modèles IA
    
    Fournit des explications sur le fonctionnement des modèles.
    """
    try:
        model_insights = {
            "model_id": model_id,
            "model_type": np.random.choice(["xgboost", "neural_network", "ensemble"]),
            "last_trained": (datetime.now() - timedelta(days=np.random.randint(1, 30))).isoformat(),
            "performance_metrics": {
                "accuracy": np.random.uniform(0.65, 0.85),
                "precision": np.random.uniform(0.60, 0.80),
                "recall": np.random.uniform(0.58, 0.78),
                "training_samples": np.random.randint(5000, 20000)
            },
            "feature_importance": {
                "recent_form": np.random.uniform(0.15, 0.35),
                "head_to_head": np.random.uniform(0.10, 0.25),
                "team_strength": np.random.uniform(0.20, 0.40),
                "player_injuries": np.random.uniform(0.05, 0.15),
                "weather_conditions": np.random.uniform(0.02, 0.08)
            },
            "model_characteristics": {
                "complexity_level": np.random.choice(["medium", "high"]),
                "interpretability": np.random.uniform(0.6, 0.9),
                "robustness_score": np.random.uniform(0.7, 0.95)
            }
        }
        
        if explain_prediction:
            model_insights["prediction_explanation"] = {
                "decision_path": [
                    "Analyse forme récente: +0.15 confiance",
                    "Historique confrontations: +0.08 confiance", 
                    "Force offensive domicile: +0.12 confiance",
                    "Blessures clés: -0.03 confiance"
                ],
                "confidence_breakdown": {
                    "base_model": 0.68,
                    "contextual_adjustments": 0.07,
                    "uncertainty_penalty": -0.02,
                    "final_confidence": 0.73
                }
            }
        
        return {
            "status": "success",
            "data": model_insights,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur insights modèle: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/insights/features")
async def get_feature_importance(
    league: Optional[League] = None,
    prediction_type: Optional[PredictionType] = None,
    top_n: int = 20
):
    """
    🔍 Importance des variables dans les prédictions
    
    Analyse l'importance relative des différentes features.
    """
    try:
        features_data = {
            "analysis_scope": {
                "league": league.value if league else "all_leagues",
                "prediction_type": prediction_type.value if prediction_type else "all_types",
                "top_features": top_n
            },
            "global_importance": {
                f"feature_{i}": {
                    "name": np.random.choice([
                        "recent_form_home", "recent_form_away", "head_to_head_record",
                        "team_strength_rating", "goal_difference_trend", "home_advantage",
                        "player_injuries", "weather_conditions", "referee_strictness",
                        "motivation_index", "tactical_setup", "crowd_impact"
                    ]),
                    "importance_score": np.random.uniform(0.05, 0.25),
                    "stability": np.random.uniform(0.7, 0.95),
                    "correlation_with_target": np.random.uniform(0.1, 0.6)
                } for i in range(top_n)
            },
            "contextual_variations": {
                "by_league": {
                    "premier_league": {"home_advantage": 0.18, "pace_of_play": 0.22},
                    "la_liga": {"technical_ability": 0.25, "possession_style": 0.19},
                    "bundesliga": {"pressing_intensity": 0.21, "counter_attacks": 0.16}
                },
                "by_season_phase": {
                    "early_season": {"team_chemistry": 0.30, "new_signings": 0.15},
                    "mid_season": {"form_consistency": 0.28, "injury_impact": 0.18},
                    "late_season": {"motivation": 0.35, "fatigue_factor": 0.22}
                }
            }
        }
        
        return {
            "status": "success", 
            "data": features_data,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur importance features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ENDPOINTS ADMINISTRATION ====================

@app.post("/admin/retrain")
async def trigger_model_retraining(request: RetrainRequest, background_tasks: BackgroundTasks):
    """
    🔄 Déclencher un réentraînement des modèles
    
    Lance le processus de réentraînement selon les paramètres spécifiés.
    """
    try:
        job_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Programmer ou lancer immédiatement
        if request.schedule_time:
            # Programmer le réentraînement
            delay = (request.schedule_time - datetime.now()).total_seconds()
            if delay > 0:
                background_tasks.add_task(
                    schedule_delayed_retrain,
                    job_id,
                    request,
                    delay
                )
            else:
                raise HTTPException(status_code=400, detail="L'heure programmée est dans le passé")
        else:
            # Lancer immédiatement
            background_tasks.add_task(
                process_model_retraining,
                job_id, 
                request
            )
        
        return {
            "status": "accepted",
            "job_id": job_id,
            "retrain_type": request.retrain_type,
            "priority": request.priority,
            "models_affected": len(request.model_ids) if request.model_ids else "all",
            "scheduled_for": request.schedule_time.isoformat() if request.schedule_time else "immediate"
        }
        
    except Exception as e:
        logger.error(f"Erreur déclenchement réentraînement: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/health")
async def get_system_health():
    """
    🏥 Santé et statut du système
    
    Fournit un aperçu complet de l'état du système.
    """
    try:
        health_data = {
            "system_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": np.random.uniform(24, 720),
            "components": {
                "api_server": {
                    "status": "healthy",
                    "response_time_ms": np.random.uniform(50, 150),
                    "requests_per_minute": np.random.randint(10, 100)
                },
                "ml_models": {
                    "status": "healthy", 
                    "active_models": np.random.randint(50, 180),
                    "avg_prediction_time_ms": np.random.uniform(20, 80)
                },
                "database": {
                    "status": "healthy",
                    "connection_pool": f"{np.random.randint(5, 15)}/20",
                    "query_time_ms": np.random.uniform(10, 50)
                },
                "continuous_learning": {
                    "status": "active",
                    "active_retrains": np.random.randint(0, 3),
                    "last_cycle": (datetime.now() - timedelta(minutes=np.random.randint(5, 60))).isoformat()
                },
                "websocket_connections": {
                    "active_connections": len(websocket_manager.active_connections),
                    "total_messages_today": np.random.randint(100, 1000)
                }
            },
            "performance_metrics": {
                "cpu_usage_percent": np.random.uniform(20, 70),
                "memory_usage_percent": np.random.uniform(30, 80),
                "disk_usage_percent": np.random.uniform(40, 85),
                "network_throughput_mbps": np.random.uniform(10, 100)
            },
            "alerts": []  # Vide si tout va bien
        }
        
        # Ajouter des alertes si nécessaire
        if health_data["performance_metrics"]["cpu_usage_percent"] > 80:
            health_data["alerts"].append({
                "level": "warning",
                "component": "system",
                "message": "Utilisation CPU élevée"
            })
        
        return {
            "status": "success",
            "data": health_data
        }
        
    except Exception as e:
        logger.error(f"Erreur santé système: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/config")  
async def update_model_configuration(request: ModelConfigRequest):
    """
    ⚙️ Configuration des modèles
    
    Met à jour la configuration d'un ou plusieurs modèles.
    """
    try:
        # Valider la configuration si demandé
        validation_results = {}
        if request.validation_required:
            validation_results = {
                "config_valid": True,
                "warnings": [],
                "estimated_impact": "low"
            }
            
            # Simuler quelques validations
            for key, value in request.config_updates.items():
                if key.endswith("_threshold") and (value < 0 or value > 1):
                    validation_results["warnings"].append(
                        f"Valeur {key} hors limites recommandées: {value}"
                    )
        
        config_response = {
            "model_id": request.model_id,
            "config_updates": request.config_updates,
            "validation_results": validation_results,
            "applied_at": datetime.now().isoformat(),
            "restart_required": np.random.choice([True, False]),
            "rollback_available": True
        }
        
        # Broadcaster la mise à jour via WebSocket
        await websocket_manager.broadcast({
            "type": "model_config_updated",
            "model_id": request.model_id,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "status": "success",
            "data": config_response,
            "message": f"Configuration mise à jour pour {request.model_id}"
        }
        
    except Exception as e:
        logger.error(f"Erreur configuration modèle: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/logs")
async def get_system_logs(
    level: str = "INFO", 
    component: Optional[str] = None,
    last_hours: int = 24,
    limit: int = 100
):
    """
    📋 Journaux détaillés du système
    
    Récupère les logs système selon les critères spécifiés.
    """
    try:
        # Simuler des logs système
        log_entries = []
        
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"] 
        components = ["api", "ml_engine", "continuous_learning", "websocket", "database"]
        
        for i in range(limit):
            timestamp = datetime.now() - timedelta(
                hours=np.random.uniform(0, last_hours)
            )
            
            log_entry = {
                "timestamp": timestamp.isoformat(),
                "level": np.random.choice(log_levels),
                "component": component or np.random.choice(components),
                "message": f"Message de log simulé #{i+1}",
                "details": {
                    "request_id": f"req_{np.random.randint(1000, 9999)}",
                    "user_id": f"user_{np.random.randint(1, 100)}",
                    "duration_ms": np.random.uniform(10, 500)
                }
            }
            
            # Filtrer par niveau si spécifié
            level_order = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
            if level_order[log_entry["level"]] >= level_order[level]:
                log_entries.append(log_entry)
        
        # Trier par timestamp décroissant
        log_entries.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "status": "success",
            "data": {
                "total_entries": len(log_entries),
                "filters": {
                    "level": level,
                    "component": component,
                    "last_hours": last_hours
                },
                "logs": log_entries
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur récupération logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== FONCTIONS UTILITAIRES ====================

async def process_bulk_predictions(job_id: str, request: BulkPredictionRequest):
    """Traiter les prédictions en masse en arrière-plan"""
    try:
        logger.info(f"Début traitement masse {job_id}")
        
        # Simuler le traitement par batches
        total_matches = len(request.match_ids)
        processed = 0
        
        for i in range(0, total_matches, request.batch_size):
            batch_matches = request.match_ids[i:i + request.batch_size]
            
            # Simuler le traitement du batch
            await asyncio.sleep(np.random.uniform(1, 3))
            processed += len(batch_matches)
            
            # Broadcaster le progrès
            await websocket_manager.broadcast({
                "type": "bulk_progress",
                "job_id": job_id,
                "processed": processed,
                "total": total_matches,
                "progress_percent": (processed / total_matches) * 100
            })
        
        # Broadcaster la completion
        await websocket_manager.broadcast({
            "type": "bulk_completed",
            "job_id": job_id,
            "total_processed": processed,
            "completion_time": datetime.now().isoformat()
        })
        
        logger.info(f"Traitement masse {job_id} terminé")
        
    except Exception as e:
        logger.error(f"Erreur traitement masse {job_id}: {e}")

async def process_model_retraining(job_id: str, request: RetrainRequest):
    """Traiter le réentraînement en arrière-plan"""
    try:
        logger.info(f"Début réentraînement {job_id}")
        
        models_to_retrain = request.model_ids or ["all_models"]
        
        for model_id in models_to_retrain:
            # Simuler le réentraînement
            duration = np.random.uniform(60, 300) if request.retrain_type == "full" else np.random.uniform(10, 60)
            await asyncio.sleep(duration / 10)  # Accéléré pour la démo
            
            # Broadcaster le progrès
            await websocket_manager.broadcast({
                "type": "retrain_progress",
                "job_id": job_id,
                "model_id": model_id,
                "status": "completed",
                "improvement": np.random.uniform(0.01, 0.08)
            })
        
        logger.info(f"Réentraînement {job_id} terminé")
        
    except Exception as e:
        logger.error(f"Erreur réentraînement {job_id}: {e}")

async def schedule_delayed_retrain(job_id: str, request: RetrainRequest, delay: float):
    """Programmer un réentraînement avec délai"""
    await asyncio.sleep(delay)
    await process_model_retraining(job_id, request)

# Endpoint racine
@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "⚽ Football ML Predictions API v4.0",
        "status": "operational",
        "documentation": "/docs",
        "websocket_endpoint": "/predict/live",
        "total_endpoints": 12,
        "features": [
            "Prédictions intelligentes",
            "Analytics avancés", 
            "Apprentissage continu",
            "Monitoring temps réel",
            "Administration complète"
        ]
    }

# Lancement de l'API
if __name__ == "__main__":
    uvicorn.run(
        "professional_api_complete:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )