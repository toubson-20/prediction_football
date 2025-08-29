"""
🎯 API AVANCÉE POUR COUPONS DE PARIS
Système complet de génération de coupons par ligue et multi-ligues
Frontend-ready avec documentation complète
"""

from fastapi import FastAPI, HTTPException, Query, Path, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path as PathLib
import joblib
from intelligent_coupon_optimizer import IntelligentCouponOptimizer, MatchOpportunity

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="⚽ Advanced Football Betting API",
    description="API complète pour coupons de paris football par ligue et multi-ligues",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS pour frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production: spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ENUMS ET MODÈLES =====

class LeagueEnum(str, Enum):
    """Ligues supportées"""
    PREMIER_LEAGUE = "premier_league"
    LA_LIGA = "la_liga"
    BUNDESLIGA = "bundesliga"
    SERIE_A = "serie_a"
    LIGUE_1 = "ligue_1"
    CHAMPIONS_LEAGUE = "champions_league"
    EUROPA_LEAGUE = "europa_league"
    ALL_LEAGUES = "all_leagues"

class PredictionTypeEnum(str, Enum):
    """Types de prédictions disponibles"""
    MATCH_RESULT = "match_result"  # 1X2
    BOTH_TEAMS_SCORE = "both_teams_score"
    OVER_2_5_GOALS = "over_2_5_goals"
    UNDER_2_5_GOALS = "under_2_5_goals"
    HOME_GOALS = "home_goals"
    AWAY_GOALS = "away_goals"
    TOTAL_GOALS = "total_goals"
    CLEAN_SHEET = "clean_sheet"
    WIN_PROBABILITY = "win_probability"
    DRAW_PROBABILITY = "draw_probability"

class RiskProfileEnum(str, Enum):
    """Profils de risque"""
    CONSERVATIVE = "conservative"  # Confiance 85%+, cotes 1.2-1.6
    BALANCED = "balanced"         # Confiance 70%+, cotes 1.5-2.5
    AGGRESSIVE = "aggressive"     # Confiance 60%+, cotes 2.0-4.0

class OptimizationStrategyEnum(str, Enum):
    """Stratégies d'optimisation intelligente"""
    BALANCED = "balanced"              # Équilibre rendement/risque
    HIGH_CONFIDENCE = "high_confidence" # Privilégie sécurité
    VALUE_HUNTING = "value_hunting"     # Chasse aux valeurs
    ANTI_CORRELATION = "anti_correlation" # Diversification max
    KELLY_OPTIMAL = "kelly_optimal"     # Optimisation Kelly
    AUTO_OPTIMAL = "auto_optimal"       # IA choisit la meilleure

class CouponTypeEnum(str, Enum):
    """Types de coupon"""
    SINGLE_LEAGUE = "single_league"
    MULTI_LEAGUE = "multi_league"
    MIXED = "mixed"
    TARGET_ODDS = "target_odds"

# ===== MODÈLES PYDANTIC =====

class MatchPrediction(BaseModel):
    """Prédiction pour un match"""
    match_id: int
    home_team: str
    away_team: str
    league: LeagueEnum
    prediction_type: PredictionTypeEnum
    prediction_value: Union[str, float, int]
    confidence: float = Field(..., ge=0, le=100, description="Confiance 0-100%")
    odds: float = Field(..., gt=0, description="Cote associée")
    expected_value: float = Field(..., description="Valeur attendue")
    risk_level: str
    match_date: datetime

class CouponRequest(BaseModel):
    """Requête de génération de coupon"""
    leagues: List[LeagueEnum] = Field(..., description="Ligues à inclure")
    prediction_types: List[PredictionTypeEnum] = Field(default_factory=lambda: [PredictionTypeEnum.MATCH_RESULT], description="Types de prédictions")
    risk_profile: RiskProfileEnum = Field(RiskProfileEnum.BALANCED, description="Profil de risque")
    optimization_strategy: OptimizationStrategyEnum = Field(OptimizationStrategyEnum.AUTO_OPTIMAL, description="Stratégie d'optimisation")
    min_confidence: float = Field(70.0, ge=0, le=100, description="Confiance minimum %")
    max_predictions: int = Field(8, ge=1, le=20, description="Nombre max de prédictions")
    min_odds: float = Field(1.2, gt=0, description="Cote minimum")
    max_odds: float = Field(10.0, gt=0, description="Cote maximum")
    target_date: Optional[datetime] = Field(None, description="Date cible pour les matchs")
    exclude_teams: Optional[List[str]] = Field(default_factory=list, description="Équipes à exclure")
    intelligent_selection: bool = Field(True, description="Utiliser sélection intelligente (recommandé)")
    
    # Paramètres cote cible (optionnels)
    target_odds: Optional[float] = Field(None, gt=1.0, le=1000.0, description="Cote cible souhaitée (optionnel)")
    target_odds_tolerance: float = Field(0.2, ge=0.0, le=1.0, description="Tolérance cote cible (0.2 = ±20%)")
    prioritize_target_success: bool = Field(True, description="Prioriser réussite vs cote exacte")
    
    @field_validator('max_odds')
    @classmethod
    def validate_odds_range(cls, v, info):
        if info.data.get('min_odds') and v <= info.data.get('min_odds'):
            raise ValueError('max_odds doit être supérieur à min_odds')
        return v

class CouponResponse(BaseModel):
    """Réponse de coupon générée"""
    coupon_id: str
    coupon_type: CouponTypeEnum
    predictions: List[MatchPrediction]
    total_odds: float
    expected_return: float
    risk_score: float
    confidence_average: float
    leagues_included: List[LeagueEnum]
    generation_timestamp: datetime
    estimated_win_probability: float
    recommended_stake: Optional[float] = None
    optimization_used: str = "standard"
    quality_score: Optional[float] = None
    kelly_weight: Optional[float] = None
    diversification_score: Optional[float] = None

class MatchPredictionRequest(BaseModel):
    """Requête pour prédiction d'un match spécifique"""
    home_team: str = Field(..., description="Nom équipe domicile")
    away_team: str = Field(..., description="Nom équipe extérieur")
    league: LeagueEnum = Field(..., description="Ligue du match")
    prediction_types: Optional[List[PredictionTypeEnum]] = Field(None, description="Types prédictions spécifiques")

class TargetOddsRequest(BaseModel):
    """Requête pour coupon avec cote cible"""
    target_odds: float = Field(..., gt=1.0, le=1000.0, description="Cote cible souhaitée")
    leagues: List[LeagueEnum] = Field(default_factory=list, description="Ligues à inclure (toutes si vide)")
    tolerance: float = Field(0.2, ge=0.0, le=1.0, description="Tolérance sur la cote (0.2 = ±20%)")
    prioritize_success: bool = Field(True, description="Prioriser la réussite vs cote exacte")
    min_confidence: float = Field(70.0, ge=50.0, le=100.0, description="Confiance minimum")
    max_predictions: int = Field(8, ge=1, le=15, description="Nombre max de prédictions")

class AdvancedFilterRequest(BaseModel):
    """Filtres avancés pour coupons"""
    leagues: List[LeagueEnum] = Field(default_factory=list)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_confidence: float = Field(50.0, ge=0, le=100)
    max_confidence: float = Field(100.0, ge=0, le=100)
    min_odds: float = Field(1.1, gt=0)
    max_odds: float = Field(50.0, gt=0)
    include_home_teams: Optional[List[str]] = None
    include_away_teams: Optional[List[str]] = None
    exclude_teams: Optional[List[str]] = None
    prediction_types: List[PredictionTypeEnum] = Field(default_factory=list)
    sort_by: Optional[str] = Field("confidence", description="Trier par: confidence, odds, expected_value")
    sort_order: Optional[str] = Field("desc", description="Ordre: asc, desc")

# ===== SERVICE PRINCIPAL =====

class AdvancedCouponService:
    """Service de génération de coupons avancés"""
    
    def __init__(self):
        self.models_dir = PathLib("models/complete_models")
        self.data_dir = PathLib("data/ultra_processed")
        self.models = self._load_models()
        self.optimizer = IntelligentCouponOptimizer()
        self.leagues_mapping = {
            LeagueEnum.PREMIER_LEAGUE: 39,
            LeagueEnum.LA_LIGA: 140,
            LeagueEnum.BUNDESLIGA: 78,
            LeagueEnum.SERIE_A: 135,
            LeagueEnum.LIGUE_1: 61,
            LeagueEnum.CHAMPIONS_LEAGUE: 2,
            LeagueEnum.EUROPA_LEAGUE: 3
        }
        
    def _load_models(self) -> Dict:
        """Charger les modèles ML"""
        models = {}
        try:
            if self.models_dir.exists():
                for model_file in self.models_dir.glob("complete_*.joblib"):
                    model_name = model_file.stem
                    models[model_name] = joblib.load(model_file)
                logger.info(f"Chargé {len(models)} modèles ML")
            return models
        except Exception as e:
            logger.error(f"Erreur chargement modèles: {e}")
            return {}
    
    def _get_mock_predictions(self, leagues: List[LeagueEnum], filters: Dict) -> List[Dict]:
        """Générer prédictions mock réalistes pour démo"""
        predictions = []
        
        teams_by_league = {
            LeagueEnum.PREMIER_LEAGUE: [
                ("Manchester City", "Liverpool"), ("Arsenal", "Chelsea"), 
                ("Manchester United", "Tottenham"), ("Newcastle", "Brighton")
            ],
            LeagueEnum.LA_LIGA: [
                ("Real Madrid", "Barcelona"), ("Atletico Madrid", "Sevilla"),
                ("Valencia", "Real Sociedad"), ("Villarreal", "Athletic Bilbao")
            ],
            LeagueEnum.BUNDESLIGA: [
                ("Bayern Munich", "Borussia Dortmund"), ("RB Leipzig", "Bayer Leverkusen"),
                ("Eintracht Frankfurt", "VfL Wolfsburg"), ("Borussia Monchengladbach", "SC Freiburg")
            ],
            LeagueEnum.SERIE_A: [
                ("Juventus", "Inter Milan"), ("AC Milan", "Napoli"),
                ("Roma", "Lazio"), ("Atalanta", "Fiorentina")
            ],
            LeagueEnum.LIGUE_1: [
                ("PSG", "Marseille"), ("Monaco", "Lyon"),
                ("Nice", "Lille"), ("Rennes", "Montpellier")
            ]
        }
        
        for league in leagues:
            if league == LeagueEnum.ALL_LEAGUES:
                continue
                
            matches = teams_by_league.get(league, [])
            
            for i, (home, away) in enumerate(matches):
                # Générer prédictions variées
                base_confidence = np.random.uniform(60, 95)
                base_odds = np.random.uniform(1.2, 4.0)
                
                # Sélection aléatoire du type de prédiction (sans numpy)
                pred_types = [PredictionTypeEnum.MATCH_RESULT, PredictionTypeEnum.BOTH_TEAMS_SCORE, 
                             PredictionTypeEnum.OVER_2_5_GOALS, PredictionTypeEnum.CLEAN_SHEET]
                selected_type = pred_types[i % len(pred_types)]
                
                prediction = {
                    "match_id": hash(f"{league}_{home}_{away}") % 100000,
                    "home_team": home,
                    "away_team": away,
                    "league": league,
                    "prediction_type": selected_type,
                    "confidence": round(base_confidence, 1),
                    "odds": round(base_odds, 2),
                    "expected_value": round((base_odds * base_confidence / 100) - 1, 3),
                    "match_date": datetime.now() + timedelta(days=int(np.random.randint(1, 7)))
                }
                
                # Appliquer filtres
                if (filters.get('min_confidence', 0) <= prediction['confidence'] <= 
                    filters.get('max_confidence', 100) and
                    filters.get('min_odds', 0) <= prediction['odds'] <= 
                    filters.get('max_odds', 100)):
                    predictions.append(prediction)
        
        return predictions
    
    def _generate_match_predictions(self, home_team: str, away_team: str, 
                                   league: LeagueEnum, 
                                   prediction_types: Optional[List[PredictionTypeEnum]] = None) -> List[MatchPrediction]:
        """Générer prédictions pour un match spécifique"""
        predictions = []
        
        # Types par défaut si non spécifiés
        if not prediction_types:
            prediction_types = [
                PredictionTypeEnum.MATCH_RESULT,
                PredictionTypeEnum.BOTH_TEAMS_SCORE,
                PredictionTypeEnum.OVER_2_5_GOALS,
                PredictionTypeEnum.CLEAN_SHEET
            ]
        
        for pred_type in prediction_types:
            # Générer données mock basées sur le match
            base_confidence = np.random.uniform(70, 95)
            base_odds = np.random.uniform(1.3, 4.5)
            
            # Logique spécifique par type
            if pred_type == PredictionTypeEnum.MATCH_RESULT:
                values = ["1", "X", "2"] 
                prediction_value = values[hash(f"{home_team}{away_team}") % 3]
            elif pred_type == PredictionTypeEnum.BOTH_TEAMS_SCORE:
                prediction_value = "Oui" if hash(home_team) % 2 else "Non"
            elif pred_type == PredictionTypeEnum.OVER_2_5_GOALS:
                prediction_value = "Oui" if hash(away_team) % 2 else "Non"  
            elif pred_type == PredictionTypeEnum.CLEAN_SHEET:
                prediction_value = home_team if hash(f"{home_team}{league}") % 2 else "Non"
            else:
                prediction_value = "N/A"
            
            # Calculer valeur attendue et niveau risque
            expected_value = round((base_confidence / 100) * base_odds - 1, 3)
            risk_level = "Faible" if base_confidence > 80 else "Moyen" if base_confidence > 70 else "Élevé"
            
            prediction = MatchPrediction(
                match_id=hash(f"{league}_{home_team}_{away_team}") % 100000,
                home_team=home_team,
                away_team=away_team,
                league=league,
                prediction_type=pred_type,
                prediction_value=prediction_value,
                confidence=round(base_confidence, 1),
                odds=round(base_odds, 2),
                expected_value=expected_value,
                risk_level=risk_level,
                match_date=datetime.now() + timedelta(days=int(np.random.randint(1, 7)))
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def generate_target_odds_coupon(self, request: TargetOddsRequest) -> CouponResponse:
        """Générer coupon visant une cote spécifique avec optimisation réussite"""
        logger.info(f"Génération coupon cote cible: {request.target_odds}")
        
        # Déterminer ligues à utiliser
        leagues = request.leagues if request.leagues else [
            LeagueEnum.PREMIER_LEAGUE, LeagueEnum.LA_LIGA, LeagueEnum.BUNDESLIGA,
            LeagueEnum.SERIE_A, LeagueEnum.LIGUE_1
        ]
        
        # Obtenir pool de prédictions
        filters = {
            'min_confidence': request.min_confidence,
            'max_confidence': 100.0,
            'min_odds': 1.2,
            'max_odds': 10.0
        }
        
        all_predictions_data = self._get_mock_predictions(leagues, filters)
        
        # Convertir en objets MatchPrediction
        all_predictions = []
        for pred_data in all_predictions_data:
            prediction = MatchPrediction(
                match_id=pred_data['match_id'],
                home_team=pred_data['home_team'], 
                away_team=pred_data['away_team'],
                league=pred_data['league'],
                prediction_type=pred_data['prediction_type'],
                prediction_value=str(pred_data.get('prediction_value', 'N/A')),
                confidence=pred_data['confidence'],
                odds=pred_data['odds'],
                expected_value=pred_data['expected_value'],
                risk_level="Faible" if pred_data['confidence'] > 80 else "Moyen",
                match_date=pred_data['match_date']
            )
            all_predictions.append(prediction)
        
        # Trier par qualité (confiance * valeur attendue)
        all_predictions.sort(key=lambda p: p.confidence * p.expected_value, reverse=True)
        
        # Algorithme de sélection optimale pour cote cible
        selected_predictions = self._select_optimal_for_target_odds(
            all_predictions, request.target_odds, request.tolerance, 
            request.prioritize_success, request.max_predictions
        )
        
        if not selected_predictions:
            # Fallback: prendre les meilleures prédictions disponibles
            selected_predictions = all_predictions[:min(3, request.max_predictions)]
        
        # Calculer métriques
        total_odds = np.prod([p.odds for p in selected_predictions])
        confidence_avg = sum(p.confidence for p in selected_predictions) / len(selected_predictions)
        win_probability = self._estimate_win_probability(selected_predictions)
        risk_score = self._calculate_coupon_risk(selected_predictions)
        
        # Évaluer proximité cote cible
        odds_deviation = abs(total_odds - request.target_odds) / request.target_odds
        target_achievement = max(0, 100 - (odds_deviation * 100))
        
        coupon_id = f"TO_{datetime.now().strftime('%Y%m%d_%H%M%S')}_TARGET"
        
        return CouponResponse(
            coupon_id=coupon_id,
            coupon_type=CouponTypeEnum.TARGET_ODDS,
            predictions=selected_predictions,
            total_odds=round(total_odds, 2),
            confidence_average=round(confidence_avg, 2),
            estimated_win_probability=round(win_probability, 2),
            expected_return=round(total_odds * (win_probability/100), 2),
            risk_score=round(risk_score, 1),
            leagues_included=[p.league.value for p in selected_predictions],
            generation_timestamp=datetime.now()
        )
    
    def _select_optimal_for_target_odds(self, predictions: List[MatchPrediction], 
                                       target_odds: float, tolerance: float,
                                       prioritize_success: bool, max_predictions: int) -> List[MatchPrediction]:
        """Sélectionner combinaison optimale pour atteindre cote cible"""
        best_combination = []
        best_score = 0
        
        # Générer combinaisons possibles
        from itertools import combinations
        
        for size in range(1, min(max_predictions + 1, len(predictions) + 1)):
            for combo in combinations(predictions[:min(20, len(predictions))], size):
                combo_odds = np.prod([p.odds for p in combo])
                combo_confidence = sum(p.confidence for p in combo) / len(combo)
                
                # Vérifier si dans la tolérance
                odds_diff = abs(combo_odds - target_odds) / target_odds
                if odds_diff > tolerance:
                    continue
                
                # Calculer score de qualité
                if prioritize_success:
                    # 70% poids sur confiance, 30% sur proximité cote
                    proximity_score = max(0, 100 - (odds_diff * 100))
                    quality_score = (combo_confidence * 0.7) + (proximity_score * 0.3)
                else:
                    # 30% poids sur confiance, 70% sur proximité cote  
                    proximity_score = max(0, 100 - (odds_diff * 100))
                    quality_score = (combo_confidence * 0.3) + (proximity_score * 0.7)
                
                if quality_score > best_score:
                    best_score = quality_score
                    best_combination = list(combo)
        
        return best_combination
    
    def generate_league_coupon(self, league: LeagueEnum, request: CouponRequest) -> CouponResponse:
        """Générer coupon pour une ligue spécifique"""
        logger.info(f"Génération coupon {league}")
        
        # Si cote cible spécifiée, utiliser l'algorithme spécialisé
        if request.target_odds:
            target_request = TargetOddsRequest(
                target_odds=request.target_odds,
                leagues=[league],
                tolerance=request.target_odds_tolerance,
                prioritize_success=request.prioritize_target_success,
                min_confidence=request.min_confidence,
                max_predictions=request.max_predictions
            )
            return self.generate_target_odds_coupon(target_request)
        
        # Logique normale sans cote cible
        # Préparer filtres
        filters = {
            'min_confidence': request.min_confidence,
            'max_confidence': 100.0,
            'min_odds': request.min_odds,
            'max_odds': request.max_odds
        }
        
        # Obtenir prédictions
        predictions_data = self._get_mock_predictions([league], filters)
        
        # Appliquer profil de risque
        predictions_data = self._apply_risk_profile(predictions_data, request.risk_profile)
        
        # Limiter nombre
        predictions_data = predictions_data[:request.max_predictions]
        
        # Convertir en objets Pydantic
        predictions = []
        total_odds = 1.0
        
        for pred_data in predictions_data:
            prediction = MatchPrediction(
                match_id=pred_data["match_id"],
                home_team=pred_data["home_team"],
                away_team=pred_data["away_team"],
                league=pred_data["league"],
                prediction_type=pred_data["prediction_type"],
                prediction_value=self._get_prediction_value(pred_data["prediction_type"]),
                confidence=pred_data["confidence"],
                odds=pred_data["odds"],
                expected_value=pred_data["expected_value"],
                risk_level=self._calculate_risk_level(pred_data["confidence"], pred_data["odds"]),
                match_date=pred_data["match_date"]
            )
            predictions.append(prediction)
            total_odds *= prediction.odds
        
        # Calculer métriques
        avg_confidence = np.mean([p.confidence for p in predictions]) if predictions else 0
        risk_score = self._calculate_coupon_risk(predictions)
        win_probability = self._estimate_win_probability(predictions)
        
        return CouponResponse(
            coupon_id=f"LC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{league}",
            coupon_type=CouponTypeEnum.SINGLE_LEAGUE,
            predictions=predictions,
            total_odds=round(total_odds, 2),
            expected_return=round((total_odds * win_probability / 100) - 1, 3),
            risk_score=risk_score,
            confidence_average=round(avg_confidence, 1),
            leagues_included=[league],
            generation_timestamp=datetime.now(),
            estimated_win_probability=win_probability
        )
    
    def generate_multi_league_coupon(self, request: CouponRequest) -> CouponResponse:
        """Générer coupon multi-ligues avec optimisation intelligente"""
        logger.info(f"Génération coupon multi-ligues: {request.leagues}")
        
        # Si cote cible spécifiée, utiliser l'algorithme spécialisé
        if request.target_odds:
            target_request = TargetOddsRequest(
                target_odds=request.target_odds,
                leagues=request.leagues,
                tolerance=request.target_odds_tolerance,
                prioritize_success=request.prioritize_target_success,
                min_confidence=request.min_confidence,
                max_predictions=request.max_predictions
            )
            return self.generate_target_odds_coupon(target_request)
        
        # Logique normale multi-ligues
        # Filtres
        filters = {
            'min_confidence': request.min_confidence,
            'max_confidence': 100.0,
            'min_odds': request.min_odds,
            'max_odds': request.max_odds
        }
        
        # Obtenir toutes les prédictions disponibles
        all_predictions = self._get_mock_predictions(request.leagues, filters)
        
        if request.intelligent_selection and len(all_predictions) > request.max_predictions:
            # OPTIMISATION INTELLIGENTE
            logger.info("Utilisation de l'optimisation intelligente")
            
            # Convertir en opportunités pour l'optimiseur
            opportunities = self.optimizer.generate_match_opportunities(all_predictions, filters)
            
            # Déterminer stratégie optimale
            strategy = request.optimization_strategy
            if strategy == OptimizationStrategyEnum.AUTO_OPTIMAL:
                # L'IA choisit la meilleure stratégie selon le contexte
                strategy = self._determine_optimal_strategy(opportunities, request)
            
            # Suggérer taille optimale si pas spécifiée intelligemment
            target_size = min(request.max_predictions, 
                            self.optimizer.suggest_optimal_coupon_size(opportunities))
            
            # Optimisation intelligente
            selected_opportunities = self.optimizer.optimize_coupon_smart(
                opportunities, target_size, strategy.value
            )
            
            # Calculer métriques avancées
            optimizer_metrics = self.optimizer.calculate_coupon_metrics(selected_opportunities)
            
            # Convertir en prédictions API
            predictions = []
            for opp in selected_opportunities:
                prediction = MatchPrediction(
                    match_id=opp.match_id,
                    home_team=opp.home_team,
                    away_team=opp.away_team,
                    league=getattr(LeagueEnum, opp.league.upper()) if hasattr(LeagueEnum, opp.league.upper()) else LeagueEnum.PREMIER_LEAGUE,
                    prediction_type=getattr(PredictionTypeEnum, opp.prediction_type.upper()) if hasattr(PredictionTypeEnum, opp.prediction_type.upper()) else PredictionTypeEnum.MATCH_RESULT,
                    prediction_value=opp.prediction_value,
                    confidence=opp.confidence,
                    odds=opp.odds,
                    expected_value=opp.expected_value,
                    risk_level=self._calculate_risk_level(opp.confidence, opp.odds),
                    match_date=datetime.now() + timedelta(days=1)
                )
                predictions.append(prediction)
            
            # Utiliser métriques optimisées
            total_odds = optimizer_metrics.get('total_odds', 1.0)
            win_probability = optimizer_metrics.get('win_probability', 0)
            expected_return = optimizer_metrics.get('expected_return', 0)
            avg_confidence = optimizer_metrics.get('avg_confidence', 0)
            leagues_used = list(set([p.league for p in predictions]))
            
            return CouponResponse(
                coupon_id=f"OPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_SMART",
                coupon_type=CouponTypeEnum.MULTI_LEAGUE,
                predictions=predictions,
                total_odds=total_odds,
                expected_return=expected_return,
                risk_score=round((100 - avg_confidence) / 2, 1),
                confidence_average=avg_confidence,
                leagues_included=leagues_used,
                generation_timestamp=datetime.now(),
                estimated_win_probability=win_probability,
                recommended_stake=optimizer_metrics.get('recommended_stake'),
                optimization_used=f"{strategy.value}_intelligent",
                quality_score=optimizer_metrics.get('quality_score'),
                kelly_weight=optimizer_metrics.get('kelly_weight'),
                diversification_score=optimizer_metrics.get('diversification_score')
            )
        
        else:
            # MÉTHODE STANDARD (fallback)
            logger.info("Utilisation de la sélection standard")
            return self._generate_standard_coupon(all_predictions, request)
    
    def _determine_optimal_strategy(self, opportunities: List[MatchOpportunity], 
                                  request: CouponRequest) -> OptimizationStrategyEnum:
        """Déterminer automatiquement la meilleure stratégie"""
        
        # Analyser qualité des opportunités
        high_quality_count = len([op for op in opportunities if op.optimization_score > 0.7])
        avg_confidence = np.mean([op.confidence for op in opportunities])
        avg_expected_value = np.mean([op.expected_value for op in opportunities])
        
        # Logique de décision intelligente
        if request.risk_profile == RiskProfileEnum.CONSERVATIVE:
            return OptimizationStrategyEnum.HIGH_CONFIDENCE
        
        elif request.risk_profile == RiskProfileEnum.AGGRESSIVE:
            if avg_expected_value > 0.2:
                return OptimizationStrategyEnum.VALUE_HUNTING
            else:
                return OptimizationStrategyEnum.KELLY_OPTIMAL
        
        else:  # BALANCED
            if high_quality_count >= 8:
                return OptimizationStrategyEnum.ANTI_CORRELATION  # Diversifier quand beaucoup d'options
            elif avg_confidence > 85:
                return OptimizationStrategyEnum.HIGH_CONFIDENCE
            elif avg_expected_value > 0.15:
                return OptimizationStrategyEnum.VALUE_HUNTING
            else:
                return OptimizationStrategyEnum.BALANCED
    
    def _generate_standard_coupon(self, all_predictions: List[Dict], 
                                request: CouponRequest) -> CouponResponse:
        """Générer coupon avec méthode standard (fallback)"""
        
        # Équilibrer par ligue
        predictions_per_league = max(1, request.max_predictions // len(request.leagues))
        balanced_predictions = []
        
        for league in request.leagues:
            if league == LeagueEnum.ALL_LEAGUES:
                continue
            league_preds = [p for p in all_predictions if p['league'] == league]
            league_preds = self._apply_risk_profile(league_preds, request.risk_profile)
            balanced_predictions.extend(league_preds[:predictions_per_league])
        
        # Compléter si nécessaire
        remaining_slots = request.max_predictions - len(balanced_predictions)
        if remaining_slots > 0:
            remaining_preds = [p for p in all_predictions if p not in balanced_predictions]
            remaining_preds = self._apply_risk_profile(remaining_preds, request.risk_profile)
            balanced_predictions.extend(remaining_preds[:remaining_slots])
        
        # Convertir en objets Pydantic
        predictions = []
        total_odds = 1.0
        
        for pred_data in balanced_predictions:
            prediction = MatchPrediction(
                match_id=pred_data["match_id"],
                home_team=pred_data["home_team"],
                away_team=pred_data["away_team"],
                league=pred_data["league"],
                prediction_type=pred_data["prediction_type"],
                prediction_value=self._get_prediction_value(pred_data["prediction_type"]),
                confidence=pred_data["confidence"],
                odds=pred_data["odds"],
                expected_value=pred_data["expected_value"],
                risk_level=self._calculate_risk_level(pred_data["confidence"], pred_data["odds"]),
                match_date=pred_data["match_date"]
            )
            predictions.append(prediction)
            total_odds *= prediction.odds
        
        # Calculer métriques
        avg_confidence = np.mean([p.confidence for p in predictions]) if predictions else 0
        risk_score = self._calculate_coupon_risk(predictions)
        win_probability = self._estimate_win_probability(predictions)
        leagues_used = list(set([p.league for p in predictions]))
        
        return CouponResponse(
            coupon_id=f"MC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_MULTI",
            coupon_type=CouponTypeEnum.MULTI_LEAGUE,
            predictions=predictions,
            total_odds=round(total_odds, 2),
            expected_return=round((total_odds * win_probability / 100) - 1, 3),
            risk_score=risk_score,
            confidence_average=round(avg_confidence, 1),
            leagues_included=leagues_used,
            generation_timestamp=datetime.now(),
            estimated_win_probability=win_probability,
            optimization_used="standard"
        )
    
    def _apply_risk_profile(self, predictions: List[Dict], profile: RiskProfileEnum) -> List[Dict]:
        """Appliquer profil de risque"""
        if profile == RiskProfileEnum.CONSERVATIVE:
            return [p for p in predictions if p['confidence'] >= 85 and 1.2 <= p['odds'] <= 1.8]
        elif profile == RiskProfileEnum.BALANCED:
            return [p for p in predictions if p['confidence'] >= 70 and 1.4 <= p['odds'] <= 2.5]
        else:  # AGGRESSIVE
            return [p for p in predictions if p['confidence'] >= 60 and 1.8 <= p['odds'] <= 4.0]
    
    def _get_prediction_value(self, pred_type: PredictionTypeEnum) -> str:
        """Obtenir valeur de prédiction selon le type"""
        import random
        values = {
            PredictionTypeEnum.MATCH_RESULT: random.choice(["1", "X", "2"]),
            PredictionTypeEnum.BOTH_TEAMS_SCORE: random.choice(["Oui", "Non"]),
            PredictionTypeEnum.OVER_2_5_GOALS: "Plus de 2.5",
            PredictionTypeEnum.UNDER_2_5_GOALS: "Moins de 2.5",
            PredictionTypeEnum.CLEAN_SHEET: random.choice(["Domicile", "Extérieur"]),
        }
        return values.get(pred_type, "Prédiction")
    
    def _calculate_risk_level(self, confidence: float, odds: float) -> str:
        """Calculer niveau de risque"""
        risk_score = (100 - confidence) * (odds / 2)
        if risk_score <= 30:
            return "Faible"
        elif risk_score <= 60:
            return "Moyen"
        else:
            return "Élevé"
    
    def _calculate_coupon_risk(self, predictions: List[MatchPrediction]) -> float:
        """Calculer score de risque du coupon"""
        if not predictions:
            return 0.0
        
        risk_scores = []
        for pred in predictions:
            risk = (100 - pred.confidence) * (pred.odds / 2) / 100
            risk_scores.append(risk)
        
        return round(np.mean(risk_scores) * 100, 1)
    
    def _estimate_win_probability(self, predictions: List[MatchPrediction]) -> float:
        """Estimer probabilité de gain du coupon"""
        if not predictions:
            return 0.0
        
        # Probabilité combinée (indépendante)
        combined_prob = 1.0
        for pred in predictions:
            pred_prob = pred.confidence / 100
            combined_prob *= pred_prob
        
        return round(combined_prob * 100, 1)

# Instance globale du service
coupon_service = AdvancedCouponService()

# ===== ENDPOINTS API =====

@app.get("/", tags=["Info"])
async def root():
    """Point d'entrée de l'API"""
    return {
        "service": "Advanced Football Betting API",
        "version": "2.0.0",
        "status": "active",
        "endpoints": {
            "docs": "/docs",
            "coupons": {
                "single_league": "/coupons/league/{league}",
                "multi_league": "/coupons/multi-league",
                "all_leagues": "/coupons/all-leagues"
            },
            "leagues": "/leagues",
            "predictions": "/predictions"
        }
    }

@app.get("/leagues", tags=["Configuration"])
async def get_available_leagues():
    """Obtenir la liste des ligues disponibles"""
    return {
        "available_leagues": [
            {
                "code": league.value,
                "name": league.value.replace("_", " ").title(),
                "supported": True
            }
            for league in LeagueEnum if league != LeagueEnum.ALL_LEAGUES
        ],
        "total_leagues": len(LeagueEnum) - 1
    }

@app.post("/coupons/league/{league}", 
          response_model=CouponResponse,
          tags=["Coupons"],
          summary="Générer coupon pour une ligue spécifique")
async def generate_league_coupon(
    league: LeagueEnum = Path(..., description="Ligue pour le coupon"),
    request: CouponRequest = None
):
    """
    Générer un coupon de paris pour une ligue spécifique.
    
    - **league**: Ligue cible (premier_league, la_liga, bundesliga, etc.)
    - **request**: Paramètres du coupon (profil de risque, confiance, etc.)
    """
    if not request:
        request = CouponRequest(leagues=[league])
    
    try:
        coupon = coupon_service.generate_league_coupon(league, request)
        logger.info(f"Coupon {league} généré: {len(coupon.predictions)} prédictions")
        return coupon
    except Exception as e:
        logger.error(f"Erreur génération coupon {league}: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur génération coupon: {str(e)}")

@app.post("/coupons/multi-league",
          response_model=CouponResponse,
          tags=["Coupons"],
          summary="Générer coupon multi-ligues")
async def generate_multi_league_coupon(request: CouponRequest):
    """
    Générer un coupon de paris combinant plusieurs ligues.
    
    - **leagues**: Liste des ligues à inclure
    - **Répartition équilibrée**: Prédictions réparties entre les ligues
    """
    if not request.leagues:
        raise HTTPException(status_code=400, detail="Au moins une ligue doit être spécifiée")
    
    try:
        coupon = coupon_service.generate_multi_league_coupon(request)
        logger.info(f"Coupon multi-ligues généré: {len(coupon.predictions)} prédictions")
        return coupon
    except Exception as e:
        logger.error(f"Erreur génération coupon multi-ligues: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur génération coupon: {str(e)}")

@app.post("/coupons/all-leagues",
          response_model=CouponResponse,
          tags=["Coupons"],
          summary="Générer coupon toutes ligues confondues")
async def generate_all_leagues_coupon(
    risk_profile: RiskProfileEnum = Query(RiskProfileEnum.BALANCED),
    optimization_strategy: OptimizationStrategyEnum = Query(OptimizationStrategyEnum.AUTO_OPTIMAL),
    min_confidence: float = Query(70.0, ge=0, le=100),
    max_predictions: int = Query(10, ge=1, le=20),
    min_odds: float = Query(1.2, gt=0),
    max_odds: float = Query(10.0, gt=0),
    intelligent_selection: bool = Query(True, description="Utiliser optimisation intelligente"),
    # Nouveaux paramètres pour cote cible
    target_odds: Optional[float] = Query(None, gt=1.0, le=1000.0, description="Cote cible souhaitée (optionnel)"),
    target_odds_tolerance: float = Query(0.2, ge=0.0, le=1.0, description="Tolérance cote cible (0.2 = ±20%)"),
    prioritize_target_success: bool = Query(True, description="Prioriser réussite vs cote exacte")
):
    """
    Générer un coupon incluant toutes les ligues disponibles.
    
    - **Optimisation automatique**: Sélectionne les meilleures prédictions
    - **Diversification**: Équilibre entre les ligues
    """
    all_leagues = [league for league in LeagueEnum if league != LeagueEnum.ALL_LEAGUES]
    
    request = CouponRequest(
        leagues=all_leagues,
        risk_profile=risk_profile,
        optimization_strategy=optimization_strategy,
        min_confidence=min_confidence,
        max_predictions=max_predictions,
        min_odds=min_odds,
        max_odds=max_odds,
        intelligent_selection=intelligent_selection,
        target_odds=target_odds,
        target_odds_tolerance=target_odds_tolerance,
        prioritize_target_success=prioritize_target_success
    )
    
    try:
        coupon = coupon_service.generate_multi_league_coupon(request)
        coupon.coupon_type = CouponTypeEnum.MIXED
        coupon.coupon_id = f"ALL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Coupon toutes ligues généré: {len(coupon.predictions)} prédictions")
        return coupon
    except Exception as e:
        logger.error(f"Erreur génération coupon toutes ligues: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur génération coupon: {str(e)}")

@app.get("/predictions/types", tags=["Configuration"])
async def get_prediction_types():
    """Obtenir les types de prédictions disponibles"""
    return {
        "prediction_types": [
            {
                "code": pred_type.value,
                "name": pred_type.value.replace("_", " ").title(),
                "description": _get_prediction_description(pred_type)
            }
            for pred_type in PredictionTypeEnum
        ]
    }

@app.get("/optimization/strategies", tags=["Configuration"])
async def get_optimization_strategies():
    """Obtenir les stratégies d'optimisation disponibles"""
    return {
        "optimization_strategies": [
            {
                "code": strategy.value,
                "name": strategy.value.replace("_", " ").title(),
                "description": _get_strategy_description(strategy)
            }
            for strategy in OptimizationStrategyEnum
        ]
    }

@app.post("/coupons/optimized",
          response_model=CouponResponse,
          tags=["Coupons"],
          summary="Générer coupon avec optimisation avancée")
async def generate_optimized_coupon(request: CouponRequest):
    """
    Générer un coupon avec optimisation intelligente avancée.
    
    - **Sélection stratégique** : IA choisit les meilleurs matchs
    - **6 stratégies** d'optimisation disponibles
    - **Métriques avancées** : Kelly, Sharpe, diversification
    - **Taille adaptive** : Suggestion automatique taille optimale
    """
    try:
        # Forcer optimisation intelligente
        request.intelligent_selection = True
        
        coupon = coupon_service.generate_multi_league_coupon(request)
        
        logger.info(f"Coupon optimisé généré: {len(coupon.predictions)} prédictions, "
                   f"qualité {coupon.quality_score}/100")
        
        return coupon
        
    except Exception as e:
        logger.error(f"Erreur génération coupon optimisé: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur génération coupon: {str(e)}")

@app.post("/predictions/match", tags=["Predictions"])
async def get_match_prediction(request: MatchPredictionRequest):
    """Prédictions pour un match spécifique"""
    try:
        # Générer prédictions pour le match
        predictions = coupon_service._generate_match_predictions(
            request.home_team, request.away_team, request.league, request.prediction_types
        )
        
        if not predictions:
            raise HTTPException(
                status_code=404,
                detail=f"Aucune prédiction trouvée pour {request.home_team} vs {request.away_team}"
            )
        
        # Calculer métriques globales
        avg_confidence = sum(p.confidence for p in predictions) / len(predictions)
        total_odds = np.prod([p.odds for p in predictions])
        
        return {
            "match": f"{request.home_team} vs {request.away_team}",
            "league": request.league,
            "predictions": [p.dict() for p in predictions],
            "total_predictions": len(predictions),
            "average_confidence": round(avg_confidence, 2),
            "combined_odds": round(total_odds, 2),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Erreur prédiction match: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/coupons/target-odds", tags=["Coupons"])
async def generate_target_odds_coupon(request: TargetOddsRequest):
    """Générer coupon visant une cote spécifique avec optimisation réussite"""
    try:
        coupon = coupon_service.generate_target_odds_coupon(request)
        
        # Calculer métriques supplémentaires
        target_achievement = 100 - (abs(coupon.total_odds - request.target_odds) / request.target_odds * 100)
        odds_deviation = abs(coupon.total_odds - request.target_odds) / request.target_odds * 100
        
        response_data = coupon.dict()
        response_data.update({
            "target_odds": request.target_odds,
            "target_achievement": round(max(0, target_achievement), 1),
            "odds_deviation": round(odds_deviation, 1),
            "prioritized_success": request.prioritize_success,
            "tolerance_used": request.tolerance * 100
        })
        
        return response_data
        
    except Exception as e:
        logger.error(f"Erreur génération coupon cote cible: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur génération coupon: {str(e)}")

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "models_loaded": len(coupon_service.models),
        "leagues_available": len(LeagueEnum) - 1
    }

def _get_prediction_description(pred_type: PredictionTypeEnum) -> str:
    """Obtenir description du type de prédiction"""
    descriptions = {
        PredictionTypeEnum.MATCH_RESULT: "Résultat du match (1X2)",
        PredictionTypeEnum.BOTH_TEAMS_SCORE: "Les deux équipes marquent",
        PredictionTypeEnum.OVER_2_5_GOALS: "Plus de 2.5 buts dans le match",
        PredictionTypeEnum.UNDER_2_5_GOALS: "Moins de 2.5 buts dans le match",
        PredictionTypeEnum.HOME_GOALS: "Nombre de buts de l'équipe domicile",
        PredictionTypeEnum.AWAY_GOALS: "Nombre de buts de l'équipe extérieur",
        PredictionTypeEnum.TOTAL_GOALS: "Nombre total de buts",
        PredictionTypeEnum.CLEAN_SHEET: "Une équipe garde sa cage inviolée",
        PredictionTypeEnum.WIN_PROBABILITY: "Probabilité de victoire",
        PredictionTypeEnum.DRAW_PROBABILITY: "Probabilité de match nul"
    }
    return descriptions.get(pred_type, "Prédiction football")

def _get_strategy_description(strategy: OptimizationStrategyEnum) -> str:
    """Obtenir description de la stratégie d'optimisation"""
    descriptions = {
        OptimizationStrategyEnum.BALANCED: "Équilibre optimal entre rendement et risque",
        OptimizationStrategyEnum.HIGH_CONFIDENCE: "Privilégie les prédictions à haute confiance (sécurité)",
        OptimizationStrategyEnum.VALUE_HUNTING: "Recherche les meilleures opportunités de valeur",
        OptimizationStrategyEnum.ANTI_CORRELATION: "Maximise la diversification (minimise corrélations)",
        OptimizationStrategyEnum.KELLY_OPTIMAL: "Optimise selon le critère de Kelly (théorie des jeux)",
        OptimizationStrategyEnum.AUTO_OPTIMAL: "IA choisit automatiquement la meilleure stratégie"
    }
    return descriptions.get(strategy, "Stratégie d'optimisation")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "advanced_coupon_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )