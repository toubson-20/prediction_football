"""
🤖🧠 HYBRID ML-AI SYSTEM - INTÉGRATION COMPLÈTE
Système hybride qui combine nos 180 modèles ML spécialisés avec l'intelligence GPT
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import sys
import os

# Ajouter le dossier parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))
from config import config

# Imports de nos composants ML existants
from intelligent_betting_coupon import IntelligentBettingCoupon
from portfolio_optimization_engine import AdvancedPortfolioOptimizer
from confidence_scoring_engine import AdvancedConfidenceScorer
from realtime_recalibration_engine import RealtimeRecalibrationEngine

# Import de la nouvelle couche IA
from ai_enhancement_layer import AIEnhancementLayer, AIPredictionExplanation

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridMLAIPredictor:
    """Prédicteur hybride ML + IA"""
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialiser le système hybride
        
        Args:
            openai_api_key: Clé API OpenAI (optionnel, utilise .env par défaut)
        """
        # Utiliser la configuration depuis .env si pas de clé fournie
        if openai_api_key is None:
            if not config.validate_ai_config():
                logger.warning("Configuration IA invalide, mode fallback activé")
                self.ai_enabled = False
            else:
                openai_api_key = config.OPENAI_API_KEY
                self.ai_enabled = config.AI_ENABLED
        else:
            self.ai_enabled = True
        # ===== COMPOSANTS ML EXISTANTS (CŒUR DU SYSTÈME) =====
        self.coupon_generator = IntelligentBettingCoupon()
        self.portfolio_optimizer = AdvancedPortfolioOptimizer()
        self.confidence_scorer = AdvancedConfidenceScorer()
        self.recalibration_engine = RealtimeRecalibrationEngine()
        
        # ===== NOUVELLE COUCHE IA (ENHANCEMENT) =====
        if self.ai_enabled and openai_api_key:
            self.ai_layer = AIEnhancementLayer(openai_api_key, config.OPENAI_MODEL)
        else:
            self.ai_layer = None
        
        # Configuration du système hybride depuis .env
        self.ai_fallback = config.AI_FALLBACK_ENABLED
        
        logger.info("🤖🧠 Système Hybride ML-IA initialisé")
    
    async def predict_with_ai_enhancement(self, match_data: Dict, team_stats: Dict, 
                                        market_odds: List[Dict] = None) -> Dict[str, Any]:
        """
        Prédiction complète avec enhancement IA
        
        Args:
            match_data: Données du match
            team_stats: Statistiques des équipes
            market_odds: Cotes du marché (optionnel)
            
        Returns:
            Prédiction enrichie ML + IA
        """
        prediction_start = datetime.now()
        
        try:
            logger.info(f"🎯 Prédiction hybride pour match {match_data.get('match_id', 'N/A')}")
            
            # ===== ÉTAPE 1: ENHANCEMENT DES FEATURES PAR IA =====
            ai_features = {}
            confidence_boost = 0.0
            
            if self.ai_enabled and self.ai_layer:
                try:
                    enhanced_features = await self.ai_layer.feature_enhancer.enhance_features(
                        match_data, team_stats
                    )
                    ai_features = enhanced_features.ai_features
                    confidence_boost = enhanced_features.confidence_boost
                    logger.info(f"✅ Features enrichies par IA: +{confidence_boost}% confiance")
                except Exception as e:
                    logger.warning(f"⚠️ IA features indisponible: {e}")
                    if not self.ai_fallback:
                        raise
            
            # ===== ÉTAPE 2: PRÉDICTION ML AVEC FEATURES ENRICHIES =====
            # Combiner features originales + IA
            enhanced_team_stats = {**team_stats, **ai_features}
            
            # Nos modèles ML font la prédiction (système existant)
            ml_prediction = await self._generate_ml_prediction(match_data, enhanced_team_stats)
            
            # Appliquer le boost de confiance IA
            if confidence_boost > 0:
                ml_prediction['confidence'] = min(95, ml_prediction['confidence'] + confidence_boost)
                ml_prediction['ai_enhanced'] = True
            
            logger.info(f"🧠 Prédiction ML: {ml_prediction['prediction_value']} ({ml_prediction['confidence']:.1f}%)")
            
            # ===== ÉTAPE 3: EXPLICATION IA DE LA PRÉDICTION =====
            ai_explanation = None
            if self.ai_enabled:
                try:
                    ai_explanation = await self.ai_layer.prediction_explainer.explain_prediction(
                        ml_prediction, match_data
                    )
                    logger.info("✅ Explication IA générée")
                except Exception as e:
                    logger.warning(f"⚠️ Explication IA indisponible: {e}")
            
            # ===== ÉTAPE 4: ANALYSE VALUE BETTING IA =====
            value_analysis = None
            if self.ai_enabled and market_odds:
                try:
                    value_analysis = await self.ai_layer.value_detector.detect_value_opportunities(
                        [ml_prediction], market_odds
                    )
                    logger.info(f"💰 Value betting: {len(value_analysis.get('value_opportunities', []))} opportunités")
                except Exception as e:
                    logger.warning(f"⚠️ Value betting IA indisponible: {e}")
            
            # ===== RÉSULTAT HYBRIDE FINAL =====
            processing_time = (datetime.now() - prediction_start).total_seconds()
            
            result = {
                # Prédiction ML (cœur du système)
                "ml_prediction": ml_prediction,
                
                # Enrichissements IA
                "ai_enhanced_features": ai_features,
                "ai_explanation": ai_explanation,
                "value_opportunities": value_analysis,
                
                # Méta-informations
                "confidence_boost": confidence_boost,
                "ai_enabled": self.ai_enabled,
                "processing_time": processing_time,
                "prediction_timestamp": datetime.now().isoformat(),
                
                # Qualité de la prédiction
                "prediction_quality": self._assess_prediction_quality(ml_prediction, ai_explanation)
            }
            
            logger.info(f"🎉 Prédiction hybride terminée en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur prédiction hybride: {e}")
            
            # Fallback: prédiction ML pure si IA échoue
            if self.ai_fallback:
                ml_prediction = await self._generate_ml_prediction(match_data, team_stats)
                return {
                    "ml_prediction": ml_prediction,
                    "ai_enhanced_features": {},
                    "ai_explanation": None,
                    "value_opportunities": None,
                    "confidence_boost": 0.0,
                    "ai_enabled": False,
                    "error": str(e),
                    "fallback_mode": True
                }
            else:
                raise
    
    async def _generate_ml_prediction(self, match_data: Dict, team_stats: Dict) -> Dict[str, Any]:
        """
        Générer une prédiction avec nos modèles ML
        (Simulation - remplacer par vraie logique des 180 modèles)
        """
        # TODO: Remplacer par l'appel à revolutionary_model_architecture.py
        # qui contient nos 180 modèles spécialisés
        
        import random
        prediction_types = ["match_result", "total_goals", "both_teams_scored"]
        prediction_values = {
            "match_result": ["1", "X", "2"],
            "total_goals": ["Under 2.5", "Over 2.5"], 
            "both_teams_scored": ["Yes", "No"]
        }
        
        pred_type = random.choice(prediction_types)
        pred_value = random.choice(prediction_values[pred_type])
        
        # Simuler nos modèles ML
        base_confidence = random.uniform(65, 85)
        
        return {
            "prediction_type": pred_type,
            "prediction_value": pred_value,
            "confidence": base_confidence,
            "odds": random.uniform(1.5, 4.0),
            "expected_value": random.uniform(0.05, 0.25),
            "model_used": "ensemble_revolutionary_180_models",
            "league": match_data.get("league", "unknown"),
            "match_id": match_data.get("match_id", "unknown")
        }
    
    def _assess_prediction_quality(self, ml_prediction: Dict, ai_explanation: Optional[AIPredictionExplanation]) -> str:
        """Évaluer la qualité globale de la prédiction"""
        confidence = ml_prediction.get("confidence", 0)
        has_ai_explanation = ai_explanation is not None
        
        if confidence >= 80 and has_ai_explanation:
            return "excellent"
        elif confidence >= 70 and has_ai_explanation:
            return "very_good"
        elif confidence >= 70:
            return "good"
        elif confidence >= 60:
            return "moderate"
        else:
            return "low_confidence"

class HybridIntelligentCoupon:
    """Générateur de coupons intelligents avec IA"""
    
    def __init__(self, openai_api_key: str):
        self.hybrid_predictor = HybridMLAIPredictor(openai_api_key)
        self.portfolio_optimizer = AdvancedPortfolioOptimizer()
    
    async def generate_ai_enhanced_coupon(self, matches_data: List[Dict], 
                                        user_config: Dict, market_odds_data: List[Dict] = None) -> Dict[str, Any]:
        """
        Générer un coupon intelligent enrichi par l'IA
        
        Args:
            matches_data: Liste des matchs disponibles
            user_config: Configuration utilisateur (risk_profile, budget, etc.)
            market_odds_data: Cotes du marché pour chaque match
            
        Returns:
            Coupon optimisé ML + IA
        """
        coupon_start = datetime.now()
        logger.info(f"🎯 Génération coupon IA pour {len(matches_data)} matchs")
        
        ai_enhanced_predictions = []
        
        # ===== PRÉDICTIONS HYBRIDES POUR CHAQUE MATCH =====
        for i, match_data in enumerate(matches_data):
            try:
                # Obtenir cotes marché pour ce match si disponibles
                match_odds = None
                if market_odds_data and i < len(market_odds_data):
                    match_odds = [market_odds_data[i]]
                
                # Simuler team stats (remplacer par vraies données)
                team_stats = self._get_team_stats(match_data)
                
                # Prédiction hybride ML + IA
                prediction = await self.hybrid_predictor.predict_with_ai_enhancement(
                    match_data, team_stats, match_odds
                )
                
                ai_enhanced_predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Erreur prédiction match {i}: {e}")
                continue
        
        # ===== SÉLECTION ET OPTIMISATION DES PRÉDICTIONS =====
        # Filtrer par confiance minimum
        min_confidence = user_config.get("min_confidence", 70)
        qualified_predictions = [
            p for p in ai_enhanced_predictions 
            if p["ml_prediction"]["confidence"] >= min_confidence
        ]
        
        logger.info(f"📊 {len(qualified_predictions)}/{len(ai_enhanced_predictions)} prédictions qualifiées")
        
        # Sélectionner les meilleures selon configuration utilisateur
        selected_predictions = self._select_best_predictions(
            qualified_predictions, user_config
        )
        
        # ===== OPTIMISATION PORTFOLIO =====
        # Convertir pour l'optimiseur de portfolio
        betting_predictions = self._convert_to_betting_predictions(selected_predictions)
        
        if betting_predictions:
            portfolio = self.portfolio_optimizer.optimize_portfolio_markowitz(
                betting_predictions, objective="max_sharpe"
            )
        else:
            portfolio = {"allocation": [], "expected_return": 0, "risk": 0}
        
        # ===== COMPILATION DES EXPLICATIONS IA =====
        ai_explanations = []
        for pred in selected_predictions:
            if pred.get("ai_explanation"):
                ai_explanations.append({
                    "match": f"{pred['ml_prediction'].get('match_id', 'N/A')}",
                    "explanation": pred["ai_explanation"]
                })
        
        # ===== ANALYSE VALUE BETTING GLOBALE =====
        total_value_opportunities = []
        for pred in selected_predictions:
            if pred.get("value_opportunities") and pred["value_opportunities"].get("value_opportunities"):
                total_value_opportunities.extend(pred["value_opportunities"]["value_opportunities"])
        
        # ===== RÉSULTAT FINAL =====
        processing_time = (datetime.now() - coupon_start).total_seconds()
        
        coupon_result = {
            "coupon_id": f"ai_coupon_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "user_config": user_config,
            
            # Prédictions sélectionnées
            "selected_predictions": selected_predictions,
            "total_predictions": len(selected_predictions),
            
            # Optimisation portfolio
            "portfolio_allocation": portfolio["allocation"] if "allocation" in portfolio else [],
            "expected_return": portfolio.get("expected_return", 0),
            "portfolio_risk": portfolio.get("risk", 0),
            
            # Enrichissements IA
            "ai_explanations": ai_explanations,
            "value_opportunities": total_value_opportunities,
            "ai_features_used": sum(1 for p in selected_predictions if p.get("ai_enhanced_features")),
            
            # Méta-informations
            "average_confidence": sum(p["ml_prediction"]["confidence"] for p in selected_predictions) / len(selected_predictions) if selected_predictions else 0,
            "confidence_boost_total": sum(p.get("confidence_boost", 0) for p in selected_predictions),
            "processing_time": processing_time,
            "generation_timestamp": datetime.now().isoformat(),
            
            # Qualité du coupon
            "coupon_quality": self._assess_coupon_quality(selected_predictions, ai_explanations)
        }
        
        logger.info(f"🎉 Coupon IA généré: {len(selected_predictions)} prédictions, qualité {coupon_result['coupon_quality']}")
        return coupon_result
    
    def _get_team_stats(self, match_data: Dict) -> Dict[str, float]:
        """Simuler les statistiques d'équipe (remplacer par vraies données)"""
        import random
        return {
            "home_goals_avg": random.uniform(1.2, 2.8),
            "away_goals_avg": random.uniform(1.0, 2.5),
            "home_possession": random.uniform(0.45, 0.65),
            "away_possession": random.uniform(0.35, 0.55),
            "home_shots_avg": random.uniform(12, 20),
            "away_shots_avg": random.uniform(10, 18)
        }
    
    def _select_best_predictions(self, predictions: List[Dict], config: Dict) -> List[Dict]:
        """Sélectionner les meilleures prédictions selon la configuration"""
        max_predictions = config.get("max_predictions", 8)
        risk_profile = config.get("risk_profile", "balanced")
        
        # Trier par confiance + value (si disponible)
        def prediction_score(p):
            base_score = p["ml_prediction"]["confidence"]
            
            # Bonus si value betting détecté
            if p.get("value_opportunities") and p["value_opportunities"].get("value_opportunities"):
                base_score += 5
            
            # Bonus si explications IA disponibles
            if p.get("ai_explanation"):
                base_score += 3
            
            # Ajustement selon profil de risque
            if risk_profile == "conservative":
                # Privilégier haute confiance, faible variance
                if p["ml_prediction"]["confidence"] > 80:
                    base_score += 10
            elif risk_profile == "aggressive":
                # Privilégier value, même avec confiance modérée
                if p.get("value_opportunities"):
                    base_score += 15
            
            return base_score
        
        # Trier et sélectionner
        sorted_predictions = sorted(predictions, key=prediction_score, reverse=True)
        return sorted_predictions[:max_predictions]
    
    def _convert_to_betting_predictions(self, ai_predictions: List[Dict]) -> List[Any]:
        """Convertir les prédictions IA en format pour portfolio optimizer"""
        # Simulation - remplacer par vraie conversion
        return []  # TODO: Implémenter conversion
    
    def _assess_coupon_quality(self, predictions: List[Dict], ai_explanations: List[Dict]) -> str:
        """Évaluer la qualité globale du coupon"""
        if not predictions:
            return "empty"
        
        avg_confidence = sum(p["ml_prediction"]["confidence"] for p in predictions) / len(predictions)
        ai_coverage = len(ai_explanations) / len(predictions)
        
        if avg_confidence >= 80 and ai_coverage >= 0.8:
            return "premium"
        elif avg_confidence >= 75 and ai_coverage >= 0.6:
            return "excellent"
        elif avg_confidence >= 70 and ai_coverage >= 0.4:
            return "very_good"
        elif avg_confidence >= 65:
            return "good"
        else:
            return "standard"

# Fonction de test
async def test_hybrid_system():
    """Tester le système hybride ML-IA"""
    print("🤖🧠 Test Système Hybride ML-IA")
    
    # ⚠️ Remplacer par votre vraie clé API OpenAI
    API_KEY = "your-openai-api-key"
    
    if API_KEY == "your-openai-api-key":
        print("❌ Veuillez définir votre clé API OpenAI")
        return
    
    # Initialiser le système hybride
    coupon_generator = HybridIntelligentCoupon(API_KEY)
    
    # Données de test
    matches_data = [
        {"match_id": 1, "home_team": "Manchester United", "away_team": "Liverpool", "league": "Premier League"},
        {"match_id": 2, "home_team": "Barcelona", "away_team": "Real Madrid", "league": "La Liga"},
        {"match_id": 3, "home_team": "Bayern Munich", "away_team": "Dortmund", "league": "Bundesliga"}
    ]
    
    user_config = {
        "risk_profile": "balanced",
        "min_confidence": 70,
        "max_predictions": 5,
        "budget": 500
    }
    
    # Test génération coupon hybride
    coupon = await coupon_generator.generate_ai_enhanced_coupon(matches_data, user_config)
    
    print(f"✅ Coupon hybride généré:")
    print(f"- Prédictions: {coupon['total_predictions']}")
    print(f"- Confiance moyenne: {coupon['average_confidence']:.1f}%")
    print(f"- Features IA utilisées: {coupon['ai_features_used']}")
    print(f"- Explications IA: {len(coupon['ai_explanations'])}")
    print(f"- Qualité: {coupon['coupon_quality']}")

if __name__ == "__main__":
    # Pour tester, décommenter et définir votre clé API
    # asyncio.run(test_hybrid_system())
    print("🤖🧠 Système Hybride ML-IA prêt - Définir la clé API pour utiliser")