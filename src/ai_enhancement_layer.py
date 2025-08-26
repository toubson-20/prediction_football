"""
🤖 AI ENHANCEMENT LAYER - INTEGRATION GPT
Couche d'intelligence artificielle pour enrichir le système ML de prédictions football
"""

import openai
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio
import numpy as np
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AIEnhancedFeatures:
    """Features enrichies par l'IA"""
    original_features: Dict[str, float]
    ai_features: Dict[str, float]
    ai_insights: Dict[str, str]
    confidence_boost: float
    processing_time: float

@dataclass
class AIPredictionExplanation:
    """Explication IA d'une prédiction ML"""
    prediction_summary: str
    key_factors: List[str]
    risk_analysis: str
    market_context: str
    recommendation: str
    confidence_reasoning: str

class GPTPromptTemplates:
    """Templates de prompts optimisés pour chaque tâche IA"""
    
    @staticmethod
    def feature_enhancement_prompt(match_data: Dict, team_stats: Dict) -> str:
        return f"""Tu es un expert en analyse football avec 20 ans d'expérience. Analyse ces données et génère des insights avancés.

DONNÉES MATCH:
{json.dumps(match_data, indent=2)}

STATISTIQUES ÉQUIPES:
{json.dumps(team_stats, indent=2)}

MISSION: Génère exactement 10 features avancées (valeurs 0-1) que des modèles ML classiques pourraient manquer.

CONSIGNES STRICTES:
1. Chaque feature doit avoir une valeur numérique entre 0 et 1
2. Base-toi sur des patterns footballistiques réels et subtils
3. Considère: momentum psychologique, compatibilité tactique, conditions contextuelles
4. Évite les features évidentes déjà calculées par les stats classiques
5. Sois précis et justifie chaque valeur

FORMAT DE RÉPONSE OBLIGATOIRE (JSON):
{{
    "ai_momentum_psychological": 0.73,
    "ai_tactical_compatibility": 0.82,
    "ai_weather_adaptation": 0.91,
    "ai_motivation_contextual": 0.67,
    "ai_fatigue_management": 0.55,
    "ai_crowd_impact": 0.88,
    "ai_referee_influence": 0.72,
    "ai_injury_cascade": 0.43,
    "ai_formation_mismatch": 0.79,
    "ai_confidence_momentum": 0.84
}}

Chaque feature doit être accompagnée d'une justification courte mais précise."""

    @staticmethod
    def prediction_explanation_prompt(ml_prediction: Dict, match_context: Dict) -> str:
        return f"""Tu es un commentateur football expert qui explique des prédictions ML à des parieurs intelligents.

PRÉDICTION ML:
- Type: {ml_prediction.get('prediction_type', 'N/A')}
- Valeur: {ml_prediction.get('prediction_value', 'N/A')}
- Confiance: {ml_prediction.get('confidence', 0)}%
- Cotes: {ml_prediction.get('odds', 'N/A')}
- Modèle utilisé: {ml_prediction.get('model_used', 'N/A')}

CONTEXTE MATCH:
{json.dumps(match_context, indent=2)}

MISSION: Explique cette prédiction ML de manière claire et actionnable.

CONSIGNES STRICTES:
1. Explique POURQUOI nos modèles ML ont cette confiance
2. Identifie 3-4 facteurs clés qui influencent la prédiction
3. Analyse les risques potentiels qui pourraient faire échouer la prédiction
4. Compare avec le marché (si les cotes semblent sur/sous-évaluées)
5. Donne une recommandation finale claire (Parier/Éviter/Prudence)
6. Utilise un langage professionnel mais accessible
7. Maximum 300 mots, structure claire

FORMAT DE RÉPONSE:
## 🎯 PRÉDICTION: [Résumé en 1 ligne]

## 📊 FACTEURS CLÉS:
- [Facteur 1 avec explication courte]
- [Facteur 2 avec explication courte]
- [Facteur 3 avec explication courte]

## ⚠️ RISQUES À SURVEILLER:
[2-3 éléments qui pourraient faire échouer la prédiction]

## 💰 ANALYSE MARCHÉ:
[Comparaison avec les cotes proposées]

## 🎲 RECOMMANDATION:
[PARIER/ÉVITER/PRUDENCE] - [Justification en 1-2 phrases]"""

    @staticmethod
    def value_betting_prompt(our_predictions: List[Dict], market_odds: List[Dict]) -> str:
        return f"""Tu es un mathématicien expert en probabilités et value betting.

NOS PRÉDICTIONS ML:
{json.dumps(our_predictions, indent=2)}

COTES DU MARCHÉ:
{json.dumps(market_odds, indent=2)}

MISSION: Identifie les opportunités de value betting en comparant nos prédictions avec le marché.

CONSIGNES MATHÉMATIQUES:
1. Calcule la valeur attendue pour chaque pari: (Proba_ML * Cote) - 1
2. Identifie les paris avec valeur attendue > 5%
3. Analyse si nos modèles sont historiquement fiables sur ce type de prédiction
4. Considère la variance et le risque de chaque opportunité
5. Classe par ordre de priorité (meilleure valeur + faible risque)

FORMAT DE RÉPONSE (JSON):
{{
    "value_opportunities": [
        {{
            "prediction_type": "match_result",
            "our_probability": 0.78,
            "market_probability": 0.65,
            "expected_value": 0.12,
            "risk_level": "medium",
            "recommendation": "Strong bet",
            "reasoning": "Nos modèles surévaluent cette équipe de 13 points. Historique solide sur ce type de match."
        }}
    ],
    "market_efficiency": 0.73,
    "best_opportunity": "match_result",
    "total_value_found": 0.08
}}"""

    @staticmethod
    def system_optimization_prompt(performance_data: Dict, error_patterns: List[str]) -> str:
        return f"""Tu es un data scientist senior expert en optimisation de systèmes ML de prédiction sportive.

DONNÉES PERFORMANCE:
{json.dumps(performance_data, indent=2)}

PATTERNS D'ERREURS IDENTIFIÉS:
{json.dumps(error_patterns, indent=2)}

MISSION: Analyse ces données et suggère des optimisations concrètes pour améliorer le système.

CONSIGNES TECHNIQUES:
1. Identifie les faiblesses dans nos prédictions (types de matchs, ligues, situations)
2. Suggère des améliorations d'architecture ML spécifiques
3. Propose des nouvelles features qui pourraient corriger les erreurs
4. Recommande des ajustements de seuils ou paramètres
5. Prioritise les optimisations par impact potentiel vs effort

FORMAT DE RÉPONSE (JSON):
{{
    "critical_issues": ["Issue 1", "Issue 2"],
    "optimization_suggestions": [
        {{
            "category": "feature_engineering",
            "suggestion": "Ajouter features de pression temporelle",
            "impact": "high",
            "effort": "medium",
            "implementation": "Calculer stress index basé sur calendrier"
        }}
    ],
    "architecture_improvements": ["Amélioration 1", "Amélioration 2"],
    "parameter_adjustments": {{
        "confidence_threshold": 0.75,
        "ensemble_weights": [0.4, 0.35, 0.25]
    }},
    "priority_ranking": ["optimization_1", "optimization_2"]
}}"""

class AIFeatureEnhancer:
    """Enrichisseur de features utilisant GPT"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialiser l'enhancer IA
        
        Args:
            api_key: Clé API OpenAI
            model: Modèle GPT à utiliser (gpt-4o-mini recommandé pour coût/performance)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.cache = {}  # Cache pour éviter les appels répétitifs
    
    async def enhance_features(self, match_data: Dict, team_stats: Dict) -> AIEnhancedFeatures:
        """
        Enrichir les features avec l'intelligence GPT
        
        Args:
            match_data: Données du match
            team_stats: Statistiques des équipes
            
        Returns:
            Features enrichies avec insights IA
        """
        start_time = datetime.now()
        
        try:
            # Vérifier le cache
            cache_key = f"{match_data.get('match_id', 'unknown')}_{hash(str(team_stats))}"
            if cache_key in self.cache:
                logger.info("Features IA récupérées depuis le cache")
                return self.cache[cache_key]
            
            # Générer le prompt
            prompt = GPTPromptTemplates.feature_enhancement_prompt(match_data, team_stats)
            
            # Appel GPT
            response = await self._make_gpt_call(prompt)
            
            # Parser la réponse JSON
            try:
                ai_features = json.loads(response)
                # Valider que toutes les valeurs sont entre 0 et 1
                ai_features = {k: max(0, min(1, v)) for k, v in ai_features.items()}
            except json.JSONDecodeError:
                logger.error("Erreur parsing JSON GPT, utilisation features par défaut")
                ai_features = self._get_default_ai_features()
            
            # Calculer boost de confiance basé sur la cohérence des features IA
            confidence_boost = self._calculate_confidence_boost(ai_features)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = AIEnhancedFeatures(
                original_features=team_stats,
                ai_features=ai_features,
                ai_insights={"generation_method": "gpt", "model": self.model},
                confidence_boost=confidence_boost,
                processing_time=processing_time
            )
            
            # Mise en cache
            self.cache[cache_key] = result
            
            logger.info(f"Features IA générées en {processing_time:.2f}s, boost: +{confidence_boost:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Erreur génération features IA: {e}")
            # Fallback avec features par défaut
            return self._get_fallback_features(team_stats, start_time)
    
    def _calculate_confidence_boost(self, ai_features: Dict[str, float]) -> float:
        """Calculer le boost de confiance basé sur les features IA"""
        # Plus les features IA sont cohérentes et diversifiées, plus le boost
        feature_values = list(ai_features.values())
        
        # Diversité des features (éviter toutes les valeurs similaires)
        diversity = np.std(feature_values)
        
        # Intensité moyenne (features proches de 0.5 = incertaines)
        intensity = np.mean([abs(v - 0.5) * 2 for v in feature_values])
        
        # Boost entre 0 et 10%
        boost = min(10, (diversity * 10) + (intensity * 5))
        return round(boost, 2)
    
    def _get_default_ai_features(self) -> Dict[str, float]:
        """Features IA par défaut en cas d'erreur"""
        return {
            "ai_momentum_psychological": 0.5,
            "ai_tactical_compatibility": 0.5,
            "ai_weather_adaptation": 0.5,
            "ai_motivation_contextual": 0.5,
            "ai_fatigue_management": 0.5,
            "ai_crowd_impact": 0.5,
            "ai_referee_influence": 0.5,
            "ai_injury_cascade": 0.5,
            "ai_formation_mismatch": 0.5,
            "ai_confidence_momentum": 0.5
        }
    
    def _get_fallback_features(self, team_stats: Dict, start_time: datetime) -> AIEnhancedFeatures:
        """Features de fallback en cas d'échec total"""
        return AIEnhancedFeatures(
            original_features=team_stats,
            ai_features=self._get_default_ai_features(),
            ai_insights={"error": "fallback_mode", "model": "none"},
            confidence_boost=0.0,
            processing_time=(datetime.now() - start_time).total_seconds()
        )
    
    async def _make_gpt_call(self, prompt: str) -> str:
        """Effectuer un appel GPT avec gestion d'erreurs"""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "Tu es un expert en analyse football et machine learning. Réponds toujours en JSON valide."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Peu créatif, plus factuel
                max_tokens=800,
                timeout=30
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Erreur appel GPT: {e}")
            raise

class AIPredictionExplainer:
    """Explicateur de prédictions utilisant GPT"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    async def explain_prediction(self, ml_prediction: Dict, match_context: Dict) -> AIPredictionExplanation:
        """
        Expliquer une prédiction ML avec l'intelligence GPT
        
        Args:
            ml_prediction: Résultat de nos modèles ML
            match_context: Contexte du match
            
        Returns:
            Explication détaillée et actionnable
        """
        try:
            prompt = GPTPromptTemplates.prediction_explanation_prompt(ml_prediction, match_context)
            
            response = await self._make_gpt_call(prompt)
            
            # Parser la réponse structurée
            explanation = self._parse_explanation(response)
            
            logger.info(f"Explication générée pour prédiction {ml_prediction.get('prediction_type', 'unknown')}")
            return explanation
            
        except Exception as e:
            logger.error(f"Erreur génération explication: {e}")
            return self._get_fallback_explanation(ml_prediction)
    
    def _parse_explanation(self, gpt_response: str) -> AIPredictionExplanation:
        """Parser la réponse GPT en structure organisée"""
        # Extraction des sections avec regex ou parsing simple
        sections = gpt_response.split("##")
        
        prediction_summary = ""
        key_factors = []
        risk_analysis = ""
        market_context = ""
        recommendation = ""
        
        for section in sections:
            if "PRÉDICTION:" in section:
                prediction_summary = section.split("PRÉDICTION:")[1].strip()[:200]
            elif "FACTEURS CLÉS:" in section:
                factors_text = section.split("FACTEURS CLÉS:")[1].strip()
                key_factors = [f.strip("- ").strip() for f in factors_text.split("\n") if f.strip().startswith("-")][:4]
            elif "RISQUES À SURVEILLER:" in section:
                risk_analysis = section.split("RISQUES À SURVEILLER:")[1].strip()[:200]
            elif "ANALYSE MARCHÉ:" in section:
                market_context = section.split("ANALYSE MARCHÉ:")[1].strip()[:200]
            elif "RECOMMANDATION:" in section:
                recommendation = section.split("RECOMMANDATION:")[1].strip()[:200]
        
        return AIPredictionExplanation(
            prediction_summary=prediction_summary or "Prédiction générée par nos modèles ML",
            key_factors=key_factors or ["Forme récente", "Statistiques historiques", "Contexte match"],
            risk_analysis=risk_analysis or "Risques standards liés aux prédictions sportives",
            market_context=market_context or "Analyse du marché en cours",
            recommendation=recommendation or "Suivre la prédiction avec prudence standard",
            confidence_reasoning="Basé sur analyse GPT des facteurs contextuels"
        )
    
    def _get_fallback_explanation(self, ml_prediction: Dict) -> AIPredictionExplanation:
        """Explication de fallback en cas d'erreur"""
        return AIPredictionExplanation(
            prediction_summary=f"Prédiction {ml_prediction.get('prediction_value', 'N/A')} avec {ml_prediction.get('confidence', 0)}% de confiance",
            key_factors=["Modèles ML spécialisés", "Données historiques", "Statistiques actuelles"],
            risk_analysis="Risques standards des prédictions sportives",
            market_context="Analyse de marché non disponible",
            recommendation="Suivre la prédiction selon votre stratégie habituelle",
            confidence_reasoning="Basé uniquement sur les modèles ML (IA indisponible)"
        )
    
    async def _make_gpt_call(self, prompt: str) -> str:
        """Effectuer un appel GPT pour les explications"""
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=[
                {"role": "system", "content": "Tu es un expert commentateur football qui explique des prédictions ML. Sois précis, structuré et actionnable."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,  # Légèrement plus créatif pour les explications
            max_tokens=600,
            timeout=30
        )
        
        return response.choices[0].message.content.strip()

class AIValueBettingDetector:
    """Détecteur d'opportunités value betting utilisant GPT"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    async def detect_value_opportunities(self, our_predictions: List[Dict], market_odds: List[Dict]) -> Dict[str, Any]:
        """
        Détecter les opportunités de value betting
        
        Args:
            our_predictions: Nos prédictions ML
            market_odds: Cotes du marché
            
        Returns:
            Analyse des opportunités de valeur
        """
        try:
            prompt = GPTPromptTemplates.value_betting_prompt(our_predictions, market_odds)
            
            response = await self._make_gpt_call(prompt)
            
            # Parser la réponse JSON
            try:
                value_analysis = json.loads(response)
                logger.info(f"Analyse value betting: {len(value_analysis.get('value_opportunities', []))} opportunités trouvées")
                return value_analysis
            except json.JSONDecodeError:
                logger.error("Erreur parsing JSON pour value betting")
                return self._get_fallback_value_analysis()
                
        except Exception as e:
            logger.error(f"Erreur détection value betting: {e}")
            return self._get_fallback_value_analysis()
    
    def _get_fallback_value_analysis(self) -> Dict[str, Any]:
        """Analyse de fallback"""
        return {
            "value_opportunities": [],
            "market_efficiency": 0.7,
            "best_opportunity": None,
            "total_value_found": 0.0,
            "error": "AI analysis unavailable"
        }
    
    async def _make_gpt_call(self, prompt: str) -> str:
        """Appel GPT pour analyse value betting"""
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=[
                {"role": "system", "content": "Tu es un mathématicien expert en probabilités et value betting. Réponds en JSON valide uniquement."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Très peu créatif, maximum précision mathématique
            max_tokens=700,
            timeout=30
        )
        
        return response.choices[0].message.content.strip()

class AISystemOptimizer:
    """Optimiseur de système utilisant GPT"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    async def suggest_optimizations(self, performance_data: Dict, error_patterns: List[str]) -> Dict[str, Any]:
        """
        Suggérer des optimisations système basées sur les performances
        
        Args:
            performance_data: Données de performance du système
            error_patterns: Patterns d'erreurs identifiés
            
        Returns:
            Suggestions d'optimisation prioritisées
        """
        try:
            prompt = GPTPromptTemplates.system_optimization_prompt(performance_data, error_patterns)
            
            response = await self._make_gpt_call(prompt)
            
            try:
                optimization_suggestions = json.loads(response)
                logger.info(f"Suggestions d'optimisation générées: {len(optimization_suggestions.get('optimization_suggestions', []))} suggestions")
                return optimization_suggestions
            except json.JSONDecodeError:
                logger.error("Erreur parsing suggestions d'optimisation")
                return self._get_fallback_optimizations()
                
        except Exception as e:
            logger.error(f"Erreur génération optimisations: {e}")
            return self._get_fallback_optimizations()
    
    def _get_fallback_optimizations(self) -> Dict[str, Any]:
        """Optimisations de fallback"""
        return {
            "critical_issues": ["AI analysis unavailable"],
            "optimization_suggestions": [],
            "architecture_improvements": [],
            "parameter_adjustments": {},
            "priority_ranking": []
        }
    
    async def _make_gpt_call(self, prompt: str) -> str:
        """Appel GPT pour optimisations"""
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=[
                {"role": "system", "content": "Tu es un data scientist senior expert en optimisation ML. Réponds en JSON valide avec des suggestions concrètes et implémentables."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
            timeout=40
        )
        
        return response.choices[0].message.content.strip()

class AIEnhancementLayer:
    """Couche principale d'enhancement IA"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialiser la couche d'enhancement IA
        
        Args:
            openai_api_key: Clé API OpenAI  
            model: Modèle GPT à utiliser (gpt-4o-mini recommandé pour équilibre coût/performance)
        """
        self.feature_enhancer = AIFeatureEnhancer(openai_api_key, model)
        self.prediction_explainer = AIPredictionExplainer(openai_api_key, model)
        self.value_detector = AIValueBettingDetector(openai_api_key, model)
        self.system_optimizer = AISystemOptimizer(openai_api_key, model)
        
        self.usage_stats = {
            "total_calls": 0,
            "feature_enhancements": 0,
            "explanations_generated": 0,
            "value_analyses": 0,
            "optimizations": 0
        }
        
        logger.info(f"🤖 AI Enhancement Layer initialisé avec {model}")
    
    async def enhance_prediction_pipeline(self, match_data: Dict, team_stats: Dict, ml_prediction: Dict, market_odds: List[Dict] = None) -> Dict[str, Any]:
        """
        Pipeline complet d'enhancement IA
        
        Args:
            match_data: Données du match
            team_stats: Statistiques des équipes  
            ml_prediction: Prédiction de nos modèles ML
            market_odds: Cotes du marché (optionnel)
            
        Returns:
            Résultat enrichi par l'IA
        """
        try:
            logger.info("🔄 Démarrage pipeline d'enhancement IA")
            
            # 1. Enhancement des features (pour futurs entraînements)
            enhanced_features = await self.feature_enhancer.enhance_features(match_data, team_stats)
            self.usage_stats["feature_enhancements"] += 1
            
            # 2. Explication de la prédiction ML
            explanation = await self.prediction_explainer.explain_prediction(ml_prediction, match_data)
            self.usage_stats["explanations_generated"] += 1
            
            # 3. Analyse value betting (si cotes disponibles)
            value_analysis = None
            if market_odds:
                value_analysis = await self.value_detector.detect_value_opportunities([ml_prediction], market_odds)
                self.usage_stats["value_analyses"] += 1
            
            self.usage_stats["total_calls"] += 1
            
            return {
                "ml_prediction": ml_prediction,  # ← Prédiction originale de nos modèles ML
                "ai_enhanced_features": enhanced_features.ai_features,
                "ai_explanation": explanation,
                "value_opportunities": value_analysis,
                "confidence_boost": enhanced_features.confidence_boost,
                "ai_processing_time": enhanced_features.processing_time,
                "enhancement_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur pipeline AI enhancement: {e}")
            # Retourner la prédiction ML sans enhancement en cas d'erreur
            return {
                "ml_prediction": ml_prediction,
                "ai_enhanced_features": {},
                "ai_explanation": None,
                "value_opportunities": None,
                "confidence_boost": 0.0,
                "error": str(e)
            }
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Obtenir les statistiques d'utilisation de l'IA"""
        return {
            **self.usage_stats,
            "average_cost_per_call": 0.002,  # Estimation coût GPT-4o-mini
            "estimated_monthly_cost": self.usage_stats["total_calls"] * 0.002 * 30
        }

# Fonction de test
async def test_ai_enhancement_layer():
    """Tester la couche d'enhancement IA"""
    print("🤖 Test AI Enhancement Layer")
    
    # Vous devez définir votre clé API OpenAI
    API_KEY = "your-openai-api-key"  # ⚠️ Remplacer par votre vraie clé API
    
    if API_KEY == "your-openai-api-key":
        print("❌ Veuillez définir votre clé API OpenAI")
        return
    
    ai_layer = AIEnhancementLayer(API_KEY)
    
    # Données de test
    match_data = {
        "match_id": 12345,
        "home_team": "Manchester United",
        "away_team": "Liverpool",
        "league": "Premier League",
        "date": "2025-08-24"
    }
    
    team_stats = {
        "home_goals_avg": 2.1,
        "away_goals_avg": 1.8,
        "home_possession": 0.58,
        "away_possession": 0.62
    }
    
    ml_prediction = {
        "prediction_type": "match_result",
        "prediction_value": "1",  # Home win
        "confidence": 76.5,
        "odds": 2.10,
        "model_used": "ensemble_xgb_nn"
    }
    
    # Test du pipeline complet
    result = await ai_layer.enhance_prediction_pipeline(
        match_data, team_stats, ml_prediction
    )
    
    print(f"✅ Enhancement terminé:")
    print(f"- Features IA générées: {len(result['ai_enhanced_features'])}")
    print(f"- Boost confiance: +{result['confidence_boost']}%")
    print(f"- Explication disponible: {'Oui' if result['ai_explanation'] else 'Non'}")
    
    # Statistiques d'usage
    stats = ai_layer.get_usage_statistics()
    print(f"\n📊 Statistiques: {stats['total_calls']} appels IA")

if __name__ == "__main__":
    # Pour tester, décommenter et définir votre clé API
    # asyncio.run(test_ai_enhancement_layer())
    print("🤖 AI Enhancement Layer prêt - Définir la clé API pour utiliser")