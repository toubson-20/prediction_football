"""
🎯 OPTIMISEUR INTELLIGENT DE COUPONS
Système d'IA pour créer des coupons optimaux en sélectionnant stratégiquement les meilleurs matchs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import itertools
from dataclasses import dataclass
import math

@dataclass
class MatchOpportunity:
    """Opportunité de match avec métriques d'évaluation"""
    match_id: int
    home_team: str
    away_team: str
    league: str
    prediction_type: str
    prediction_value: str
    confidence: float
    odds: float
    expected_value: float
    risk_score: float
    league_reliability: float
    recent_form_home: float
    recent_form_away: float
    historical_accuracy: float
    market_efficiency: float
    correlation_factor: float
    
    @property
    def kelly_criterion(self) -> float:
        """Calculer le critère de Kelly pour optimiser la mise"""
        prob = self.confidence / 100
        b = self.odds - 1
        if prob * b > 1:
            return (prob * b - 1) / b
        return 0
    
    @property
    def sharpe_ratio(self) -> float:
        """Ratio de Sharpe pour mesurer le rendement ajusté au risque"""
        if self.risk_score == 0:
            return float('inf')
        return self.expected_value / (self.risk_score / 100)
    
    @property
    def optimization_score(self) -> float:
        """Score global d'optimisation combinant plusieurs métriques"""
        # Pondération des différents facteurs
        confidence_weight = 0.25
        ev_weight = 0.20
        kelly_weight = 0.15
        sharpe_weight = 0.15
        reliability_weight = 0.10
        accuracy_weight = 0.10
        efficiency_weight = 0.05
        
        score = (
            (self.confidence / 100) * confidence_weight +
            max(0, self.expected_value) * ev_weight +
            self.kelly_criterion * kelly_weight +
            min(10, self.sharpe_ratio) / 10 * sharpe_weight +
            self.league_reliability * reliability_weight +
            self.historical_accuracy * accuracy_weight +
            (1 - self.market_efficiency) * efficiency_weight
        )
        
        return score

class IntelligentCouponOptimizer:
    """Optimiseur intelligent pour création de coupons stratégiques"""
    
    def __init__(self):
        # Paramètres d'optimisation
        self.league_reliability = {
            'premier_league': 0.92,
            'la_liga': 0.89,
            'bundesliga': 0.87,
            'serie_a': 0.85,
            'ligue_1': 0.83,
            'champions_league': 0.95,
            'europa_league': 0.88
        }
        
        # Corrélations entre types de prédictions (pour éviter redondance)
        self.prediction_correlations = {
            ('match_result', 'win_probability'): 0.85,
            ('both_teams_score', 'over_2_5_goals'): 0.70,
            ('clean_sheet', 'under_2_5_goals'): 0.60,
            ('home_goals', 'match_result'): 0.55
        }
        
    def generate_match_opportunities(self, available_matches: List[Dict], 
                                   filters: Dict) -> List[MatchOpportunity]:
        """Générer opportunités enrichies à partir des matchs disponibles"""
        opportunities = []
        
        for match in available_matches:
            # Enrichir avec métriques avancées
            opportunity = MatchOpportunity(
                match_id=match['match_id'],
                home_team=match['home_team'],
                away_team=match['away_team'],
                league=match['league'],
                prediction_type=match['prediction_type'],
                prediction_value=match.get('prediction_value', 'N/A'),
                confidence=match['confidence'],
                odds=match['odds'],
                expected_value=match['expected_value'],
                risk_score=self._calculate_risk_score(match),
                league_reliability=self.league_reliability.get(match['league'], 0.80),
                recent_form_home=self._simulate_form_metric(),
                recent_form_away=self._simulate_form_metric(),
                historical_accuracy=self._simulate_accuracy_metric(),
                market_efficiency=self._simulate_market_efficiency(),
                correlation_factor=0.0  # Calculé plus tard
            )
            
            opportunities.append(opportunity)
        
        return opportunities
    
    def optimize_coupon_smart(self, opportunities: List[MatchOpportunity],
                            target_size: int = 6,
                            strategy: str = 'balanced') -> List[MatchOpportunity]:
        """
        Optimisation intelligente du coupon selon différentes stratégies
        
        Strategies:
        - 'balanced': Équilibre entre rendement et risque
        - 'high_confidence': Privilégie la sécurité (confiance élevée)
        - 'value_hunting': Recherche les opportunités à forte valeur
        - 'anti_correlation': Minimise les corrélations entre prédictions
        - 'kelly_optimal': Optimise selon le critère de Kelly
        """
        
        if strategy == 'balanced':
            return self._optimize_balanced(opportunities, target_size)
        elif strategy == 'high_confidence':
            return self._optimize_high_confidence(opportunities, target_size)
        elif strategy == 'value_hunting':
            return self._optimize_value_hunting(opportunities, target_size)
        elif strategy == 'anti_correlation':
            return self._optimize_anti_correlation(opportunities, target_size)
        elif strategy == 'kelly_optimal':
            return self._optimize_kelly(opportunities, target_size)
        else:
            return self._optimize_balanced(opportunities, target_size)
    
    def _optimize_balanced(self, opportunities: List[MatchOpportunity], 
                          target_size: int) -> List[MatchOpportunity]:
        """Stratégie équilibrée : meilleur rapport rendement/risque"""
        
        # Filtrer les opportunités viables
        viable_ops = [op for op in opportunities 
                     if op.confidence >= 65 and op.expected_value > -0.1]
        
        if len(viable_ops) <= target_size:
            return viable_ops
        
        # Tri par score d'optimisation global
        viable_ops.sort(key=lambda x: x.optimization_score, reverse=True)
        
        # Sélection avec diversification
        selected = []
        leagues_used = set()
        prediction_types_used = set()
        
        for op in viable_ops:
            if len(selected) >= target_size:
                break
                
            # Favoriser diversification
            diversity_bonus = 0
            if op.league not in leagues_used:
                diversity_bonus += 0.1
            if op.prediction_type not in prediction_types_used:
                diversity_bonus += 0.05
                
            # Score ajusté avec diversification
            adjusted_score = op.optimization_score + diversity_bonus
            
            # Vérifier corrélations avec sélections existantes
            correlation_penalty = self._calculate_correlation_penalty(op, selected)
            final_score = adjusted_score - correlation_penalty
            
            if final_score > 0.4:  # Seuil minimum
                selected.append(op)
                leagues_used.add(op.league)
                prediction_types_used.add(op.prediction_type)
        
        return selected[:target_size]
    
    def _optimize_high_confidence(self, opportunities: List[MatchOpportunity],
                                 target_size: int) -> List[MatchOpportunity]:
        """Stratégie sécurisée : privilégier confiance élevée"""
        
        # Filtrer haute confiance
        high_conf_ops = [op for op in opportunities if op.confidence >= 80]
        
        # Tri par confiance puis par valeur attendue
        high_conf_ops.sort(key=lambda x: (x.confidence, x.expected_value), reverse=True)
        
        # Sélection avec diversification minimale
        selected = []
        leagues_used = set()
        
        for op in high_conf_ops:
            if len(selected) >= target_size:
                break
                
            # Éviter concentration excessive sur une ligue
            league_count = sum(1 for s in selected if s.league == op.league)
            if league_count >= target_size // 2:
                continue
                
            selected.append(op)
            leagues_used.add(op.league)
        
        return selected
    
    def _optimize_value_hunting(self, opportunities: List[MatchOpportunity],
                               target_size: int) -> List[MatchOpportunity]:
        """Stratégie agressive : chasse aux valeurs"""
        
        # Filtrer valeurs positives
        value_ops = [op for op in opportunities 
                    if op.expected_value > 0.1 and op.confidence >= 60]
        
        # Tri par valeur attendue et critère de Kelly
        value_ops.sort(key=lambda x: (x.expected_value, x.kelly_criterion), reverse=True)
        
        selected = []
        total_kelly = 0
        
        for op in value_ops:
            if len(selected) >= target_size:
                break
                
            # Limiter exposition selon Kelly
            if total_kelly + op.kelly_criterion <= 0.25:  # Max 25% bankroll
                selected.append(op)
                total_kelly += op.kelly_criterion
        
        return selected
    
    def _optimize_anti_correlation(self, opportunities: List[MatchOpportunity],
                                  target_size: int) -> List[MatchOpportunity]:
        """Stratégie de diversification : minimiser corrélations"""
        
        # Commencer par la meilleure opportunité
        viable_ops = [op for op in opportunities 
                     if op.confidence >= 70 and op.expected_value > 0]
        
        if not viable_ops:
            return []
        
        viable_ops.sort(key=lambda x: x.optimization_score, reverse=True)
        selected = [viable_ops[0]]
        
        # Sélection itérative minimisant corrélations
        remaining = viable_ops[1:]
        
        while len(selected) < target_size and remaining:
            best_candidate = None
            lowest_correlation = float('inf')
            
            for candidate in remaining:
                total_correlation = self._calculate_total_correlation(candidate, selected)
                
                # Score ajusté par corrélation
                adjusted_score = candidate.optimization_score - (total_correlation * 0.5)
                
                if adjusted_score > lowest_correlation:
                    lowest_correlation = adjusted_score
                    best_candidate = candidate
            
            if best_candidate and lowest_correlation > 0.3:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        return selected
    
    def _optimize_kelly(self, opportunities: List[MatchOpportunity],
                       target_size: int) -> List[MatchOpportunity]:
        """Optimisation selon le critère de Kelly"""
        
        # Filtrer critère Kelly positif
        kelly_ops = [op for op in opportunities if op.kelly_criterion > 0.02]
        
        # Tri par Kelly puis par Sharpe ratio
        kelly_ops.sort(key=lambda x: (x.kelly_criterion, x.sharpe_ratio), reverse=True)
        
        selected = []
        total_kelly_weight = 0
        
        for op in kelly_ops:
            if len(selected) >= target_size:
                break
                
            # Respecter limite Kelly totale
            if total_kelly_weight + op.kelly_criterion <= 0.20:
                selected.append(op)
                total_kelly_weight += op.kelly_criterion
        
        return selected
    
    def calculate_coupon_metrics(self, selected_matches: List[MatchOpportunity]) -> Dict:
        """Calculer métriques complètes du coupon optimisé"""
        if not selected_matches:
            return {}
        
        # Métriques de base
        total_odds = np.prod([match.odds for match in selected_matches])
        avg_confidence = np.mean([match.confidence for match in selected_matches])
        combined_probability = np.prod([match.confidence / 100 for match in selected_matches])
        
        # Métriques avancées
        total_kelly = sum(match.kelly_criterion for match in selected_matches)
        avg_sharpe = np.mean([match.sharpe_ratio for match in selected_matches])
        portfolio_ev = total_odds * combined_probability - 1
        
        # Diversification
        unique_leagues = len(set(match.league for match in selected_matches))
        unique_types = len(set(match.prediction_type for match in selected_matches))
        diversification_ratio = (unique_leagues + unique_types) / (2 * len(selected_matches))
        
        # Score de qualité global (0-100)
        quality_score = (
            min(100, avg_confidence) * 0.30 +
            min(100, max(0, portfolio_ev * 20 + 50)) * 0.25 +
            min(100, total_kelly * 200) * 0.20 +
            min(100, diversification_ratio * 100) * 0.15 +
            min(100, min(10, avg_sharpe) * 10) * 0.10
        )
        
        return {
            'total_odds': round(total_odds, 2),
            'win_probability': round(combined_probability * 100, 2),
            'expected_return': round(portfolio_ev, 3),
            'avg_confidence': round(avg_confidence, 1),
            'kelly_weight': round(total_kelly, 3),
            'avg_sharpe_ratio': round(avg_sharpe, 2),
            'diversification_score': round(diversification_ratio * 100, 1),
            'quality_score': round(quality_score, 1),
            'recommended_stake': round(min(total_kelly, 0.15), 3),
            'risk_level': self._classify_risk_level(quality_score, total_kelly),
            'strategy_used': 'optimized_selection'
        }
    
    def suggest_optimal_coupon_size(self, opportunities: List[MatchOpportunity]) -> int:
        """Suggérer taille optimale du coupon selon les opportunités"""
        
        high_quality = len([op for op in opportunities if op.optimization_score > 0.7])
        medium_quality = len([op for op in opportunities if 0.5 <= op.optimization_score <= 0.7])
        
        if high_quality >= 8:
            return min(10, high_quality + 2)
        elif high_quality >= 5:
            return min(8, high_quality + 1)
        elif high_quality >= 3:
            return min(6, high_quality + medium_quality // 2)
        else:
            return max(3, min(5, high_quality + medium_quality // 3))
    
    def _calculate_risk_score(self, match: Dict) -> float:
        """Calculer score de risque personnalisé"""
        base_risk = (100 - match['confidence']) / 100
        odds_risk = min(1.0, (match['odds'] - 1) / 5)  # Normalisé sur cotes 1-6
        return (base_risk * 0.7 + odds_risk * 0.3) * 100
    
    def _simulate_form_metric(self) -> float:
        """Simuler métrique de forme récente"""
        return np.random.uniform(0.3, 0.95)
    
    def _simulate_accuracy_metric(self) -> float:
        """Simuler précision historique"""
        return np.random.uniform(0.60, 0.85)
    
    def _simulate_market_efficiency(self) -> float:
        """Simuler efficacité du marché (plus élevé = moins d'opportunités)"""
        return np.random.uniform(0.70, 0.95)
    
    def _calculate_correlation_penalty(self, candidate: MatchOpportunity,
                                     selected: List[MatchOpportunity]) -> float:
        """Calculer pénalité de corrélation"""
        if not selected:
            return 0
        
        total_penalty = 0
        for sel in selected:
            # Corrélation par ligue
            if candidate.league == sel.league:
                total_penalty += 0.05
            
            # Corrélation par type de prédiction
            corr_key = tuple(sorted([candidate.prediction_type, sel.prediction_type]))
            if corr_key in self.prediction_correlations:
                total_penalty += self.prediction_correlations[corr_key] * 0.1
        
        return total_penalty
    
    def _calculate_total_correlation(self, candidate: MatchOpportunity,
                                   selected: List[MatchOpportunity]) -> float:
        """Calculer corrélation totale avec sélections existantes"""
        if not selected:
            return 0
        
        correlations = []
        for sel in selected:
            league_corr = 0.3 if candidate.league == sel.league else 0
            
            corr_key = tuple(sorted([candidate.prediction_type, sel.prediction_type]))
            type_corr = self.prediction_correlations.get(corr_key, 0)
            
            total_corr = max(league_corr, type_corr)
            correlations.append(total_corr)
        
        return np.mean(correlations)
    
    def _classify_risk_level(self, quality_score: float, kelly_weight: float) -> str:
        """Classifier niveau de risque du coupon"""
        if quality_score >= 80 and kelly_weight <= 0.10:
            return "Très Faible"
        elif quality_score >= 70 and kelly_weight <= 0.15:
            return "Faible"
        elif quality_score >= 60:
            return "Modéré"
        elif quality_score >= 50:
            return "Élevé"
        else:
            return "Très Élevé"

# Exemple d'utilisation
if __name__ == "__main__":
    optimizer = IntelligentCouponOptimizer()
    
    # Simulation d'opportunités
    mock_matches = [
        {
            'match_id': i,
            'home_team': f'Team A{i}',
            'away_team': f'Team B{i}',
            'league': np.random.choice(['premier_league', 'la_liga', 'bundesliga']),
            'prediction_type': np.random.choice(['match_result', 'both_teams_score', 'over_2_5_goals']),
            'confidence': np.random.uniform(60, 95),
            'odds': np.random.uniform(1.3, 4.0),
            'expected_value': np.random.uniform(-0.2, 0.4)
        }
        for i in range(20)
    ]
    
    # Génération des opportunités
    opportunities = optimizer.generate_match_opportunities(mock_matches, {})
    
    # Test des différentes stratégies
    strategies = ['balanced', 'high_confidence', 'value_hunting', 'anti_correlation', 'kelly_optimal']
    
    print("=== TEST OPTIMISEUR INTELLIGENT ===\n")
    
    for strategy in strategies:
        print(f"🎯 Stratégie: {strategy.upper()}")
        selected = optimizer.optimize_coupon_smart(opportunities, 6, strategy)
        metrics = optimizer.calculate_coupon_metrics(selected)
        
        print(f"   Sélections: {len(selected)} matchs")
        print(f"   Cote totale: {metrics.get('total_odds', 0)}")
        print(f"   Probabilité gain: {metrics.get('win_probability', 0)}%")
        print(f"   Score qualité: {metrics.get('quality_score', 0)}/100")
        print(f"   Niveau risque: {metrics.get('risk_level', 'N/A')}")
        print()
    
    # Suggestion taille optimale
    optimal_size = optimizer.suggest_optimal_coupon_size(opportunities)
    print(f"📏 Taille optimale suggérée: {optimal_size} matchs")