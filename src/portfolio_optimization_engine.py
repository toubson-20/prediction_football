"""
📊 PORTFOLIO OPTIMIZATION ENGINE - OPTIMISATION AVANCÉE PORTEFEUILLE PARIS
Optimisation mathématique avancée des portefeuilles de paris avec gestion des risques

Version: 3.0 - Phase 3 ML Transformation  
Créé: 23 août 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, multivariate_normal
import itertools
import warnings
warnings.filterwarnings('ignore')

# Import des composants précédents
try:
    from intelligent_betting_coupon import BettingPrediction, CouponOptimizer
    from confidence_scoring_engine import AdvancedConfidenceScorer
except ImportError as e:
    print(f"Warning: Import manque - {e}")

class RiskMetricsCalculator:
    """Calculateur de métriques de risque avancées"""
    
    def __init__(self):
        self.correlation_cache = {}
        self.volatility_cache = {}
        
    def calculate_portfolio_var(self, predictions: List[BettingPrediction], 
                              confidence_level: float = 0.05) -> Dict:
        """Calcule la Value at Risk (VaR) du portefeuille"""
        
        stakes = np.array([pred.get_kelly_stake() for pred in predictions])
        odds = np.array([pred.odds for pred in predictions])
        win_probs = np.array([pred.confidence_score / 100.0 for pred in predictions])
        
        # Simulation Monte Carlo pour la distribution des gains/pertes
        n_simulations = 10000
        portfolio_returns = []
        
        # Matrice de corrélation (simplifiée)
        correlation_matrix = self._build_correlation_matrix(predictions)
        
        for _ in range(n_simulations):
            # Génération de résultats corrélés
            random_outcomes = self._generate_correlated_outcomes(win_probs, correlation_matrix)
            
            # Calcul du return du portefeuille
            portfolio_return = 0
            for i, outcome in enumerate(random_outcomes):
                if outcome:  # Pari gagné
                    portfolio_return += stakes[i] * (odds[i] - 1)
                else:  # Pari perdu
                    portfolio_return -= stakes[i]
            
            portfolio_returns.append(portfolio_return)
        
        portfolio_returns = np.array(portfolio_returns)
        
        # Calculs VaR
        var_95 = np.percentile(portfolio_returns, confidence_level * 100)
        var_99 = np.percentile(portfolio_returns, 1)
        expected_return = np.mean(portfolio_returns)
        volatility = np.std(portfolio_returns)
        
        # Expected Shortfall (Conditional VaR)
        es_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
        
        return {
            'expected_return': round(expected_return, 2),
            'volatility': round(volatility, 2),
            'var_95': round(var_95, 2),
            'var_99': round(var_99, 2),
            'expected_shortfall_95': round(es_95, 2),
            'sharpe_ratio': round(expected_return / volatility if volatility > 0 else 0, 3),
            'skewness': round(float(pd.Series(portfolio_returns).skew()), 3),
            'kurtosis': round(float(pd.Series(portfolio_returns).kurtosis()), 3)
        }
    
    def _build_correlation_matrix(self, predictions: List[BettingPrediction]) -> np.ndarray:
        """Construit la matrice de corrélation entre prédictions"""
        
        n = len(predictions)
        correlation_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i+1, n):
                # Corrélation basée sur la similarité des prédictions
                correlation = self._calculate_prediction_correlation(predictions[i], predictions[j])
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        return correlation_matrix
    
    def _calculate_prediction_correlation(self, pred1: BettingPrediction, pred2: BettingPrediction) -> float:
        """Calcule la corrélation entre deux prédictions"""
        
        cache_key = f"{pred1.prediction_type}_{pred2.prediction_type}"
        
        if cache_key in self.correlation_cache:
            return self.correlation_cache[cache_key]
        
        # Matrice de corrélation prédéfinie
        correlation_rules = {
            ('match_result', 'total_goals'): 0.25,
            ('match_result', 'both_teams_scored'): 0.30,
            ('total_goals', 'over_2_5_goals'): 0.85,
            ('total_goals', 'both_teams_scored'): 0.60,
            ('home_goals', 'total_goals'): 0.70,
            ('away_goals', 'total_goals'): 0.70,
            ('corners_total', 'cards_total'): 0.15,
            ('first_half_result', 'match_result'): 0.45,
            ('clean_sheet_home', 'total_goals'): -0.40,
            ('clean_sheet_away', 'total_goals'): -0.40
        }
        
        # Types identiques = corrélation parfaite
        if pred1.prediction_type == pred2.prediction_type:
            correlation = 1.0
        else:
            # Recherche dans les règles
            key1 = (pred1.prediction_type, pred2.prediction_type)
            key2 = (pred2.prediction_type, pred1.prediction_type)
            correlation = correlation_rules.get(key1, correlation_rules.get(key2, 0.05))
        
        # Ajustement selon contexte
        if pred1.league == pred2.league:
            correlation += 0.05  # Bonus même ligue
        
        # Cache du résultat
        self.correlation_cache[cache_key] = correlation
        
        return min(0.95, max(-0.95, correlation))
    
    def _generate_correlated_outcomes(self, win_probs: np.ndarray, 
                                    correlation_matrix: np.ndarray) -> np.ndarray:
        """Génère des résultats corrélés selon la matrice de corrélation"""
        
        # Transformation vers variables normales
        normal_vars = np.random.multivariate_normal(np.zeros(len(win_probs)), correlation_matrix)
        
        # Transformation inverse vers probabilités uniformes puis Bernoulli
        uniform_vars = norm.cdf(normal_vars)
        outcomes = uniform_vars < win_probs
        
        return outcomes
    
    def calculate_maximum_drawdown(self, predictions: List[BettingPrediction],
                                 n_scenarios: int = 1000) -> Dict:
        """Calcule le maximum drawdown potentiel"""
        
        stakes = [pred.get_kelly_stake() for pred in predictions]
        odds = [pred.odds for pred in predictions]
        win_probs = [pred.confidence_score / 100.0 for pred in predictions]
        
        max_drawdowns = []
        
        for _ in range(n_scenarios):
            portfolio_value = sum(stakes)  # Capital initial
            running_max = portfolio_value
            max_drawdown = 0
            
            # Simulation séquentielle des résultats
            for i, (stake, odd, prob) in enumerate(zip(stakes, odds, win_probs)):
                if np.random.random() < prob:  # Pari gagné
                    portfolio_value += stake * (odd - 1)
                else:  # Pari perdu
                    portfolio_value -= stake
                
                # Mise à jour du drawdown
                if portfolio_value > running_max:
                    running_max = portfolio_value
                
                current_drawdown = (running_max - portfolio_value) / running_max
                max_drawdown = max(max_drawdown, current_drawdown)
            
            max_drawdowns.append(max_drawdown)
        
        return {
            'expected_max_drawdown': round(np.mean(max_drawdowns), 4),
            'worst_case_drawdown_95': round(np.percentile(max_drawdowns, 95), 4),
            'worst_case_drawdown_99': round(np.percentile(max_drawdowns, 99), 4)
        }

class AdvancedPortfolioOptimizer:
    """Optimisateur avancé de portefeuilles avec contraintes multiples"""
    
    def __init__(self):
        self.risk_calculator = RiskMetricsCalculator()
        
        # Paramètres d'optimisation par défaut
        self.optimization_config = {
            'max_portfolio_size': 12,
            'min_portfolio_size': 5,
            'max_single_bet_weight': 0.15,  # 15% maximum par pari
            'min_expected_return': 0.05,     # 5% minimum
            'max_portfolio_risk': 0.25,     # 25% maximum VAR
            'target_sharpe_ratio': 1.0,
            'max_correlation': 0.6,
            'risk_free_rate': 0.02
        }
        
    def optimize_portfolio_markowitz(self, predictions: List[BettingPrediction],
                                   objective: str = 'max_sharpe') -> Dict:
        """Optimisation Markowitz moderne avec contraintes"""
        
        n_predictions = len(predictions)
        
        if n_predictions < self.optimization_config['min_portfolio_size']:
            return {'error': 'Pas assez de prédictions pour optimisation'}
        
        # Données d'entrée
        expected_returns = np.array([pred.expected_value for pred in predictions])
        stakes = np.array([pred.get_kelly_stake() for pred in predictions])
        
        # Matrice de covariance (approximée)
        correlation_matrix = self.risk_calculator._build_correlation_matrix(predictions)
        volatilities = np.array([self._estimate_prediction_volatility(pred) for pred in predictions])
        covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Fonction objectif
        def portfolio_objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            
            if objective == 'max_sharpe':
                return -(portfolio_return - self.optimization_config['risk_free_rate']) / (portfolio_risk + 1e-8)
            elif objective == 'min_risk':
                return portfolio_risk
            elif objective == 'max_return':
                return -portfolio_return
            else:
                # Utility function
                risk_aversion = 2.0
                return -(portfolio_return - 0.5 * risk_aversion * portfolio_risk**2)
        
        # Contraintes
        constraints = []
        
        # Somme des poids = 1
        constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        # Rendement minimum
        if self.optimization_config['min_expected_return'] > 0:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: np.dot(w, expected_returns) - self.optimization_config['min_expected_return']
            })
        
        # Risque maximum
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: self.optimization_config['max_portfolio_risk'] - np.sqrt(np.dot(w, np.dot(covariance_matrix, w)))
        })
        
        # Limites des poids
        bounds = [(0, self.optimization_config['max_single_bet_weight']) for _ in range(n_predictions)]
        
        # Point de départ (équipondéré)
        initial_weights = np.ones(n_predictions) / n_predictions
        
        # Optimisation
        result = minimize(
            portfolio_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            
            # Sélection des prédictions avec poids non-négligeables
            min_weight_threshold = 0.01
            selected_indices = [i for i, w in enumerate(optimal_weights) if w >= min_weight_threshold]
            
            optimized_predictions = [predictions[i] for i in selected_indices]
            optimized_weights = [optimal_weights[i] for i in selected_indices]
            
            # Renormalisation
            total_weight = sum(optimized_weights)
            optimized_weights = [w / total_weight for w in optimized_weights]
            
            # Métriques du portefeuille optimisé
            portfolio_metrics = self._calculate_optimized_portfolio_metrics(
                optimized_predictions, optimized_weights
            )
            
            return {
                'status': 'success',
                'optimization_method': 'markowitz',
                'objective': objective,
                'selected_predictions': optimized_predictions,
                'optimal_weights': optimized_weights,
                'portfolio_metrics': portfolio_metrics,
                'optimization_result': {
                    'converged': result.success,
                    'iterations': result.nit,
                    'objective_value': result.fun
                }
            }
        
        else:
            return {
                'status': 'failed',
                'error': 'Optimisation échouée',
                'message': result.message
            }
    
    def optimize_portfolio_genetic(self, predictions: List[BettingPrediction],
                                 generations: int = 100) -> Dict:
        """Optimisation par algorithme génétique"""
        
        n_predictions = len(predictions)
        
        # Fonction fitness (à maximiser)
        def fitness_function(weights):
            # Sélection des prédictions selon les poids
            selected_indices = [i for i, w in enumerate(weights) if w > 0.01]
            
            if len(selected_indices) < self.optimization_config['min_portfolio_size']:
                return -1000  # Pénalité forte
            
            if len(selected_indices) > self.optimization_config['max_portfolio_size']:
                return -500   # Pénalité modérée
            
            selected_preds = [predictions[i] for i in selected_indices]
            selected_weights = [weights[i] for i in selected_indices]
            
            # Renormalisation
            total_weight = sum(selected_weights)
            if total_weight > 0:
                selected_weights = [w / total_weight for w in selected_weights]
            
            # Calcul fitness
            portfolio_return = sum(pred.expected_value * weight 
                                 for pred, weight in zip(selected_preds, selected_weights))
            
            # Pénalité pour corrélation excessive
            correlation_penalty = self._calculate_correlation_penalty(selected_preds)
            
            # Pénalité pour concentration
            concentration_penalty = max(0, max(selected_weights) - self.optimization_config['max_single_bet_weight'])
            
            fitness = portfolio_return - correlation_penalty * 0.1 - concentration_penalty * 0.5
            
            return fitness
        
        # Limites pour l'algorithme génétique
        bounds = [(0, 1) for _ in range(n_predictions)]
        
        # Optimisation
        result = differential_evolution(
            lambda x: -fitness_function(x),  # Minimisation -> maximisation
            bounds,
            maxiter=generations,
            seed=42,
            atol=1e-6,
            tol=1e-6
        )
        
        if result.success:
            optimal_weights = result.x
            
            # Sélection finale
            selected_indices = [i for i, w in enumerate(optimal_weights) if w > 0.01]
            optimized_predictions = [predictions[i] for i in selected_indices]
            optimized_weights = [optimal_weights[i] for i in selected_indices]
            
            # Renormalisation
            total_weight = sum(optimized_weights)
            optimized_weights = [w / total_weight for w in optimized_weights]
            
            # Métriques
            portfolio_metrics = self._calculate_optimized_portfolio_metrics(
                optimized_predictions, optimized_weights
            )
            
            return {
                'status': 'success',
                'optimization_method': 'genetic',
                'selected_predictions': optimized_predictions,
                'optimal_weights': optimized_weights,
                'portfolio_metrics': portfolio_metrics,
                'fitness_score': -result.fun
            }
        
        else:
            return {
                'status': 'failed',
                'error': 'Optimisation génétique échouée'
            }
    
    def optimize_portfolio_kelly_advanced(self, predictions: List[BettingPrediction]) -> Dict:
        """Optimisation Kelly avancée avec contraintes de portefeuille"""
        
        # Calcul des fractions Kelly individuelles
        kelly_fractions = []
        for pred in predictions:
            prob = pred.confidence_score / 100.0
            odds = pred.odds
            
            # Kelly fraction: f = (bp - q) / b
            b = odds - 1
            q = 1 - prob
            kelly = (b * prob - q) / b
            
            kelly_fractions.append(max(0, kelly))  # Pas de mise négative
        
        # Ajustement pour contraintes de portefeuille
        total_kelly = sum(kelly_fractions)
        
        if total_kelly > 1.0:  # Surmise -> normalisation
            kelly_fractions = [k / total_kelly for k in kelly_fractions]
        
        # Limitation des poids individuels
        max_weight = self.optimization_config['max_single_bet_weight']
        for i, kelly in enumerate(kelly_fractions):
            if kelly > max_weight:
                # Redistribution de l'excès
                excess = kelly - max_weight
                kelly_fractions[i] = max_weight
                
                # Redistribution proportionnelle sur les autres
                other_indices = [j for j in range(len(kelly_fractions)) if j != i]
                if other_indices:
                    redistribution = excess / len(other_indices)
                    for j in other_indices:
                        kelly_fractions[j] += redistribution
        
        # Sélection des prédictions avec Kelly > seuil
        min_kelly = 0.005  # 0.5% minimum
        selected_data = [(pred, kelly) for pred, kelly in zip(predictions, kelly_fractions) 
                        if kelly >= min_kelly]
        
        if not selected_data:
            return {'error': 'Aucune prédiction avec Kelly positif significatif'}
        
        selected_predictions, selected_weights = zip(*selected_data)
        
        # Renormalisation finale
        total_weight = sum(selected_weights)
        selected_weights = [w / total_weight for w in selected_weights]
        
        # Métriques
        portfolio_metrics = self._calculate_optimized_portfolio_metrics(
            list(selected_predictions), list(selected_weights)
        )
        
        return {
            'status': 'success',
            'optimization_method': 'kelly_advanced',
            'selected_predictions': list(selected_predictions),
            'optimal_weights': list(selected_weights),
            'portfolio_metrics': portfolio_metrics,
            'original_kelly_sum': total_kelly
        }
    
    def _estimate_prediction_volatility(self, prediction: BettingPrediction) -> float:
        """Estime la volatilité d'une prédiction"""
        
        prob = prediction.confidence_score / 100.0
        odds = prediction.odds
        
        # Volatilité basée sur la variance Bernoulli
        win_amount = odds - 1
        lose_amount = -1
        
        expected_return = prob * win_amount + (1 - prob) * lose_amount
        variance = prob * (win_amount - expected_return)**2 + (1 - prob) * (lose_amount - expected_return)**2
        
        return np.sqrt(variance)
    
    def _calculate_correlation_penalty(self, predictions: List[BettingPrediction]) -> float:
        """Calcule la pénalité pour corrélation excessive"""
        
        if len(predictions) < 2:
            return 0.0
        
        total_correlation = 0.0
        pairs = 0
        
        for i in range(len(predictions)):
            for j in range(i+1, len(predictions)):
                correlation = self.risk_calculator._calculate_prediction_correlation(
                    predictions[i], predictions[j]
                )
                
                if correlation > self.optimization_config['max_correlation']:
                    total_correlation += correlation - self.optimization_config['max_correlation']
                
                pairs += 1
        
        return total_correlation / pairs if pairs > 0 else 0.0
    
    def _calculate_optimized_portfolio_metrics(self, predictions: List[BettingPrediction],
                                             weights: List[float]) -> Dict:
        """Calcule les métriques d'un portefeuille optimisé"""
        
        # Expected return pondéré
        expected_return = sum(pred.expected_value * weight 
                            for pred, weight in zip(predictions, weights))
        
        # Risk metrics
        var_metrics = self.risk_calculator.calculate_portfolio_var(predictions)
        drawdown_metrics = self.risk_calculator.calculate_maximum_drawdown(predictions)
        
        # Diversification metrics
        n_predictions = len(predictions)
        max_weight = max(weights) if weights else 0
        
        # HHI (Herfindahl-Hirschman Index) pour concentration
        hhi = sum(w**2 for w in weights)
        diversification_ratio = (1 - hhi) / (1 - 1/n_predictions) if n_predictions > 1 else 0
        
        return {
            'portfolio_size': n_predictions,
            'expected_return': round(expected_return, 4),
            'max_weight': round(max_weight, 4),
            'diversification_ratio': round(diversification_ratio, 4),
            'hhi_concentration': round(hhi, 4),
            'risk_metrics': var_metrics,
            'drawdown_metrics': drawdown_metrics
        }

class MultiObjectiveOptimizer:
    """Optimiseur multi-objectifs (rendement vs risque vs diversification)"""
    
    def __init__(self):
        self.advanced_optimizer = AdvancedPortfolioOptimizer()
        
    def pareto_optimization(self, predictions: List[BettingPrediction],
                          n_points: int = 50) -> Dict:
        """Génère la frontière de Pareto rendement-risque"""
        
        pareto_solutions = []
        
        # Range de risk aversion parameters
        risk_aversions = np.linspace(0.1, 10.0, n_points)
        
        for risk_aversion in risk_aversions:
            # Optimisation avec utility function
            self.advanced_optimizer.optimization_config['risk_free_rate'] = 1.0 / risk_aversion
            
            result = self.advanced_optimizer.optimize_portfolio_markowitz(
                predictions, objective='utility'
            )
            
            if result.get('status') == 'success':
                metrics = result['portfolio_metrics']
                
                pareto_point = {
                    'risk_aversion': risk_aversion,
                    'expected_return': metrics['expected_return'],
                    'portfolio_risk': metrics['risk_metrics']['volatility'],
                    'sharpe_ratio': metrics['risk_metrics']['sharpe_ratio'],
                    'portfolio_size': metrics['portfolio_size'],
                    'max_weight': metrics['max_weight'],
                    'predictions': result['selected_predictions'],
                    'weights': result['optimal_weights']
                }
                
                pareto_solutions.append(pareto_point)
        
        # Tri par rendement
        pareto_solutions.sort(key=lambda x: x['expected_return'])
        
        return {
            'pareto_solutions': pareto_solutions,
            'n_solutions': len(pareto_solutions),
            'best_sharpe_solution': max(pareto_solutions, key=lambda x: x['sharpe_ratio']) if pareto_solutions else None
        }
    
    def multi_criteria_optimization(self, predictions: List[BettingPrediction],
                                  criteria_weights: Dict[str, float] = None) -> Dict:
        """Optimisation multi-critères avec pondération"""
        
        if criteria_weights is None:
            criteria_weights = {
                'expected_return': 0.4,
                'risk_minimization': 0.3,
                'diversification': 0.2,
                'kelly_criterion': 0.1
            }
        
        # Optimisation selon chaque critère
        optimization_results = {}
        
        # 1. Maximum expected return
        if criteria_weights.get('expected_return', 0) > 0:
            result = self.advanced_optimizer.optimize_portfolio_markowitz(
                predictions, objective='max_return'
            )
            optimization_results['max_return'] = result
        
        # 2. Minimum risk
        if criteria_weights.get('risk_minimization', 0) > 0:
            result = self.advanced_optimizer.optimize_portfolio_markowitz(
                predictions, objective='min_risk'
            )
            optimization_results['min_risk'] = result
        
        # 3. Kelly optimal
        if criteria_weights.get('kelly_criterion', 0) > 0:
            result = self.advanced_optimizer.optimize_portfolio_kelly_advanced(predictions)
            optimization_results['kelly'] = result
        
        # 4. Maximum Sharpe
        result = self.advanced_optimizer.optimize_portfolio_markowitz(
            predictions, objective='max_sharpe'
        )
        optimization_results['max_sharpe'] = result
        
        # Combinaison pondérée des solutions
        combined_solution = self._combine_solutions(optimization_results, criteria_weights)
        
        return {
            'individual_solutions': optimization_results,
            'combined_solution': combined_solution,
            'criteria_weights': criteria_weights
        }
    
    def _combine_solutions(self, solutions: Dict[str, Dict], 
                          weights: Dict[str, float]) -> Dict:
        """Combine plusieurs solutions d'optimisation"""
        
        # Extraction des prédictions et poids de chaque solution
        all_predictions = []
        prediction_scores = {}
        
        for method, result in solutions.items():
            if result.get('status') != 'success':
                continue
            
            method_weight = weights.get(method, weights.get('max_sharpe', 0.25))
            
            for pred, pred_weight in zip(result['selected_predictions'], result['optimal_weights']):
                pred_id = f"{pred.prediction_type}_{pred.league}"
                
                if pred_id not in prediction_scores:
                    prediction_scores[pred_id] = {'prediction': pred, 'total_score': 0.0}
                
                prediction_scores[pred_id]['total_score'] += method_weight * pred_weight
        
        # Sélection des prédictions avec les meilleurs scores combinés
        sorted_predictions = sorted(
            prediction_scores.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
        
        # Limitation du portefeuille
        max_size = self.advanced_optimizer.optimization_config['max_portfolio_size']
        selected_items = sorted_predictions[:max_size]
        
        final_predictions = [item[1]['prediction'] for item in selected_items]
        raw_weights = [item[1]['total_score'] for item in selected_items]
        
        # Normalisation des poids
        total_weight = sum(raw_weights)
        final_weights = [w / total_weight for w in raw_weights] if total_weight > 0 else []
        
        # Métriques du portefeuille combiné
        if final_predictions and final_weights:
            portfolio_metrics = self.advanced_optimizer._calculate_optimized_portfolio_metrics(
                final_predictions, final_weights
            )
            
            return {
                'status': 'success',
                'method': 'multi_criteria_combined',
                'selected_predictions': final_predictions,
                'optimal_weights': final_weights,
                'portfolio_metrics': portfolio_metrics
            }
        
        else:
            return {'status': 'failed', 'error': 'Aucune solution combinée viable'}

def test_portfolio_optimization_system():
    """Test du système d'optimisation de portefeuilles"""
    
    print("=== TEST PORTFOLIO OPTIMIZATION ENGINE ===")
    
    # Génération de prédictions de test
    test_predictions = []
    
    prediction_types = ['match_result', 'total_goals', 'both_teams_scored', 'over_2_5_goals', 
                       'home_goals', 'corners_total', 'cards_total']
    
    for i, pred_type in enumerate(prediction_types):
        pred = BettingPrediction(
            prediction_type=pred_type,
            prediction_value=f'value_{i}',
            confidence_score=np.random.uniform(65, 90),
            odds=np.random.uniform(1.5, 4.0),
            expected_value=np.random.uniform(-0.1, 0.3),
            model_used=f'model_{i}',
            league='Premier_League',
            match_context={'match_id': f'match_{i%3}'}
        )
        test_predictions.append(pred)
    
    print(f"Généré {len(test_predictions)} prédictions de test")
    
    # Test optimiseur avancé
    print(f"\\n--- Test Optimisation Markowitz ---")
    
    advanced_optimizer = AdvancedPortfolioOptimizer()
    
    # Optimisation Max Sharpe
    sharpe_result = advanced_optimizer.optimize_portfolio_markowitz(
        test_predictions, objective='max_sharpe'
    )
    
    if sharpe_result['status'] == 'success':
        print(f"✅ Optimisation Max Sharpe réussie:")
        print(f"   Prédictions sélectionnées: {len(sharpe_result['selected_predictions'])}")
        print(f"   Expected Return: {sharpe_result['portfolio_metrics']['expected_return']}")
        print(f"   Sharpe Ratio: {sharpe_result['portfolio_metrics']['risk_metrics']['sharpe_ratio']}")
        print(f"   VaR 95%: {sharpe_result['portfolio_metrics']['risk_metrics']['var_95']}")
        print(f"   Max Drawdown: {sharpe_result['portfolio_metrics']['drawdown_metrics']['expected_max_drawdown']}")
    
    # Test optimisation génétique
    print(f"\\n--- Test Optimisation Génétique ---")
    
    genetic_result = advanced_optimizer.optimize_portfolio_genetic(
        test_predictions, generations=50
    )
    
    if genetic_result['status'] == 'success':
        print(f"✅ Optimisation génétique réussie:")
        print(f"   Prédictions sélectionnées: {len(genetic_result['selected_predictions'])}")
        print(f"   Expected Return: {genetic_result['portfolio_metrics']['expected_return']}")
        print(f"   Fitness Score: {genetic_result['fitness_score']:.4f}")
        print(f"   Diversification: {genetic_result['portfolio_metrics']['diversification_ratio']:.3f}")
    
    # Test Kelly avancé
    print(f"\\n--- Test Kelly Avancé ---")
    
    kelly_result = advanced_optimizer.optimize_portfolio_kelly_advanced(test_predictions)
    
    if kelly_result['status'] == 'success':
        print(f"✅ Optimisation Kelly réussie:")
        print(f"   Prédictions sélectionnées: {len(kelly_result['selected_predictions'])}")
        print(f"   Kelly sum original: {kelly_result['original_kelly_sum']:.4f}")
        print(f"   Expected Return: {kelly_result['portfolio_metrics']['expected_return']}")
        print(f"   Concentration HHI: {kelly_result['portfolio_metrics']['hhi_concentration']:.4f}")
    
    # Test optimisation multi-objectifs
    print(f"\\n--- Test Multi-Objectif ---")
    
    multi_optimizer = MultiObjectiveOptimizer()
    
    # Frontière de Pareto
    pareto_result = multi_optimizer.pareto_optimization(test_predictions, n_points=10)
    
    print(f"✅ Analyse Pareto:")
    print(f"   Solutions générées: {pareto_result['n_solutions']}")
    
    if pareto_result['best_sharpe_solution']:
        best = pareto_result['best_sharpe_solution']
        print(f"   Meilleure solution Sharpe:")
        print(f"     Return: {best['expected_return']:.4f}")
        print(f"     Risk: {best['portfolio_risk']:.4f}")
        print(f"     Sharpe: {best['sharpe_ratio']:.3f}")
        print(f"     Taille: {best['portfolio_size']}")
    
    # Multi-critères
    criteria_result = multi_optimizer.multi_criteria_optimization(test_predictions)
    
    print(f"\\n✅ Optimisation Multi-Critères:")
    print(f"   Solutions individuelles: {len(criteria_result['individual_solutions'])}")
    
    if criteria_result['combined_solution']['status'] == 'success':
        combined = criteria_result['combined_solution']
        print(f"   Solution combinée:")
        print(f"     Prédictions: {len(combined['selected_predictions'])}")
        print(f"     Expected Return: {combined['portfolio_metrics']['expected_return']}")
        print(f"     Diversification: {combined['portfolio_metrics']['diversification_ratio']:.3f}")
    
    # Comparaison des méthodes
    print(f"\\n--- Comparaison Méthodes ---")
    
    methods = [
        ('Markowitz Max Sharpe', sharpe_result),
        ('Algorithme Génétique', genetic_result),
        ('Kelly Avancé', kelly_result),
        ('Multi-Critères', criteria_result['combined_solution'])
    ]
    
    comparison_data = []
    
    for method_name, result in methods:
        if result.get('status') == 'success':
            metrics = result['portfolio_metrics']
            comparison_data.append({
                'method': method_name,
                'size': metrics['portfolio_size'],
                'expected_return': metrics['expected_return'],
                'sharpe_ratio': metrics.get('risk_metrics', {}).get('sharpe_ratio', 0),
                'diversification': metrics['diversification_ratio']
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        print("\\nTableau comparatif:")
        print(df_comparison.to_string(index=False, float_format='%.4f'))
    
    print("\\n=== TEST TERMINÉ ===")

if __name__ == "__main__":
    test_portfolio_optimization_system()