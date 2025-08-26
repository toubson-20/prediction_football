"""
🎯 INTELLIGENT BETTING COUPON - SYSTÈME DE COUPON ADAPTATIF INTELLIGENT
Génération automatique de coupons de paris optimisés 5-12 prédictions

Version: 3.0 - Phase 3 ML Transformation
Créé: 23 août 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import des composants précédents
try:
    from revolutionary_model_architecture import RevolutionaryModelArchitecture
    from massive_model_trainer import MassiveModelTrainer
    from deep_learning_models import DeepLearningEnsemble
    from intelligent_meta_model import IntelligentMetaModel
    from transfer_learning_system import TransferLearningOrchestrator
except ImportError as e:
    print(f"Warning: Import manque - {e}")

class BettingPrediction:
    """Classe pour une prédiction de pari individuelle"""
    
    def __init__(self, 
                 prediction_type: str,
                 prediction_value: Any,
                 confidence_score: float,
                 odds: float,
                 expected_value: float,
                 model_used: str,
                 league: str,
                 match_context: Dict):
        
        self.prediction_type = prediction_type
        self.prediction_value = prediction_value
        self.confidence_score = confidence_score  # 0-100
        self.odds = odds
        self.expected_value = expected_value
        self.model_used = model_used
        self.league = league
        self.match_context = match_context
        self.risk_category = self._calculate_risk_category()
        
    def _calculate_risk_category(self) -> str:
        """Calcule la catégorie de risque basée sur les cotes et confiance"""
        
        if self.confidence_score >= 85 and self.odds <= 1.60:
            return "SAFE"  # Paris sûrs
        elif self.confidence_score >= 70 and 1.60 < self.odds <= 2.50:
            return "BALANCED"  # Paris équilibrés
        elif self.confidence_score >= 75 and 2.50 < self.odds <= 5.00:
            return "VALUE"  # Paris de valeur
        elif self.confidence_score >= 80 and self.odds > 5.00:
            return "LONGSHOT"  # Paris longshot
        else:
            return "RISKY"  # Paris risqués
    
    def get_kelly_stake(self, bankroll: float = 1000.0) -> float:
        """Calcule la mise optimale selon Kelly Criterion"""
        
        prob_win = self.confidence_score / 100.0
        decimal_odds = self.odds
        
        # Kelly formula: f = (bp - q) / b
        # où b = odds-1, p = probability, q = 1-p
        b = decimal_odds - 1
        q = 1 - prob_win
        
        kelly_fraction = (b * prob_win - q) / b
        
        # Limitation à 5% du bankroll maximum pour sécurité
        kelly_fraction = max(0, min(kelly_fraction, 0.05))
        
        return kelly_fraction * bankroll
    
    def to_dict(self) -> Dict:
        """Conversion en dictionnaire pour sérialisation"""
        
        return {
            'prediction_type': self.prediction_type,
            'prediction_value': self.prediction_value,
            'confidence_score': round(self.confidence_score, 2),
            'odds': self.odds,
            'expected_value': round(self.expected_value, 4),
            'model_used': self.model_used,
            'league': self.league,
            'risk_category': self.risk_category,
            'kelly_stake': round(self.get_kelly_stake(), 2),
            'match_context': self.match_context
        }

class CouponOptimizer:
    """Optimisateur de portefeuille de paris"""
    
    def __init__(self):
        self.max_correlation_threshold = 0.7
        self.min_total_odds = 2.0
        self.max_total_odds = 50.0
        
    def calculate_prediction_correlation(self, pred1: BettingPrediction, pred2: BettingPrediction) -> float:
        """Calcule la corrélation entre deux prédictions"""
        
        # Corrélations prédéfinies basées sur les types de paris
        correlation_matrix = {
            ('match_result', 'total_goals'): 0.3,
            ('match_result', 'both_teams_scored'): 0.4,
            ('total_goals', 'over_2_5_goals'): 0.8,  # Forte corrélation
            ('both_teams_scored', 'over_1_5_goals'): 0.6,
            ('home_goals', 'total_goals'): 0.7,
            ('away_goals', 'total_goals'): 0.7,
            ('corners_total', 'cards_total'): 0.2,
            ('first_half_result', 'match_result'): 0.5,
        }
        
        # Même type de prédiction = corrélation parfaite
        if pred1.prediction_type == pred2.prediction_type:
            return 1.0
        
        # Recherche dans la matrice de corrélation
        key1 = (pred1.prediction_type, pred2.prediction_type)
        key2 = (pred2.prediction_type, pred1.prediction_type)
        
        correlation = correlation_matrix.get(key1, correlation_matrix.get(key2, 0.1))
        
        # Bonus si même match/ligue
        if pred1.league == pred2.league:
            correlation += 0.1
        
        return min(correlation, 1.0)
    
    def optimize_portfolio(self, predictions: List[BettingPrediction], 
                          target_size: int = 8,
                          risk_distribution: Dict[str, float] = None) -> List[BettingPrediction]:
        """Optimise le portefeuille de prédictions"""
        
        if risk_distribution is None:
            risk_distribution = {
                'SAFE': 0.3,      # 30% de paris sûrs
                'BALANCED': 0.4,  # 40% de paris équilibrés  
                'VALUE': 0.2,     # 20% de paris de valeur
                'LONGSHOT': 0.1   # 10% de paris longshot
            }
        
        # Tri par expected value décroissant
        sorted_predictions = sorted(predictions, key=lambda p: p.expected_value, reverse=True)
        
        # Sélection initiale basée sur la distribution de risque
        selected = []
        remaining = list(sorted_predictions)
        
        for risk_cat, target_ratio in risk_distribution.items():
            target_count = int(target_size * target_ratio)
            category_predictions = [p for p in remaining if p.risk_category == risk_cat]
            
            # Sélection des meilleurs de cette catégorie
            selected_from_cat = category_predictions[:target_count]
            selected.extend(selected_from_cat)
            
            # Retirer de la liste restante
            for pred in selected_from_cat:
                if pred in remaining:
                    remaining.remove(pred)
        
        # Compléter si nécessaire avec les meilleures prédictions restantes
        while len(selected) < target_size and remaining:
            # Éviter les corrélations trop fortes
            best_candidate = None
            best_score = -1
            
            for candidate in remaining[:10]:  # Top 10 candidats
                max_correlation = 0
                for selected_pred in selected:
                    correlation = self.calculate_prediction_correlation(candidate, selected_pred)
                    max_correlation = max(max_correlation, correlation)
                
                # Score = expected_value - pénalité corrélation
                score = candidate.expected_value - max_correlation * 0.1
                
                if score > best_score and max_correlation < self.max_correlation_threshold:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                # Si pas de candidat acceptable, prendre le meilleur restant
                if remaining:
                    selected.append(remaining.pop(0))
        
        return selected[:target_size]
    
    def calculate_portfolio_metrics(self, portfolio: List[BettingPrediction]) -> Dict:
        """Calcule les métriques du portefeuille"""
        
        if not portfolio:
            return {}
        
        total_odds = np.prod([pred.odds for pred in portfolio])
        avg_confidence = np.mean([pred.confidence_score for pred in portfolio])
        total_expected_value = sum([pred.expected_value for pred in portfolio])
        
        # Distribution des risques
        risk_distribution = {}
        for pred in portfolio:
            risk_cat = pred.risk_category
            risk_distribution[risk_cat] = risk_distribution.get(risk_cat, 0) + 1
        
        # Calcul de la corrélation moyenne
        correlations = []
        for i, pred1 in enumerate(portfolio):
            for j, pred2 in enumerate(portfolio[i+1:], i+1):
                correlation = self.calculate_prediction_correlation(pred1, pred2)
                correlations.append(correlation)
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        return {
            'portfolio_size': len(portfolio),
            'total_odds': round(total_odds, 2),
            'average_confidence': round(avg_confidence, 2),
            'total_expected_value': round(total_expected_value, 4),
            'risk_distribution': risk_distribution,
            'average_correlation': round(avg_correlation, 3),
            'kelly_total_stake': round(sum([pred.get_kelly_stake() for pred in portfolio]), 2)
        }

class IntelligentBettingCoupon:
    """Système principal de génération de coupons intelligents"""
    
    def __init__(self):
        # Composants IA
        self.architecture = None
        self.meta_model = None
        self.optimizer = CouponOptimizer()
        
        # Configuration par défaut
        self.default_config = {
            'min_predictions': 5,
            'max_predictions': 12,
            'target_predictions': 8,
            'min_confidence': 65.0,
            'max_correlation': 0.7,
            'risk_distribution': {
                'SAFE': 0.25,
                'BALANCED': 0.45,
                'VALUE': 0.25,
                'LONGSHOT': 0.05
            }
        }
        
        # Caches et historique
        self.prediction_cache = {}
        self.coupon_history = []
        
    def initialize_components(self):
        """Initialise les composants IA nécessaires"""
        
        try:
            self.architecture = RevolutionaryModelArchitecture()
            self.meta_model = IntelligentMetaModel()
            print("Composants IA initialisés avec succès")
        except Exception as e:
            print(f"Erreur initialisation composants: {e}")
    
    def generate_predictions_for_match(self, match_data: Dict, available_odds: Dict[str, float] = None) -> List[BettingPrediction]:
        """Génère toutes les prédictions possibles pour un match"""
        
        if available_odds is None:
            available_odds = self._generate_mock_odds()
        
        predictions = []
        league = match_data.get('league', 'Premier_League')
        
        # Types de prédictions disponibles
        prediction_types = [
            'match_result', 'total_goals', 'both_teams_scored', 'over_2_5_goals',
            'home_goals', 'away_goals', 'first_half_result', 'correct_score',
            'double_chance', 'handicap_home', 'corners_total', 'cards_total',
            'over_1_5_goals', 'under_3_5_goals', 'clean_sheet_home', 'clean_sheet_away'
        ]
        
        for pred_type in prediction_types:
            try:
                prediction = self._generate_single_prediction(
                    match_data, pred_type, available_odds, league
                )
                
                if prediction and prediction.confidence_score >= self.default_config['min_confidence']:
                    predictions.append(prediction)
                    
            except Exception as e:
                print(f"Erreur génération prédiction {pred_type}: {e}")
        
        return predictions
    
    def _generate_single_prediction(self, match_data: Dict, prediction_type: str, 
                                   available_odds: Dict, league: str) -> Optional[BettingPrediction]:
        """Génère une prédiction unique pour un type donné"""
        
        # Simulation de prédiction (remplacer par vrais modèles)
        confidence_base = np.random.uniform(60, 95)
        
        # Ajustement confiance selon le contexte
        match_importance = match_data.get('match_importance', 'normal')
        if match_importance == 'high':
            confidence_base *= 0.95  # Légère réduction pour matchs importants
        
        # Génération valeur prédiction selon le type
        if prediction_type == 'match_result':
            prediction_value = np.random.choice(['1', 'X', '2'], p=[0.45, 0.25, 0.30])
            odds_key = f"match_result_{prediction_value}"
        elif prediction_type == 'total_goals':
            prediction_value = np.random.choice(['Over 2.5', 'Under 2.5'], p=[0.6, 0.4])
            odds_key = 'over_2_5_goals' if 'Over' in prediction_value else 'under_2_5_goals'
        elif prediction_type == 'both_teams_scored':
            prediction_value = np.random.choice(['Yes', 'No'], p=[0.65, 0.35])
            odds_key = 'both_teams_scored_yes' if prediction_value == 'Yes' else 'both_teams_scored_no'
        else:
            # Prédictions génériques
            prediction_value = f"{prediction_type}_prediction"
            odds_key = prediction_type
        
        # Récupération des cotes
        odds = available_odds.get(odds_key, np.random.uniform(1.5, 4.0))
        
        # Calcul expected value
        prob_win = confidence_base / 100.0
        expected_value = (prob_win * odds) - 1.0
        
        # Sélection du meilleur modèle (simulation)
        if self.meta_model:
            try:
                ensemble = self.meta_model.get_optimal_model_ensemble(
                    prediction_type, match_data, 
                    ['xgb_model', 'rf_model', 'nn_model'], 
                    ensemble_size=3
                )
                model_used = '+'.join(ensemble['models'][:2])  # Top 2 modèles
                confidence_base *= (1 + ensemble['confidence'] * 0.1)  # Bonus confiance ensemble
            except:
                model_used = 'default_ensemble'
        else:
            model_used = 'simulation_model'
        
        return BettingPrediction(
            prediction_type=prediction_type,
            prediction_value=prediction_value,
            confidence_score=min(confidence_base, 95.0),  # Cap à 95%
            odds=odds,
            expected_value=expected_value,
            model_used=model_used,
            league=league,
            match_context=match_data
        )
    
    def _generate_mock_odds(self) -> Dict[str, float]:
        """Génère des cotes simulées pour testing"""
        
        return {
            'match_result_1': np.random.uniform(1.8, 4.0),
            'match_result_X': np.random.uniform(3.0, 4.5),
            'match_result_2': np.random.uniform(1.8, 4.0),
            'over_2_5_goals': np.random.uniform(1.6, 2.4),
            'under_2_5_goals': np.random.uniform(1.5, 2.2),
            'both_teams_scored_yes': np.random.uniform(1.7, 2.3),
            'both_teams_scored_no': np.random.uniform(1.8, 2.5),
            'total_goals': np.random.uniform(1.8, 3.0),
            'home_goals': np.random.uniform(2.0, 3.5),
            'away_goals': np.random.uniform(2.2, 4.0),
            'corners_total': np.random.uniform(1.9, 2.8),
            'cards_total': np.random.uniform(2.0, 3.2)
        }
    
    def generate_intelligent_coupon(self, matches_data: List[Dict], 
                                  coupon_config: Dict = None) -> Dict:
        """Génère un coupon intelligent multi-matchs"""
        
        if coupon_config is None:
            coupon_config = self.default_config.copy()
        
        print(f"Génération coupon intelligent pour {len(matches_data)} matchs...")
        
        # Génération de toutes les prédictions possibles
        all_predictions = []
        
        for i, match_data in enumerate(matches_data):
            print(f"  Analyse match {i+1}/{len(matches_data)} - {match_data.get('home_team', 'Team A')} vs {match_data.get('away_team', 'Team B')}")
            
            match_predictions = self.generate_predictions_for_match(match_data)
            all_predictions.extend(match_predictions)
        
        print(f"  Total prédictions générées: {len(all_predictions)}")
        
        # Filtrage par confiance minimum
        filtered_predictions = [
            pred for pred in all_predictions 
            if pred.confidence_score >= coupon_config['min_confidence']
        ]
        
        print(f"  Prédictions après filtrage confiance: {len(filtered_predictions)}")
        
        if len(filtered_predictions) < coupon_config['min_predictions']:
            return {
                'status': 'insufficient_predictions',
                'message': f"Pas assez de prédictions confiantes ({len(filtered_predictions)} < {coupon_config['min_predictions']})"
            }
        
        # Optimisation du portefeuille
        target_size = min(coupon_config['target_predictions'], len(filtered_predictions))
        
        optimized_portfolio = self.optimizer.optimize_portfolio(
            filtered_predictions,
            target_size=target_size,
            risk_distribution=coupon_config['risk_distribution']
        )
        
        # Calcul des métriques du coupon
        portfolio_metrics = self.optimizer.calculate_portfolio_metrics(optimized_portfolio)
        
        # Génération du coupon final
        coupon = {
            'coupon_id': f"COUPON_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generation_timestamp': datetime.now().isoformat(),
            'matches_analyzed': len(matches_data),
            'predictions_generated': len(all_predictions),
            'predictions_selected': len(optimized_portfolio),
            
            'portfolio_metrics': portfolio_metrics,
            'predictions': [pred.to_dict() for pred in optimized_portfolio],
            
            'risk_summary': self._generate_risk_summary(optimized_portfolio),
            'betting_advice': self._generate_betting_advice(optimized_portfolio, portfolio_metrics),
            
            'configuration_used': coupon_config,
            'status': 'success'
        }
        
        # Sauvegarde dans l'historique
        self.coupon_history.append(coupon)
        
        return coupon
    
    def _generate_risk_summary(self, portfolio: List[BettingPrediction]) -> Dict:
        """Génère un résumé des risques du portefeuille"""
        
        risk_counts = {}
        for pred in portfolio:
            risk_counts[pred.risk_category] = risk_counts.get(pred.risk_category, 0) + 1
        
        total_kelly = sum([pred.get_kelly_stake() for pred in portfolio])
        avg_confidence = np.mean([pred.confidence_score for pred in portfolio])
        
        return {
            'risk_distribution': risk_counts,
            'average_confidence': round(avg_confidence, 1),
            'total_kelly_stake': round(total_kelly, 2),
            'risk_level': 'LOW' if avg_confidence >= 80 else 'MEDIUM' if avg_confidence >= 70 else 'HIGH'
        }
    
    def _generate_betting_advice(self, portfolio: List[BettingPrediction], metrics: Dict) -> List[str]:
        """Génère des conseils de mise personnalisés"""
        
        advice = []
        
        # Conseil sur les cotes totales
        total_odds = metrics.get('total_odds', 1.0)
        if total_odds > 30:
            advice.append("⚠️ Cotes très élevées - Considérer réduire la mise ou sélectionner moins de prédictions")
        elif total_odds < 3:
            advice.append("💡 Cotes faibles - Coupon sécurisé mais gain limité")
        else:
            advice.append("✅ Cotes équilibrées pour un bon rapport risque/récompense")
        
        # Conseil sur la confiance
        avg_confidence = metrics.get('average_confidence', 50)
        if avg_confidence >= 80:
            advice.append("🎯 Confiance élevée - Coupon recommandé")
        elif avg_confidence >= 70:
            advice.append("👍 Confiance correcte - Coupon acceptable")
        else:
            advice.append("⚠️ Confiance modérée - Mise prudente recommandée")
        
        # Conseil Kelly
        kelly_stake = metrics.get('kelly_total_stake', 0)
        if kelly_stake > 50:
            advice.append("💰 Forte valeur détectée - Critère de Kelly suggère mise importante")
        elif kelly_stake < 10:
            advice.append("🔍 Valeur limitée - Considérer comme pari de divertissement")
        
        # Conseil diversification
        correlation = metrics.get('average_correlation', 0)
        if correlation > 0.5:
            advice.append("🔗 Corrélations élevées entre prédictions - Risque concentré")
        else:
            advice.append("📊 Bonne diversification du portefeuille")
        
        return advice
    
    def recalibrate_coupon_realtime(self, coupon_id: str, 
                                   updated_data: Dict) -> Optional[Dict]:
        """Recalibrage temps réel d'un coupon existant"""
        
        # Recherche du coupon dans l'historique
        original_coupon = None
        for coupon in self.coupon_history:
            if coupon['coupon_id'] == coupon_id:
                original_coupon = coupon
                break
        
        if not original_coupon:
            return None
        
        print(f"Recalibrage coupon {coupon_id}...")
        
        # Facteurs de recalibrage
        recalibration_factors = []
        
        # Changements de compositions
        if 'lineup_changes' in updated_data:
            changes = updated_data['lineup_changes']
            impact = len(changes) * 0.02  # 2% par changement
            recalibration_factors.append(('lineup_changes', -impact))
        
        # Changements météo
        if 'weather_update' in updated_data:
            weather = updated_data['weather_update']
            if weather.get('rain_probability', 0) > 70:
                recalibration_factors.append(('weather_rain', -0.05))
            if weather.get('wind_speed', 0) > 20:
                recalibration_factors.append(('weather_wind', -0.03))
        
        # Nouvelles blessures
        if 'injury_updates' in updated_data:
            injuries = updated_data['injury_updates']
            impact = len(injuries) * 0.03
            recalibration_factors.append(('injuries', -impact))
        
        # Application des recalibrations
        updated_predictions = []
        for pred_dict in original_coupon['predictions']:
            updated_pred = pred_dict.copy()
            
            # Application des facteurs
            confidence_adjustment = 0
            for factor_name, factor_value in recalibration_factors:
                confidence_adjustment += factor_value * 100  # Conversion en pourcentage
            
            # Mise à jour confiance
            new_confidence = max(50.0, min(95.0, updated_pred['confidence_score'] + confidence_adjustment))
            updated_pred['confidence_score'] = round(new_confidence, 2)
            
            # Recalcul expected value
            prob_win = new_confidence / 100.0
            updated_pred['expected_value'] = round((prob_win * updated_pred['odds']) - 1.0, 4)
            
            updated_predictions.append(updated_pred)
        
        # Nouveau coupon recalibré
        recalibrated_coupon = original_coupon.copy()
        recalibrated_coupon.update({
            'coupon_id': f"{original_coupon['coupon_id']}_RECAL_{datetime.now().strftime('%H%M%S')}",
            'recalibration_timestamp': datetime.now().isoformat(),
            'original_coupon_id': original_coupon['coupon_id'],
            'recalibration_factors': recalibration_factors,
            'predictions': updated_predictions,
            'status': 'recalibrated'
        })
        
        # Recalcul métriques globales
        # (Simplification - devrait recalculer avec objects BettingPrediction)
        avg_confidence = np.mean([pred['confidence_score'] for pred in updated_predictions])
        total_ev = sum([pred['expected_value'] for pred in updated_predictions])
        
        recalibrated_coupon['portfolio_metrics'].update({
            'average_confidence': round(avg_confidence, 2),
            'total_expected_value': round(total_ev, 4)
        })
        
        # Sauvegarde
        self.coupon_history.append(recalibrated_coupon)
        
        return recalibrated_coupon
    
    def get_coupon_performance_analysis(self, coupon_id: str, actual_results: Dict) -> Dict:
        """Analyse de performance d'un coupon après résultats"""
        
        # Recherche du coupon
        coupon = None
        for c in self.coupon_history:
            if c['coupon_id'] == coupon_id:
                coupon = c
                break
        
        if not coupon:
            return {'error': 'Coupon non trouvé'}
        
        predictions = coupon['predictions']
        analysis = {
            'coupon_id': coupon_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'predictions_analyzed': len(predictions),
            'results': []
        }
        
        correct_predictions = 0
        total_stake = 100.0  # Mise fictive de 100€
        
        for pred in predictions:
            pred_type = pred['prediction_type']
            predicted_value = pred['prediction_value']
            actual_value = actual_results.get(pred_type)
            
            is_correct = (predicted_value == actual_value)
            if is_correct:
                correct_predictions += 1
            
            result = {
                'prediction_type': pred_type,
                'predicted': predicted_value,
                'actual': actual_value,
                'correct': is_correct,
                'confidence_was': pred['confidence_score'],
                'odds': pred['odds']
            }
            
            analysis['results'].append(result)
        
        # Calcul performance globale
        accuracy = correct_predictions / len(predictions) if predictions else 0.0
        coupon_won = (correct_predictions == len(predictions))
        
        if coupon_won:
            total_return = total_stake * coupon['portfolio_metrics']['total_odds']
            profit = total_return - total_stake
        else:
            total_return = 0.0
            profit = -total_stake
        
        analysis.update({
            'accuracy': round(accuracy * 100, 1),
            'correct_predictions': correct_predictions,
            'coupon_won': coupon_won,
            'profit_loss': round(profit, 2),
            'roi_percentage': round((profit / total_stake) * 100, 1),
            'confidence_validation': self._validate_confidence_accuracy(predictions, analysis['results'])
        })
        
        return analysis
    
    def _validate_confidence_accuracy(self, predictions: List[Dict], results: List[Dict]) -> Dict:
        """Valide la précision des scores de confiance"""
        
        confidence_ranges = {'60-70': [], '70-80': [], '80-90': [], '90-100': []}
        
        for pred, result in zip(predictions, results):
            confidence = pred['confidence_score']
            is_correct = result['correct']
            
            if 60 <= confidence < 70:
                confidence_ranges['60-70'].append(is_correct)
            elif 70 <= confidence < 80:
                confidence_ranges['70-80'].append(is_correct)
            elif 80 <= confidence < 90:
                confidence_ranges['80-90'].append(is_correct)
            else:
                confidence_ranges['90-100'].append(is_correct)
        
        validation = {}
        for range_name, results_list in confidence_ranges.items():
            if results_list:
                actual_accuracy = sum(results_list) / len(results_list) * 100
                expected_accuracy = (int(range_name.split('-')[0]) + int(range_name.split('-')[1])) / 2
                calibration_error = abs(actual_accuracy - expected_accuracy)
                
                validation[range_name] = {
                    'predictions_count': len(results_list),
                    'actual_accuracy': round(actual_accuracy, 1),
                    'expected_accuracy': expected_accuracy,
                    'calibration_error': round(calibration_error, 1)
                }
        
        return validation
    
    def save_coupon_system(self, filepath: str):
        """Sauvegarde le système de coupons"""
        
        system_data = {
            'coupon_history': self.coupon_history,
            'default_config': self.default_config,
            'system_stats': {
                'total_coupons_generated': len(self.coupon_history),
                'last_update': datetime.now().isoformat()
            }
        }
        
        with open(f"{filepath}_coupon_system.json", 'w') as f:
            json.dump(system_data, f, indent=2, default=str)
        
        print(f"Système de coupons sauvegardé: {filepath}")

def test_intelligent_coupon_system():
    """Test complet du système de coupons intelligents"""
    
    print("=== TEST SYSTÈME COUPON INTELLIGENT ===")
    
    # Initialisation
    coupon_system = IntelligentBettingCoupon()
    coupon_system.initialize_components()
    
    # Données de matchs simulées
    matches_data = [
        {
            'home_team': 'Manchester United',
            'away_team': 'Liverpool',
            'league': 'Premier_League',
            'match_importance': 'high',
            'date': '2025-01-25',
            'home_team_form_points': 12,
            'away_team_form_points': 15
        },
        {
            'home_team': 'Barcelona',
            'away_team': 'Real Madrid',
            'league': 'La_Liga',
            'match_importance': 'high',
            'date': '2025-01-25',
            'home_team_form_points': 14,
            'away_team_form_points': 13
        },
        {
            'home_team': 'Bayern Munich',
            'away_team': 'Borussia Dortmund',
            'league': 'Bundesliga',
            'match_importance': 'normal',
            'date': '2025-01-25',
            'home_team_form_points': 16,
            'away_team_form_points': 11
        }
    ]
    
    print(f"\\n--- Génération Coupon Multi-Matchs ---")
    coupon = coupon_system.generate_intelligent_coupon(matches_data)
    
    if coupon['status'] == 'success':
        print(f"✅ Coupon généré: {coupon['coupon_id']}")
        print(f"   Matchs analysés: {coupon['matches_analyzed']}")
        print(f"   Prédictions sélectionnées: {coupon['predictions_selected']}")
        print(f"   Cotes totales: {coupon['portfolio_metrics']['total_odds']}")
        print(f"   Confiance moyenne: {coupon['portfolio_metrics']['average_confidence']}%")
        print(f"   Expected Value: {coupon['portfolio_metrics']['total_expected_value']}")
        
        print(f"\\n   Top 3 Prédictions:")
        for i, pred in enumerate(coupon['predictions'][:3]):
            print(f"     {i+1}. {pred['prediction_type']}: {pred['prediction_value']}")
            print(f"        Confiance: {pred['confidence_score']}% | Cotes: {pred['odds']} | Risque: {pred['risk_category']}")
        
        print(f"\\n   Conseils de mise:")
        for advice in coupon['betting_advice'][:3]:
            print(f"     • {advice}")
        
        # Test recalibrage temps réel
        print(f"\\n--- Test Recalibrage Temps Réel ---")
        
        recalibration_data = {
            'lineup_changes': ['Player X injured', 'Player Y suspended'],
            'weather_update': {'rain_probability': 80, 'wind_speed': 25},
            'injury_updates': ['Key striker out']
        }
        
        recalibrated = coupon_system.recalibrate_coupon_realtime(
            coupon['coupon_id'], recalibration_data
        )
        
        if recalibrated:
            print(f"✅ Coupon recalibré: {recalibrated['coupon_id']}")
            print(f"   Facteurs de recalibrage: {len(recalibrated['recalibration_factors'])}")
            print(f"   Nouvelle confiance moyenne: {recalibrated['portfolio_metrics']['average_confidence']}%")
        
        # Test analyse performance
        print(f"\\n--- Test Analyse Performance ---")
        
        # Résultats simulés
        mock_results = {}
        for pred in coupon['predictions']:
            # 70% de chance que la prédiction soit correcte (simulation)
            if np.random.random() < 0.7:
                mock_results[pred['prediction_type']] = pred['prediction_value']
            else:
                # Résultat différent
                mock_results[pred['prediction_type']] = f"not_{pred['prediction_value']}"
        
        performance = coupon_system.get_coupon_performance_analysis(
            coupon['coupon_id'], mock_results
        )
        
        print(f"✅ Analyse performance terminée:")
        print(f"   Précision: {performance['accuracy']}%")
        print(f"   Prédictions correctes: {performance['correct_predictions']}/{performance['predictions_analyzed']}")
        print(f"   Coupon gagné: {'Oui' if performance['coupon_won'] else 'Non'}")
        print(f"   ROI: {performance['roi_percentage']}%")
        
        # Statistiques système
        print(f"\\n--- Statistiques Système ---")
        print(f"   Coupons générés: {len(coupon_system.coupon_history)}")
        print(f"   Cache prédictions: {len(coupon_system.prediction_cache)}")
        
    else:
        print(f"❌ Échec génération coupon: {coupon.get('message', 'Erreur inconnue')}")
    
    print("\\n=== TEST TERMINÉ ===")

if __name__ == "__main__":
    test_intelligent_coupon_system()