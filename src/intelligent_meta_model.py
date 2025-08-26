"""
🎯 INTELLIGENT META-MODEL - SÉLECTION AUTOMATIQUE DE MODÈLES
Système intelligent pour sélectionner les meilleurs modèles selon le contexte

Version: 2.0 - Phase 2 ML Transformation  
Créé: 23 août 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

class ModelPerformanceTracker:
    """Suivi des performances historiques par modèle et contexte"""
    
    def __init__(self):
        self.performance_history = {}
        self.context_mapping = {}
        
    def record_performance(self, 
                          model_name: str,
                          context: Dict,
                          prediction: float,
                          actual: float,
                          confidence: float = 0.0):
        """Enregistre une performance de modèle"""
        
        context_key = self._create_context_key(context)
        
        if model_name not in self.performance_history:
            self.performance_history[model_name] = {}
        
        if context_key not in self.performance_history[model_name]:
            self.performance_history[model_name][context_key] = {
                'predictions': [],
                'actuals': [],
                'confidences': [],
                'errors': [],
                'total_predictions': 0,
                'correct_predictions': 0,
                'context_info': context
            }
        
        entry = self.performance_history[model_name][context_key]
        error = abs(prediction - actual)
        
        entry['predictions'].append(prediction)
        entry['actuals'].append(actual)
        entry['confidences'].append(confidence)
        entry['errors'].append(error)
        entry['total_predictions'] += 1
        
        # Pour classification (résultat match)
        if context.get('prediction_type') == 'match_result':
            if round(prediction) == round(actual):
                entry['correct_predictions'] += 1
    
    def _create_context_key(self, context: Dict) -> str:
        """Crée une clé unique pour le contexte"""
        key_elements = [
            context.get('league', 'unknown'),
            context.get('prediction_type', 'unknown'),
            context.get('season_period', 'unknown'),
            context.get('match_importance', 'normal')
        ]
        return "_".join(key_elements)
    
    def get_model_performance(self, 
                            model_name: str, 
                            context: Dict = None,
                            min_predictions: int = 10) -> Dict:
        """Récupère les performances d'un modèle"""
        
        if model_name not in self.performance_history:
            return {'accuracy': 0.0, 'mae': float('inf'), 'confidence': 0.0, 'predictions_count': 0}
        
        if context:
            context_key = self._create_context_key(context)
            if context_key not in self.performance_history[model_name]:
                return {'accuracy': 0.0, 'mae': float('inf'), 'confidence': 0.0, 'predictions_count': 0}
            
            data = self.performance_history[model_name][context_key]
        else:
            # Agrégation toutes contextes
            all_data = {
                'predictions': [],
                'actuals': [],
                'confidences': [],
                'errors': [],
                'total_predictions': 0,
                'correct_predictions': 0
            }
            
            for context_data in self.performance_history[model_name].values():
                all_data['predictions'].extend(context_data['predictions'])
                all_data['actuals'].extend(context_data['actuals'])
                all_data['confidences'].extend(context_data['confidences'])
                all_data['errors'].extend(context_data['errors'])
                all_data['total_predictions'] += context_data['total_predictions']
                all_data['correct_predictions'] += context_data['correct_predictions']
            
            data = all_data
        
        if data['total_predictions'] < min_predictions:
            return {'accuracy': 0.0, 'mae': float('inf'), 'confidence': 0.0, 'predictions_count': data['total_predictions']}
        
        # Calculs métriques
        accuracy = data['correct_predictions'] / data['total_predictions'] if data['total_predictions'] > 0 else 0.0
        mae = np.mean(data['errors']) if data['errors'] else float('inf')
        avg_confidence = np.mean(data['confidences']) if data['confidences'] else 0.0
        
        return {
            'accuracy': accuracy,
            'mae': mae,
            'confidence': avg_confidence,
            'predictions_count': data['total_predictions'],
            'recent_performance': self._get_recent_performance(data)
        }
    
    def _get_recent_performance(self, data: Dict, recent_count: int = 20) -> float:
        """Performance sur les N dernières prédictions"""
        if len(data['errors']) < recent_count:
            return np.mean(data['errors']) if data['errors'] else float('inf')
        
        recent_errors = data['errors'][-recent_count:]
        return np.mean(recent_errors)

class ContextAnalyzer:
    """Analyse le contexte des matchs pour la sélection de modèles"""
    
    def __init__(self):
        self.league_characteristics = {
            'Premier_League': {'physicality': 0.9, 'pace': 0.95, 'tactical_complexity': 0.8},
            'La_Liga': {'physicality': 0.7, 'pace': 0.8, 'tactical_complexity': 0.95},
            'Bundesliga': {'physicality': 0.8, 'pace': 0.9, 'tactical_complexity': 0.85},
            'Serie_A': {'physicality': 0.75, 'pace': 0.75, 'tactical_complexity': 0.9},
            'Ligue_1': {'physicality': 0.7, 'pace': 0.85, 'tactical_complexity': 0.8},
            'Champions_League': {'physicality': 0.85, 'pace': 0.9, 'tactical_complexity': 0.95}
        }
    
    def analyze_match_context(self, match_data: Dict) -> Dict:
        """Analyse complète du contexte d'un match"""
        
        context = {
            'league': match_data.get('league', 'unknown'),
            'prediction_type': match_data.get('prediction_type', 'unknown'),
            'match_importance': self._assess_match_importance(match_data),
            'season_period': self._get_season_period(match_data),
            'team_form_difference': self._calculate_form_difference(match_data),
            'historical_h2h': match_data.get('h2h_matches_count', 0),
            'data_quality': self._assess_data_quality(match_data),
            'uncertainty_factors': self._identify_uncertainty_factors(match_data)
        }
        
        # Ajout caractéristiques ligue
        league_chars = self.league_characteristics.get(context['league'], {})
        context.update(league_chars)
        
        return context
    
    def _assess_match_importance(self, match_data: Dict) -> str:
        """Évalue l'importance du match"""
        
        # Championnat européen
        if 'Champions' in match_data.get('league', '') or 'Europa' in match_data.get('league', ''):
            return 'high'
        
        # Fin de saison avec enjeux
        home_position = match_data.get('home_team_position', 10)
        away_position = match_data.get('away_team_position', 10)
        
        if home_position <= 6 or away_position <= 6:  # Top 6
            return 'high'
        elif home_position >= 17 or away_position >= 17:  # Lutte relégation
            return 'high'
        
        return 'normal'
    
    def _get_season_period(self, match_data: Dict) -> str:
        """Détermine la période de la saison"""
        match_date = match_data.get('date', '')
        
        if not match_date:
            return 'unknown'
        
        try:
            month = int(match_date.split('-')[1])
            
            if month in [8, 9, 10]:
                return 'early_season'
            elif month in [11, 12, 1, 2]:
                return 'mid_season'
            elif month in [3, 4, 5]:
                return 'late_season'
            else:
                return 'off_season'
        except:
            return 'unknown'
    
    def _calculate_form_difference(self, match_data: Dict) -> float:
        """Calcule la différence de forme entre équipes"""
        home_form = match_data.get('home_team_form_points', 0)
        away_form = match_data.get('away_team_form_points', 0)
        
        if home_form > 0 and away_form > 0:
            return abs(home_form - away_form) / max(home_form, away_form)
        
        return 0.0
    
    def _assess_data_quality(self, match_data: Dict) -> float:
        """Évalue la qualité des données disponibles"""
        
        required_fields = [
            'home_team_form', 'away_team_form', 'h2h_matches_count',
            'home_team_goals_avg', 'away_team_goals_avg'
        ]
        
        available_fields = sum(1 for field in required_fields if field in match_data and match_data[field] is not None)
        data_quality = available_fields / len(required_fields)
        
        return data_quality
    
    def _identify_uncertainty_factors(self, match_data: Dict) -> List[str]:
        """Identifie les facteurs d'incertitude"""
        
        factors = []
        
        # Équipes nouvellement promues
        if match_data.get('home_team_promoted', False) or match_data.get('away_team_promoted', False):
            factors.append('newly_promoted')
        
        # Manque de confrontations historiques
        if match_data.get('h2h_matches_count', 0) < 5:
            factors.append('limited_h2h')
        
        # Grandes différences de forme
        if self._calculate_form_difference(match_data) > 0.5:
            factors.append('form_disparity')
        
        # Début de saison
        if self._get_season_period(match_data) == 'early_season':
            factors.append('early_season_unpredictability')
        
        return factors

class ModelSelector:
    """Sélectionneur intelligent de modèles"""
    
    def __init__(self, performance_tracker: ModelPerformanceTracker):
        self.performance_tracker = performance_tracker
        self.context_analyzer = ContextAnalyzer()
        self.selection_strategy = {}
        
        # Règles de sélection par défaut
        self._initialize_selection_rules()
    
    def _initialize_selection_rules(self):
        """Initialise les règles de sélection par défaut"""
        
        self.selection_strategy = {
            'match_result': {
                'preferred_models': ['xgb_match_result', 'rf_match_result', 'transformer'],
                'context_weights': {
                    'high_importance': {'xgb': 0.4, 'transformer': 0.4, 'rf': 0.2},
                    'normal_importance': {'xgb': 0.5, 'rf': 0.3, 'lstm': 0.2}
                }
            },
            'total_goals': {
                'preferred_models': ['xgb_total_goals', 'lstm', 'nn_total_goals'],
                'context_weights': {
                    'early_season': {'lstm': 0.4, 'xgb': 0.4, 'rf': 0.2},
                    'late_season': {'xgb': 0.5, 'nn': 0.3, 'lstm': 0.2}
                }
            },
            'both_teams_scored': {
                'preferred_models': ['rf_both_teams_scored', 'xgb_both_teams_scored', 'cnn1d'],
                'context_weights': {
                    'Premier_League': {'rf': 0.4, 'cnn1d': 0.4, 'xgb': 0.2},
                    'La_Liga': {'xgb': 0.5, 'rf': 0.3, 'cnn1d': 0.2}
                }
            }
        }
    
    def select_best_models(self, 
                          prediction_type: str,
                          match_context: Dict,
                          available_models: List[str],
                          top_k: int = 3) -> List[Tuple[str, float]]:
        """Sélectionne les meilleurs modèles pour le contexte donné"""
        
        context = self.context_analyzer.analyze_match_context(match_context)
        model_scores = {}
        
        for model_name in available_models:
            if prediction_type not in model_name and prediction_type != 'all':
                continue
            
            score = self._calculate_model_score(model_name, context, prediction_type)
            model_scores[model_name] = score
        
        # Trier par score décroissant
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_models[:top_k]
    
    def _calculate_model_score(self, model_name: str, context: Dict, prediction_type: str) -> float:
        """Calcule le score d'un modèle pour le contexte"""
        
        # Performance historique
        perf = self.performance_tracker.get_model_performance(model_name, context)
        historical_score = 0.0
        
        if perf['predictions_count'] >= 10:
            # Score basé sur accuracy et MAE
            accuracy_score = perf['accuracy']
            mae_score = max(0, 1.0 - perf['mae']) if perf['mae'] != float('inf') else 0.0
            confidence_score = perf['confidence']
            
            historical_score = (accuracy_score * 0.4 + mae_score * 0.4 + confidence_score * 0.2)
        
        # Score contextuel basé sur les règles
        contextual_score = self._get_contextual_score(model_name, context, prediction_type)
        
        # Score de qualité des données
        data_quality_score = context.get('data_quality', 0.5)
        
        # Score final pondéré
        final_score = (
            historical_score * 0.5 +
            contextual_score * 0.3 +
            data_quality_score * 0.2
        )
        
        # Pénalité pour facteurs d'incertitude
        uncertainty_penalty = len(context.get('uncertainty_factors', [])) * 0.05
        final_score = max(0.0, final_score - uncertainty_penalty)
        
        return final_score
    
    def _get_contextual_score(self, model_name: str, context: Dict, prediction_type: str) -> float:
        """Score contextuel basé sur les règles définies"""
        
        if prediction_type not in self.selection_strategy:
            return 0.5  # Score neutre
        
        strategy = self.selection_strategy[prediction_type]
        
        # Bonus si modèle dans les préférés
        base_score = 0.7 if any(pref in model_name for pref in strategy['preferred_models']) else 0.3
        
        # Ajustement selon contexte spécifique
        context_weights = strategy.get('context_weights', {})
        
        for context_key, weights in context_weights.items():
            if context_key in context.get('league', '') or context_key == context.get('match_importance', ''):
                for model_type, weight in weights.items():
                    if model_type in model_name.lower():
                        base_score *= (1.0 + weight)
                        break
        
        return min(1.0, base_score)

class IntelligentMetaModel:
    """Meta-modèle intelligent pour sélection automatique"""
    
    def __init__(self):
        self.performance_tracker = ModelPerformanceTracker()
        self.model_selector = ModelSelector(self.performance_tracker)
        self.ensemble_weights = {}
        self.selection_history = []
        
        # Meta-modèle pour apprendre les patterns de sélection
        self.meta_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.meta_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.is_meta_trained = False
    
    def record_prediction_result(self, 
                               model_name: str,
                               context: Dict,
                               prediction: float,
                               actual: float,
                               confidence: float = 0.0):
        """Enregistre le résultat d'une prédiction pour apprentissage"""
        
        self.performance_tracker.record_performance(
            model_name, context, prediction, actual, confidence
        )
        
        # Historique pour meta-apprentissage
        self.selection_history.append({
            'model_name': model_name,
            'context': context,
            'prediction': prediction,
            'actual': actual,
            'confidence': confidence,
            'error': abs(prediction - actual)
        })
    
    def get_optimal_model_ensemble(self, 
                                 prediction_type: str,
                                 match_data: Dict,
                                 available_models: List[str],
                                 ensemble_size: int = 3) -> Dict:
        """Obtient l'ensemble optimal de modèles pour le contexte"""
        
        context = self.model_selector.context_analyzer.analyze_match_context(match_data)
        
        # Sélection des meilleurs modèles
        selected_models = self.model_selector.select_best_models(
            prediction_type, match_data, available_models, ensemble_size
        )
        
        if not selected_models:
            return {'models': [], 'weights': {}, 'confidence': 0.0}
        
        # Calcul des poids d'ensemble
        total_score = sum(score for _, score in selected_models)
        weights = {}
        
        for model_name, score in selected_models:
            weights[model_name] = score / total_score if total_score > 0 else 1.0 / len(selected_models)
        
        # Calcul de la confiance globale
        avg_confidence = self._calculate_ensemble_confidence(selected_models, context)
        
        return {
            'models': [model for model, _ in selected_models],
            'weights': weights,
            'confidence': avg_confidence,
            'context': context,
            'selection_reasoning': self._generate_selection_reasoning(selected_models, context)
        }
    
    def _calculate_ensemble_confidence(self, selected_models: List[Tuple[str, float]], context: Dict) -> float:
        """Calcule la confiance de l'ensemble"""
        
        if not selected_models:
            return 0.0
        
        # Confiance basée sur les scores et performances historiques
        confidence_scores = []
        
        for model_name, score in selected_models:
            perf = self.performance_tracker.get_model_performance(model_name, context)
            model_confidence = perf.get('confidence', 0.0)
            combined_confidence = (score + model_confidence) / 2.0
            confidence_scores.append(combined_confidence)
        
        # Moyenne pondérée
        weights = [score for _, score in selected_models]
        total_weight = sum(weights)
        
        if total_weight > 0:
            weighted_confidence = sum(conf * weight for conf, weight in zip(confidence_scores, weights)) / total_weight
        else:
            weighted_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Ajustement selon qualité des données
        data_quality = context.get('data_quality', 0.5)
        final_confidence = weighted_confidence * data_quality
        
        return min(1.0, final_confidence)
    
    def _generate_selection_reasoning(self, selected_models: List[Tuple[str, float]], context: Dict) -> str:
        """Génère l'explication du choix des modèles"""
        
        reasoning_parts = []
        
        # Contexte du match
        reasoning_parts.append(f"Match {context['league']} - {context['match_importance']} importance")
        reasoning_parts.append(f"Période: {context['season_period']}")
        
        # Modèles sélectionnés
        for i, (model_name, score) in enumerate(selected_models):
            reasoning_parts.append(f"{i+1}. {model_name} (score: {score:.3f})")
        
        # Facteurs d'incertitude
        if context.get('uncertainty_factors'):
            reasoning_parts.append(f"Facteurs d'incertitude: {', '.join(context['uncertainty_factors'])}")
        
        return " | ".join(reasoning_parts)
    
    def train_meta_model(self, min_samples: int = 100):
        """Entraîne le meta-modèle sur l'historique"""
        
        if len(self.selection_history) < min_samples:
            print(f"Pas assez d'échantillons pour meta-apprentissage ({len(self.selection_history)}<{min_samples})")
            return False
        
        # Préparation des données pour meta-apprentissage
        X_meta, y_classification, y_regression = self._prepare_meta_training_data()
        
        if len(X_meta) == 0:
            return False
        
        # Entraînement classificateur (sélection modèle optimal)
        self.meta_classifier.fit(X_meta, y_classification)
        
        # Entraînement régresseur (prédiction performance)
        self.meta_regressor.fit(X_meta, y_regression)
        
        self.is_meta_trained = True
        
        print(f"Meta-modèle entraîné sur {len(X_meta)} échantillons")
        return True
    
    def _prepare_meta_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prépare les données pour l'entraînement du meta-modèle"""
        
        X_meta = []
        y_classification = []  # Meilleur modèle pour le contexte
        y_regression = []      # Performance attendue
        
        # Grouper par contexte similaire
        context_groups = {}
        
        for entry in self.selection_history:
            context_key = self._create_context_key(entry['context'])
            
            if context_key not in context_groups:
                context_groups[context_key] = []
            
            context_groups[context_key].append(entry)
        
        # Pour chaque groupe, identifier le meilleur modèle
        for context_key, entries in context_groups.items():
            if len(entries) < 5:  # Pas assez d'échantillons
                continue
            
            # Features du contexte
            context_features = self._extract_context_features(entries[0]['context'])
            
            # Trouver le modèle avec la meilleure performance moyenne
            model_performances = {}
            for entry in entries:
                model = entry['model_name']
                if model not in model_performances:
                    model_performances[model] = []
                model_performances[model].append(entry['error'])
            
            # Modèle avec la plus petite erreur moyenne
            best_model = min(model_performances.items(), key=lambda x: np.mean(x[1]))[0]
            best_performance = np.mean(model_performances[best_model])
            
            X_meta.append(context_features)
            y_classification.append(best_model)
            y_regression.append(1.0 / (1.0 + best_performance))  # Score inversé de l'erreur
        
        return np.array(X_meta), np.array(y_classification), np.array(y_regression)
    
    def _extract_context_features(self, context: Dict) -> List[float]:
        """Extrait les features numériques du contexte"""
        
        features = [
            # Encodage de la ligue
            1.0 if context.get('league') == 'Premier_League' else 0.0,
            1.0 if context.get('league') == 'La_Liga' else 0.0,
            1.0 if context.get('league') == 'Bundesliga' else 0.0,
            
            # Importance du match
            1.0 if context.get('match_importance') == 'high' else 0.0,
            
            # Période de la saison
            1.0 if context.get('season_period') == 'early_season' else 0.0,
            1.0 if context.get('season_period') == 'mid_season' else 0.0,
            1.0 if context.get('season_period') == 'late_season' else 0.0,
            
            # Métriques numériques
            context.get('team_form_difference', 0.0),
            context.get('historical_h2h', 0.0) / 20.0,  # Normalisé
            context.get('data_quality', 0.5),
            len(context.get('uncertainty_factors', [])) / 5.0  # Normalisé
        ]
        
        return features
    
    def _create_context_key(self, context: Dict) -> str:
        """Crée une clé de contexte pour groupement"""
        return f"{context.get('league', 'unknown')}_{context.get('prediction_type', 'unknown')}_{context.get('match_importance', 'normal')}"
    
    def predict_best_model(self, context: Dict) -> Tuple[str, float]:
        """Prédit le meilleur modèle avec le meta-modèle"""
        
        if not self.is_meta_trained:
            return 'xgb_default', 0.5
        
        context_features = np.array([self._extract_context_features(context)])
        
        # Prédiction du meilleur modèle
        best_model = self.meta_classifier.predict(context_features)[0]
        
        # Prédiction de la performance attendue
        expected_performance = self.meta_regressor.predict(context_features)[0]
        
        return best_model, expected_performance
    
    def get_performance_summary(self) -> Dict:
        """Résumé des performances du meta-modèle"""
        
        summary = {
            'total_predictions': len(self.selection_history),
            'models_tracked': len(set(entry['model_name'] for entry in self.selection_history)),
            'avg_error': np.mean([entry['error'] for entry in self.selection_history]) if self.selection_history else 0.0,
            'meta_model_trained': self.is_meta_trained
        }
        
        if self.selection_history:
            # Performance par modèle
            model_stats = {}
            for entry in self.selection_history:
                model = entry['model_name']
                if model not in model_stats:
                    model_stats[model] = {'errors': [], 'predictions': 0}
                
                model_stats[model]['errors'].append(entry['error'])
                model_stats[model]['predictions'] += 1
            
            # Top modèles
            for model, stats in model_stats.items():
                stats['avg_error'] = np.mean(stats['errors'])
            
            best_models = sorted(model_stats.items(), key=lambda x: x[1]['avg_error'])[:5]
            summary['best_models'] = [(model, stats['avg_error']) for model, stats in best_models]
        
        return summary
    
    def save_meta_model(self, filepath: str):
        """Sauvegarde le meta-modèle"""
        meta_data = {
            'performance_tracker': self.performance_tracker.performance_history,
            'selection_history': self.selection_history,
            'ensemble_weights': self.ensemble_weights
        }
        
        # Sauvegarde des modèles sklearn
        if self.is_meta_trained:
            joblib.dump(self.meta_classifier, f"{filepath}_classifier.joblib")
            joblib.dump(self.meta_regressor, f"{filepath}_regressor.joblib")
        
        # Sauvegarde des données JSON
        with open(f"{filepath}_data.json", 'w') as f:
            json.dump(meta_data, f, indent=2)
        
        print(f"Meta-modèle sauvegardé: {filepath}")
    
    def load_meta_model(self, filepath: str):
        """Charge le meta-modèle"""
        try:
            # Chargement des données
            with open(f"{filepath}_data.json", 'r') as f:
                meta_data = json.load(f)
            
            self.performance_tracker.performance_history = meta_data['performance_tracker']
            self.selection_history = meta_data['selection_history']
            self.ensemble_weights = meta_data['ensemble_weights']
            
            # Chargement des modèles sklearn
            import os
            if os.path.exists(f"{filepath}_classifier.joblib"):
                self.meta_classifier = joblib.load(f"{filepath}_classifier.joblib")
                self.meta_regressor = joblib.load(f"{filepath}_regressor.joblib")
                self.is_meta_trained = True
            
            print(f"Meta-modèle chargé: {filepath}")
            return True
            
        except Exception as e:
            print(f"Erreur chargement meta-modèle: {str(e)}")
            return False

def test_meta_model():
    """Test du meta-modèle intelligent"""
    print("=== TEST META-MODELE INTELLIGENT ===")
    
    meta_model = IntelligentMetaModel()
    
    # Simulation de données historiques
    leagues = ['Premier_League', 'La_Liga', 'Bundesliga']
    prediction_types = ['match_result', 'total_goals', 'both_teams_scored']
    models = ['xgb_match_result', 'rf_match_result', 'nn_match_result', 'lstm', 'transformer']
    
    print("Simulation de données historiques...")
    
    # Génération de 200 prédictions simulées
    for i in range(200):
        context = {
            'league': np.random.choice(leagues),
            'prediction_type': np.random.choice(prediction_types),
            'match_importance': np.random.choice(['normal', 'high']),
            'date': '2025-01-15',
            'home_team_form_points': np.random.randint(0, 15),
            'away_team_form_points': np.random.randint(0, 15)
        }
        
        model_name = np.random.choice(models)
        prediction = np.random.uniform(0, 3)
        actual = np.random.uniform(0, 3)
        confidence = np.random.uniform(0.5, 0.95)
        
        meta_model.record_prediction_result(model_name, context, prediction, actual, confidence)
    
    print(f"Enregistré {len(meta_model.selection_history)} prédictions")
    
    # Test sélection de modèles
    test_match = {
        'league': 'Premier_League',
        'prediction_type': 'match_result',
        'match_importance': 'high',
        'date': '2025-01-20',
        'home_team_form_points': 12,
        'away_team_form_points': 8
    }
    
    print(f"\n--- Test sélection modèles ---")
    ensemble = meta_model.get_optimal_model_ensemble(
        'match_result', test_match, models, ensemble_size=3
    )
    
    print(f"Modèles sélectionnés: {ensemble['models']}")
    print(f"Poids: {ensemble['weights']}")
    print(f"Confiance: {ensemble['confidence']:.3f}")
    print(f"Raisonnement: {ensemble['selection_reasoning']}")
    
    # Entraînement meta-modèle
    print(f"\n--- Entraînement meta-modèle ---")
    success = meta_model.train_meta_model(min_samples=50)
    
    if success:
        best_model, expected_perf = meta_model.predict_best_model(test_match)
        print(f"Meilleur modèle prédit: {best_model}")
        print(f"Performance attendue: {expected_perf:.3f}")
    
    # Résumé des performances
    summary = meta_model.get_performance_summary()
    print(f"\n--- Résumé performances ---")
    print(f"Total prédictions: {summary['total_predictions']}")
    print(f"Modèles suivis: {summary['models_tracked']}")
    print(f"Erreur moyenne: {summary['avg_error']:.4f}")
    
    if 'best_models' in summary:
        print("Top 3 modèles:")
        for i, (model, error) in enumerate(summary['best_models'][:3]):
            print(f"  {i+1}. {model}: {error:.4f}")
    
    print("\n=== TEST TERMINE ===")

if __name__ == "__main__":
    test_meta_model()