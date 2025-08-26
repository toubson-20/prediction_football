"""
🎯 CONFIDENCE SCORING ENGINE - SYSTÈME DE NOTATION CONFIANCE 0-100%
Calcul précis des scores de confiance avec calibration automatique

Version: 3.0 - Phase 3 ML Transformation
Créé: 23 août 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ConfidenceFeatureExtractor:
    """Extracteur de features pour le scoring de confiance"""
    
    def __init__(self):
        self.feature_importance = {}
        self.feature_history = []
    
    def extract_model_features(self, prediction_data: Dict) -> Dict[str, float]:
        """Extrait les features liées au modèle de prédiction"""
        
        features = {}
        
        # Features du modèle utilisé
        model_name = prediction_data.get('model_used', 'unknown')
        
        # Historique de performance du modèle
        features['model_historical_accuracy'] = self._get_model_accuracy(model_name)
        features['model_prediction_count'] = self._get_model_usage_count(model_name)
        
        # Consensus entre modèles (si ensemble)
        if '+' in model_name:  # Ensemble de modèles
            features['model_consensus'] = prediction_data.get('consensus_score', 0.5)
            features['model_ensemble_size'] = len(model_name.split('+'))
        else:
            features['model_consensus'] = 1.0
            features['model_ensemble_size'] = 1.0
        
        # Incertitude du modèle
        features['prediction_uncertainty'] = prediction_data.get('prediction_uncertainty', 0.3)
        
        return features
    
    def extract_data_quality_features(self, match_data: Dict) -> Dict[str, float]:
        """Extrait les features de qualité des données"""
        
        features = {}
        
        # Complétude des données
        required_fields = ['home_team_form', 'away_team_form', 'h2h_history', 'league', 'date']
        available_fields = sum(1 for field in required_fields if match_data.get(field) is not None)
        features['data_completeness'] = available_fields / len(required_fields)
        
        # Fraîcheur des données
        match_date = match_data.get('date', '2025-01-01')
        try:
            days_since_data = (pd.Timestamp.now() - pd.to_datetime(match_date)).days
            features['data_freshness'] = max(0, min(1, (7 - abs(days_since_data)) / 7))
        except:
            features['data_freshness'] = 0.5
        
        # Qualité des données historiques
        h2h_count = match_data.get('h2h_matches_count', 0)
        features['historical_depth'] = min(1.0, h2h_count / 10.0)  # Normalisé sur 10 matchs
        
        # Cohérence des statistiques
        home_goals_avg = match_data.get('home_team_goals_avg', 1.5)
        away_goals_avg = match_data.get('away_team_goals_avg', 1.5)
        features['stats_coherence'] = 1.0 - abs(home_goals_avg - away_goals_avg) / 5.0
        
        return features
    
    def extract_contextual_features(self, match_data: Dict) -> Dict[str, float]:
        """Extrait les features contextuelles du match"""
        
        features = {}
        
        # Importance du match
        importance = match_data.get('match_importance', 'normal')
        features['match_importance'] = {'high': 1.0, 'normal': 0.5, 'low': 0.2}.get(importance, 0.5)
        
        # Période de la saison
        date_str = match_data.get('date', '2025-03-01')
        try:
            match_date = pd.to_datetime(date_str)
            month = match_date.month
            if month in [8, 9]:  # Début saison
                features['season_period'] = 0.7  # Moins prévisible
            elif month in [10, 11, 12, 1, 2]:  # Milieu saison  
                features['season_period'] = 1.0  # Plus prévisible
            elif month in [3, 4, 5]:  # Fin saison
                features['season_period'] = 0.8  # Enjeux variables
            else:
                features['season_period'] = 0.5
        except:
            features['season_period'] = 0.5
        
        # Équilibre des équipes
        home_form = match_data.get('home_team_form_points', 7)
        away_form = match_data.get('away_team_form_points', 7)
        form_difference = abs(home_form - away_form)
        features['team_balance'] = max(0, 1.0 - form_difference / 15.0)
        
        # Facteurs d'incertitude
        uncertainty_factors = match_data.get('uncertainty_factors', [])
        features['uncertainty_penalty'] = max(0, 1.0 - len(uncertainty_factors) * 0.1)
        
        return features
    
    def extract_prediction_type_features(self, prediction_type: str) -> Dict[str, float]:
        """Extrait les features spécifiques au type de prédiction"""
        
        # Difficulté intrinsèque par type de prédiction
        difficulty_mapping = {
            'match_result': 0.7,  # Difficulté moyenne
            'total_goals': 0.6,   # Plus prévisible
            'both_teams_scored': 0.8,
            'over_2_5_goals': 0.7,
            'correct_score': 0.3,  # Très difficile
            'first_half_result': 0.5,
            'corners_total': 0.4,  # Difficile
            'cards_total': 0.4,
            'home_goals': 0.6,
            'away_goals': 0.6,
            'double_chance': 0.8,  # Plus facile
            'handicap_home': 0.6,
        }
        
        features = {
            'prediction_difficulty': difficulty_mapping.get(prediction_type, 0.5),
            'prediction_volatility': self._get_prediction_volatility(prediction_type)
        }
        
        return features
    
    def _get_model_accuracy(self, model_name: str) -> float:
        """Récupère la précision historique du modèle"""
        # Simulation basée sur le type de modèle
        accuracy_mapping = {
            'xgb': 0.75,
            'rf': 0.72,
            'nn': 0.70,
            'lstm': 0.68,
            'transformer': 0.73,
            'ensemble': 0.78
        }
        
        for key, accuracy in accuracy_mapping.items():
            if key in model_name.lower():
                return accuracy + np.random.normal(0, 0.05)  # Variation
        
        return 0.65  # Défaut
    
    def _get_model_usage_count(self, model_name: str) -> float:
        """Nombre normalisé d'utilisations du modèle"""
        return min(1.0, np.random.randint(10, 1000) / 1000.0)
    
    def _get_prediction_volatility(self, prediction_type: str) -> float:
        """Volatilité historique du type de prédiction"""
        volatility_mapping = {
            'match_result': 0.6,
            'total_goals': 0.4,
            'both_teams_scored': 0.5,
            'correct_score': 0.9,
            'corners_total': 0.7,
            'cards_total': 0.8
        }
        
        return volatility_mapping.get(prediction_type, 0.5)

class ConfidenceCalibrator:
    """Calibrateur de confiance basé sur l'historique"""
    
    def __init__(self):
        self.calibration_data = {}
        self.calibration_models = {}
        self.is_calibrated = False
    
    def add_calibration_data(self, predicted_confidence: float, actual_outcome: bool,
                           prediction_type: str, context: Dict = None):
        """Ajoute des données pour la calibration"""
        
        if prediction_type not in self.calibration_data:
            self.calibration_data[prediction_type] = {
                'predicted_confidences': [],
                'actual_outcomes': [],
                'contexts': []
            }
        
        self.calibration_data[prediction_type]['predicted_confidences'].append(predicted_confidence)
        self.calibration_data[prediction_type]['actual_outcomes'].append(int(actual_outcome))
        self.calibration_data[prediction_type]['contexts'].append(context or {})
    
    def calibrate_confidence_models(self, min_samples: int = 50):
        """Calibre les modèles de confiance basés sur l'historique"""
        
        for pred_type, data in self.calibration_data.items():
            if len(data['predicted_confidences']) < min_samples:
                continue
            
            # Préparation des données
            confidences = np.array(data['predicted_confidences'])
            outcomes = np.array(data['actual_outcomes'])
            
            # Calibration par bins
            calibrated_mapping = self._isotonic_calibration(confidences, outcomes)
            self.calibration_models[pred_type] = calibrated_mapping
        
        self.is_calibrated = len(self.calibration_models) > 0
    
    def _isotonic_calibration(self, confidences: np.ndarray, outcomes: np.ndarray) -> Dict:
        """Calibration isotonique des confidences"""
        
        # Division en bins
        bins = np.linspace(0, 100, 11)  # 10 bins de 10%
        calibrated_mapping = {}
        
        for i in range(len(bins)-1):
            bin_start, bin_end = bins[i], bins[i+1]
            
            # Sélection des prédictions dans ce bin
            in_bin = (confidences >= bin_start) & (confidences < bin_end)
            
            if np.sum(in_bin) > 0:
                actual_accuracy = np.mean(outcomes[in_bin])
                calibrated_mapping[f"{bin_start}-{bin_end}"] = {
                    'samples': np.sum(in_bin),
                    'predicted_avg': np.mean(confidences[in_bin]),
                    'actual_accuracy': actual_accuracy,
                    'calibration_error': abs(np.mean(confidences[in_bin]) - actual_accuracy * 100)
                }
        
        return calibrated_mapping
    
    def apply_calibration(self, raw_confidence: float, prediction_type: str) -> float:
        """Applique la calibration à une confiance brute"""
        
        if not self.is_calibrated or prediction_type not in self.calibration_models:
            return raw_confidence  # Pas de calibration disponible
        
        calibration_model = self.calibration_models[prediction_type]
        
        # Recherche du bin approprié
        for bin_range, bin_data in calibration_model.items():
            bin_start, bin_end = map(float, bin_range.split('-'))
            
            if bin_start <= raw_confidence < bin_end:
                # Ajustement basé sur la calibration historique
                calibration_factor = bin_data['actual_accuracy'] * 100 / bin_data['predicted_avg']
                calibrated = raw_confidence * calibration_factor
                
                # Limitation des ajustements extrêmes
                max_adjustment = 15.0  # Maximum 15 points d'ajustement
                adjustment = calibrated - raw_confidence
                adjustment = np.sign(adjustment) * min(abs(adjustment), max_adjustment)
                
                return max(30.0, min(95.0, raw_confidence + adjustment))
        
        return raw_confidence  # Aucun bin trouvé

class AdvancedConfidenceScorer:
    """Système avancé de scoring de confiance"""
    
    def __init__(self):
        self.feature_extractor = ConfidenceFeatureExtractor()
        self.calibrator = ConfidenceCalibrator()
        
        # Poids des différentes composantes
        self.component_weights = {
            'base_model_confidence': 0.35,
            'data_quality_score': 0.25,
            'contextual_adjustment': 0.20,
            'prediction_difficulty': 0.20
        }
        
        # Historique pour amélioration continue
        self.confidence_history = []
    
    def calculate_confidence_score(self, prediction_data: Dict, match_data: Dict) -> Dict:
        """Calcule le score de confiance complet (0-100%)"""
        
        prediction_type = prediction_data.get('prediction_type', 'unknown')
        
        # 1. Confiance de base du modèle
        base_confidence = self._calculate_base_model_confidence(prediction_data)
        
        # 2. Score qualité des données
        data_quality = self._calculate_data_quality_score(match_data)
        
        # 3. Ajustements contextuels
        contextual_score = self._calculate_contextual_adjustments(match_data, prediction_type)
        
        # 4. Difficulté de la prédiction
        difficulty_penalty = self._calculate_difficulty_penalty(prediction_type)
        
        # Calcul du score composite
        raw_score = (
            base_confidence * self.component_weights['base_model_confidence'] +
            data_quality * self.component_weights['data_quality_score'] +
            contextual_score * self.component_weights['contextual_adjustment'] +
            difficulty_penalty * self.component_weights['prediction_difficulty']
        ) * 100
        
        # Application de la calibration si disponible
        calibrated_score = self.calibrator.apply_calibration(raw_score, prediction_type)
        
        # Score final avec limites
        final_score = max(30.0, min(95.0, calibrated_score))
        
        # Détails du calcul
        confidence_breakdown = {
            'final_confidence_score': round(final_score, 2),
            'raw_score': round(raw_score, 2),
            'calibrated_score': round(calibrated_score, 2),
            'components': {
                'base_model_confidence': round(base_confidence * 100, 2),
                'data_quality_score': round(data_quality * 100, 2),
                'contextual_score': round(contextual_score * 100, 2),
                'difficulty_penalty': round(difficulty_penalty * 100, 2)
            },
            'confidence_category': self._categorize_confidence(final_score),
            'reliability_indicators': self._generate_reliability_indicators(prediction_data, match_data)
        }
        
        # Sauvegarde pour historique
        self.confidence_history.append({
            'prediction_type': prediction_type,
            'confidence_score': final_score,
            'breakdown': confidence_breakdown,
            'timestamp': pd.Timestamp.now()
        })
        
        return confidence_breakdown
    
    def _calculate_base_model_confidence(self, prediction_data: Dict) -> float:
        """Calcule la confiance de base du modèle"""
        
        model_features = self.feature_extractor.extract_model_features(prediction_data)
        
        # Combinaison des features du modèle
        base_score = (
            model_features.get('model_historical_accuracy', 0.7) * 0.5 +
            model_features.get('model_consensus', 0.8) * 0.3 +
            (1.0 - model_features.get('prediction_uncertainty', 0.3)) * 0.2
        )
        
        # Bonus pour les ensembles
        if model_features.get('model_ensemble_size', 1) > 1:
            ensemble_bonus = min(0.1, model_features['model_ensemble_size'] * 0.02)
            base_score += ensemble_bonus
        
        return min(1.0, base_score)
    
    def _calculate_data_quality_score(self, match_data: Dict) -> float:
        """Calcule le score de qualité des données"""
        
        quality_features = self.feature_extractor.extract_data_quality_features(match_data)
        
        # Moyenne pondérée des features de qualité
        quality_score = (
            quality_features.get('data_completeness', 0.5) * 0.4 +
            quality_features.get('data_freshness', 0.5) * 0.3 +
            quality_features.get('historical_depth', 0.5) * 0.2 +
            quality_features.get('stats_coherence', 0.5) * 0.1
        )
        
        return quality_score
    
    def _calculate_contextual_adjustments(self, match_data: Dict, prediction_type: str) -> float:
        """Calcule les ajustements contextuels"""
        
        contextual_features = self.feature_extractor.extract_contextual_features(match_data)
        
        # Score contextuel de base
        contextual_score = (
            contextual_features.get('match_importance', 0.5) * 0.3 +
            contextual_features.get('season_period', 0.5) * 0.3 +
            contextual_features.get('team_balance', 0.5) * 0.2 +
            contextual_features.get('uncertainty_penalty', 1.0) * 0.2
        )
        
        return contextual_score
    
    def _calculate_difficulty_penalty(self, prediction_type: str) -> float:
        """Calcule la pénalité de difficulté"""
        
        type_features = self.feature_extractor.extract_prediction_type_features(prediction_type)
        
        # Score inversé de difficulté
        difficulty_score = (
            type_features.get('prediction_difficulty', 0.5) * 0.7 +
            (1.0 - type_features.get('prediction_volatility', 0.5)) * 0.3
        )
        
        return difficulty_score
    
    def _categorize_confidence(self, confidence_score: float) -> str:
        """Catégorise le score de confiance"""
        
        if confidence_score >= 85:
            return "TRES_HAUTE"
        elif confidence_score >= 75:
            return "HAUTE"
        elif confidence_score >= 65:
            return "MOYENNE_HAUTE"
        elif confidence_score >= 55:
            return "MOYENNE"
        elif confidence_score >= 45:
            return "MOYENNE_FAIBLE"
        else:
            return "FAIBLE"
    
    def _generate_reliability_indicators(self, prediction_data: Dict, match_data: Dict) -> Dict:
        """Génère des indicateurs de fiabilité"""
        
        indicators = {}
        
        # Indicateur de consensus
        model_name = prediction_data.get('model_used', '')
        if '+' in model_name:
            indicators['multi_model_consensus'] = True
        else:
            indicators['multi_model_consensus'] = False
        
        # Indicateur de données complètes
        required_fields = ['home_team_form', 'away_team_form', 'h2h_history']
        complete_data = all(match_data.get(field) is not None for field in required_fields)
        indicators['complete_data_available'] = complete_data
        
        # Indicateur de contexte favorable
        match_importance = match_data.get('match_importance', 'normal')
        indicators['favorable_context'] = match_importance != 'high'
        
        # Indicateur de stabilité
        uncertainty_factors = match_data.get('uncertainty_factors', [])
        indicators['stable_conditions'] = len(uncertainty_factors) <= 1
        
        return indicators
    
    def bulk_confidence_calculation(self, predictions_data: List[Dict]) -> List[Dict]:
        """Calcul de confiance en masse pour plusieurs prédictions"""
        
        results = []
        
        for pred_data in predictions_data:
            try:
                confidence_info = self.calculate_confidence_score(
                    pred_data['prediction_data'],
                    pred_data['match_data']
                )
                
                result = {
                    'prediction_id': pred_data.get('prediction_id', 'unknown'),
                    'confidence_info': confidence_info,
                    'status': 'success'
                }
                
            except Exception as e:
                result = {
                    'prediction_id': pred_data.get('prediction_id', 'unknown'),
                    'confidence_info': None,
                    'status': 'failed',
                    'error': str(e)
                }
            
            results.append(result)
        
        return results
    
    def get_confidence_statistics(self) -> Dict:
        """Retourne les statistiques du système de confiance"""
        
        if not self.confidence_history:
            return {'message': 'Aucune donnée disponible'}
        
        df_history = pd.DataFrame(self.confidence_history)
        
        stats = {
            'total_predictions_scored': len(df_history),
            'average_confidence': df_history['confidence_score'].mean(),
            'confidence_distribution': df_history['confidence_score'].describe().to_dict(),
            'confidence_by_type': df_history.groupby('prediction_type')['confidence_score'].agg(['mean', 'count']).to_dict(),
            'calibration_available': self.calibrator.is_calibrated,
            'calibration_data_points': sum(len(data['predicted_confidences']) for data in self.calibrator.calibration_data.values())
        }
        
        return stats
    
    def update_calibration_from_results(self, results: List[Dict]):
        """Met à jour la calibration basée sur les résultats réels"""
        
        for result in results:
            prediction_type = result.get('prediction_type')
            predicted_confidence = result.get('predicted_confidence')
            actual_outcome = result.get('actual_outcome')  # True/False
            
            if all(x is not None for x in [prediction_type, predicted_confidence, actual_outcome]):
                self.calibrator.add_calibration_data(
                    predicted_confidence, actual_outcome, prediction_type
                )
        
        # Recalibration si suffisant de données
        self.calibrator.calibrate_confidence_models()

def test_confidence_scoring_system():
    """Test du système de scoring de confiance"""
    
    print("=== TEST CONFIDENCE SCORING ENGINE ===")
    
    # Initialisation
    confidence_engine = AdvancedConfidenceScorer()
    
    # Données de test
    test_prediction = {
        'prediction_type': 'match_result',
        'prediction_value': '1',
        'model_used': 'xgb+rf+nn_ensemble',
        'consensus_score': 0.85,
        'prediction_uncertainty': 0.2
    }
    
    test_match = {
        'home_team': 'Manchester United',
        'away_team': 'Liverpool', 
        'league': 'Premier_League',
        'date': '2025-01-25',
        'match_importance': 'high',
        'home_team_form': 12,
        'away_team_form': 15,
        'h2h_history': 10,
        'home_team_goals_avg': 2.1,
        'away_team_goals_avg': 1.8,
        'uncertainty_factors': ['key_player_injured']
    }
    
    print("\\n--- Test Calcul Confiance Individuel ---")
    
    confidence_result = confidence_engine.calculate_confidence_score(test_prediction, test_match)
    
    print(f"✅ Score de confiance calculé:")
    print(f"   Score final: {confidence_result['final_confidence_score']}%")
    print(f"   Catégorie: {confidence_result['confidence_category']}")
    print(f"   Score brut: {confidence_result['raw_score']}%")
    print(f"   Score calibré: {confidence_result['calibrated_score']}%")
    
    print(f"\\n   Détail des composantes:")
    for component, score in confidence_result['components'].items():
        print(f"     • {component}: {score}%")
    
    print(f"\\n   Indicateurs de fiabilité:")
    for indicator, value in confidence_result['reliability_indicators'].items():
        print(f"     • {indicator}: {'✅' if value else '❌'}")
    
    # Test en masse
    print(f"\\n--- Test Calcul Confiance en Masse ---")
    
    bulk_data = []
    prediction_types = ['match_result', 'total_goals', 'both_teams_scored', 'over_2_5_goals']
    
    for i, pred_type in enumerate(prediction_types):
        bulk_data.append({
            'prediction_id': f'pred_{i+1}',
            'prediction_data': {
                'prediction_type': pred_type,
                'prediction_value': f'value_{i+1}',
                'model_used': np.random.choice(['xgb', 'rf+xgb', 'nn_ensemble']),
                'consensus_score': np.random.uniform(0.7, 0.9)
            },
            'match_data': test_match
        })
    
    bulk_results = confidence_engine.bulk_confidence_calculation(bulk_data)
    
    print(f"✅ {len(bulk_results)} prédictions traitées en masse:")
    for result in bulk_results:
        if result['status'] == 'success':
            conf_score = result['confidence_info']['final_confidence_score']
            category = result['confidence_info']['confidence_category']
            print(f"   • {result['prediction_id']}: {conf_score}% ({category})")
    
    # Test calibration
    print(f"\\n--- Test Système de Calibration ---")
    
    # Simulation de données de calibration
    calibration_data = []
    for i in range(100):
        pred_conf = np.random.uniform(50, 90)
        # Simulation: plus la confiance est élevée, plus la prédiction a de chances d'être correcte
        actual_outcome = np.random.random() < (pred_conf / 100.0 * 0.8 + 0.1)
        
        calibration_data.append({
            'prediction_type': 'match_result',
            'predicted_confidence': pred_conf,
            'actual_outcome': actual_outcome
        })
    
    confidence_engine.update_calibration_from_results(calibration_data)
    
    print(f"✅ Calibration mise à jour avec {len(calibration_data)} échantillons")
    
    # Test après calibration
    confidence_after_calib = confidence_engine.calculate_confidence_score(test_prediction, test_match)
    print(f"   Score après calibration: {confidence_after_calib['final_confidence_score']}%")
    print(f"   Différence: {abs(confidence_after_calib['final_confidence_score'] - confidence_result['final_confidence_score']):.2f} points")
    
    # Statistiques système
    print(f"\\n--- Statistiques Système ---")
    stats = confidence_engine.get_confidence_statistics()
    
    print(f"   Total prédictions: {stats['total_predictions_scored']}")
    print(f"   Confiance moyenne: {stats['average_confidence']:.2f}%")
    print(f"   Calibration active: {'Oui' if stats['calibration_available'] else 'Non'}")
    print(f"   Données calibration: {stats['calibration_data_points']}")
    
    if 'confidence_by_type' in stats:
        print(f"\\n   Confiance par type:")
        for pred_type, type_stats in stats['confidence_by_type']['mean'].items():
            count = stats['confidence_by_type']['count'][pred_type]
            print(f"     • {pred_type}: {type_stats:.1f}% (n={count})")
    
    print("\\n=== TEST TERMINÉ ===")

if __name__ == "__main__":
    test_confidence_scoring_system()