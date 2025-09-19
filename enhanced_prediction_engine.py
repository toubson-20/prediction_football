#!/usr/bin/env python3
"""
Moteur de Prédiction Enrichi
Utilise les modèles avec features lineups, odds, h2h intégrées
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

class EnhancedPredictionEngine:
    def __init__(self):
        self.models_path = Path("models/enhanced_models")
        self.base_path = Path("data")

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Modèles disponibles
        self.model_types = [
            'goals_scored',
            'both_teams_score',
            'over_2_5_goals',
            'next_match_result'
        ]

        # Compétitions
        self.competitions = {
            39: 'premier_league',
            140: 'la_liga',
            61: 'ligue_1',
            78: 'bundesliga',
            135: 'serie_a',
            2: 'champions_league'
        }

        # Cache des modèles chargés
        self._models_cache = {}
        self._scalers_cache = {}

    def load_enhanced_model(self, league_id: int, model_type: str):
        """Charger modèle enrichi pour une ligue/type"""
        cache_key = f"{league_id}_{model_type}"

        if cache_key not in self._models_cache:
            model_file = self.models_path / f"enhanced_{league_id}_{model_type}.joblib"
            scaler_file = self.models_path / f"enhanced_scaler_{league_id}_{model_type}.joblib"

            if model_file.exists() and scaler_file.exists():
                try:
                    model = joblib.load(model_file)
                    scaler = joblib.load(scaler_file)

                    self._models_cache[cache_key] = model
                    self._scalers_cache[cache_key] = scaler

                    self.logger.info(f"Modèle enrichi chargé: {cache_key}")
                    return model, scaler

                except Exception as e:
                    self.logger.error(f"Erreur chargement {cache_key}: {e}")
                    return None, None
            else:
                self.logger.warning(f"Modèle enrichi non trouvé: {cache_key}")
                return None, None

        return self._models_cache[cache_key], self._scalers_cache[cache_key]

    def get_enhanced_features(self, match_data: Dict) -> Dict:
        """Obtenir features enrichies pour un match"""

        # Features de base (à récupérer du système existant)
        base_features = self._get_base_features(match_data)

        # Features lineups
        lineup_features = self._get_lineup_features(match_data)

        # Features odds
        odds_features = self._get_odds_features(match_data)

        # Features h2h
        h2h_features = self._get_h2h_features(match_data)

        # Combiner toutes les features
        enhanced_features = {
            **base_features,
            **lineup_features,
            **odds_features,
            **h2h_features
        }

        return enhanced_features

    def _get_base_features(self, match_data: Dict) -> Dict:
        """Récupérer features de base du système existant"""
        # Simplification - features de base simulées
        return {
            'home_team_form': match_data.get('home_form', 0.5),
            'away_team_form': match_data.get('away_form', 0.5),
            'home_goals_avg': match_data.get('home_goals_avg', 1.5),
            'away_goals_avg': match_data.get('away_goals_avg', 1.5),
            'home_conceded_avg': match_data.get('home_conceded_avg', 1.0),
            'away_conceded_avg': match_data.get('away_conceded_avg', 1.0),
            'league_avg_goals': match_data.get('league_avg_goals', 2.5),
            'home_advantage': 0.55,
            'season_progress': match_data.get('season_progress', 0.5)
        }

    def _get_lineup_features(self, match_data: Dict) -> Dict:
        """Récupérer features des compositions"""
        fixture_id = match_data.get('fixture_id')
        league_id = match_data.get('league_id')

        if fixture_id and league_id:
            # Utiliser le même système que dans l'intégrateur
            from enhanced_features_integrator import EnhancedFeaturesIntegrator
            integrator = EnhancedFeaturesIntegrator()
            return integrator.extract_lineups_features(fixture_id, league_id)

        # Features par défaut
        return {
            'lineup_strength_home': 0.6,
            'lineup_strength_away': 0.6,
            'formation_attacking_home': 0.5,
            'formation_attacking_away': 0.5,
            'key_players_missing_home': 0,
            'key_players_missing_away': 0,
            'lineup_experience_home': 0.6,
            'lineup_experience_away': 0.6,
            'formation_familiarity_home': 0.7,
            'formation_familiarity_away': 0.7
        }

    def _get_odds_features(self, match_data: Dict) -> Dict:
        """Récupérer features des cotes"""
        fixture_id = match_data.get('fixture_id')
        league_id = match_data.get('league_id')

        if fixture_id and league_id:
            from enhanced_features_integrator import EnhancedFeaturesIntegrator
            integrator = EnhancedFeaturesIntegrator()
            return integrator.extract_odds_features(fixture_id, league_id)

        # Features par défaut
        return {
            'market_confidence_home': 0.4,
            'market_confidence_away': 0.35,
            'market_confidence_draw': 0.25,
            'odds_value_home': 2.2,
            'odds_value_away': 2.8,
            'odds_value_draw': 3.1,
            'market_efficiency': 0.95,
            'bookmakers_consensus': 0.8,
            'over25_market_prob': 0.55,
            'bts_market_prob': 0.48
        }

    def _get_h2h_features(self, match_data: Dict) -> Dict:
        """Récupérer features historiques"""
        home_team_id = match_data.get('home_team_id')
        away_team_id = match_data.get('away_team_id')
        league_id = match_data.get('league_id')

        if home_team_id and away_team_id and league_id:
            from enhanced_features_integrator import EnhancedFeaturesIntegrator
            integrator = EnhancedFeaturesIntegrator()
            return integrator.extract_h2h_features(home_team_id, away_team_id, league_id)

        # Features par défaut
        return {
            'h2h_home_wins': 0.4,
            'h2h_draws': 0.25,
            'h2h_away_wins': 0.35,
            'h2h_total_matches': 8,
            'h2h_avg_goals': 2.6,
            'h2h_avg_home_goals': 1.4,
            'h2h_avg_away_goals': 1.2,
            'h2h_home_advantage': 0.52,
            'recent_h2h_trend_home': 0.45,
            'h2h_over25_rate': 0.6,
            'h2h_bts_rate': 0.55,
            'h2h_recency_weight': 0.8
        }

    def predict_match_enhanced(self, match_data: Dict) -> Dict:
        """Prédire un match avec modèles enrichis"""
        league_id = match_data.get('league_id')

        if not league_id or league_id not in self.competitions:
            return {'error': 'Ligue non supportée'}

        # Obtenir features enrichies
        features = self.get_enhanced_features(match_data)

        # Préparer array de features
        feature_names = sorted(features.keys())
        X = np.array([[features[name] for name in feature_names]])

        predictions = {}
        confidence_scores = {}

        # Prédire avec chaque type de modèle
        for model_type in self.model_types:
            model, scaler = self.load_enhanced_model(league_id, model_type)

            if model and scaler:
                try:
                    # Normaliser features
                    X_scaled = scaler.transform(X)

                    # Prédiction
                    pred = model.predict(X_scaled)[0]

                    # Ajuster selon le type de prédiction
                    if model_type == 'goals_scored':
                        pred = max(0.0, min(6.0, pred))
                    elif model_type in ['both_teams_score', 'over_2_5_goals']:
                        pred = max(0.0, min(1.0, pred))
                    elif model_type == 'next_match_result':
                        pred = max(0.0, min(2.0, pred))

                    predictions[model_type] = pred

                    # Score de confiance basé sur la cohérence des features
                    confidence = self._calculate_confidence(features, model_type)
                    confidence_scores[model_type] = confidence

                except Exception as e:
                    self.logger.error(f"Erreur prédiction {model_type}: {e}")

            else:
                # Fallback vers prédictions de base
                predictions[model_type] = self._fallback_prediction(features, model_type)
                confidence_scores[model_type] = 0.6

        return {
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'features_used': len(features),
            'enhanced': True,
            'league': self.competitions[league_id]
        }

    def _calculate_confidence(self, features: Dict, model_type: str) -> float:
        """Calculer score de confiance basé sur la qualité des features"""

        # Facteurs de confiance
        confidence_factors = []

        # Qualité des données lineups
        if features.get('lineup_strength_home', 0) > 0.3 and features.get('lineup_strength_away', 0) > 0.3:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)

        # Qualité des données odds
        if features.get('market_efficiency', 0) > 0.9 and features.get('bookmakers_consensus', 0) > 0.7:
            confidence_factors.append(0.85)
        else:
            confidence_factors.append(0.65)

        # Qualité des données h2h
        h2h_matches = features.get('h2h_total_matches', 0)
        if h2h_matches >= 5:
            confidence_factors.append(0.9)
        elif h2h_matches >= 2:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)

        # Cohérence générale
        recency_weight = features.get('h2h_recency_weight', 0.5)
        confidence_factors.append(0.6 + 0.3 * recency_weight)

        return np.mean(confidence_factors)

    def _fallback_prediction(self, features: Dict, model_type: str) -> float:
        """Prédiction de secours si modèle enrichi non disponible"""

        if model_type == 'goals_scored':
            home_avg = features.get('home_goals_avg', 1.5)
            away_avg = features.get('away_goals_avg', 1.5)
            league_avg = features.get('league_avg_goals', 2.5)
            return (home_avg + away_avg + league_avg) / 3

        elif model_type == 'both_teams_score':
            home_attack = features.get('home_goals_avg', 1.5)
            away_attack = features.get('away_goals_avg', 1.5)
            bts_prob = min(home_attack * away_attack / 4, 0.9)
            return bts_prob

        elif model_type == 'over_2_5_goals':
            total_expected = features.get('h2h_avg_goals', 2.5)
            return 1.0 if total_expected > 2.5 else 0.3

        elif model_type == 'next_match_result':
            home_form = features.get('home_team_form', 0.5)
            away_form = features.get('away_team_form', 0.5)
            if home_form > away_form + 0.1:
                return 1.0  # Victoire domicile
            elif away_form > home_form + 0.1:
                return 0.0  # Victoire extérieur
            else:
                return 0.5  # Match nul

        return 0.5

    def predict_matches_batch(self, matches_data: List[Dict]) -> List[Dict]:
        """Prédire plusieurs matchs en lot"""

        results = []

        for match_data in matches_data:
            try:
                prediction = self.predict_match_enhanced(match_data)
                prediction['match_id'] = match_data.get('fixture_id', 'unknown')
                results.append(prediction)

            except Exception as e:
                self.logger.error(f"Erreur prédiction match {match_data.get('fixture_id')}: {e}")
                results.append({
                    'match_id': match_data.get('fixture_id', 'unknown'),
                    'error': str(e)
                })

        return results

# Test du moteur
if __name__ == "__main__":
    print("=== TEST MOTEUR PREDICTION ENRICHI ===")

    engine = EnhancedPredictionEngine()

    # Test avec match simulé
    test_match = {
        'fixture_id': 12345,
        'league_id': 39,  # Premier League
        'home_team_id': 33,  # Manchester United
        'away_team_id': 40,  # Liverpool
        'home_form': 0.7,
        'away_form': 0.8,
        'home_goals_avg': 2.1,
        'away_goals_avg': 2.3,
        'season_progress': 0.3
    }

    result = engine.predict_match_enhanced(test_match)

    print(f"Prédictions: {result['predictions']}")
    print(f"Confiance: {result['confidence_scores']}")
    print(f"Features utilisées: {result['features_used']}")
    print(f"Enrichi: {result['enhanced']}")