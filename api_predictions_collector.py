#!/usr/bin/env python3
"""
Collecteur de Prédictions API Football
Récupère les prédictions officielles pour meta-learning
"""

import requests
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from config import Config

class APIPredictionsCollector:
    def __init__(self):
        self.api_key = Config.FOOTBALL_API_KEY
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            'x-apisports-key': self.api_key
        }

        # Répertoire pour prédictions
        self.predictions_dir = Path("data/api_predictions")
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Competitions principales
        self.competitions = {
            39: 'premier_league',
            140: 'la_liga',
            61: 'ligue_1',
            78: 'bundesliga',
            135: 'serie_a',
            2: 'champions_league'
        }

    def api_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Faire requête API avec gestion d'erreurs"""
        url = f"{self.base_url}/{endpoint}"

        try:
            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Erreur API {response.status_code}: {response.text}")
                return None

        except Exception as e:
            self.logger.error(f"Erreur requête API: {e}")
            return None

    def get_upcoming_fixtures(self, league_id: int, days_ahead: int = 7) -> List[Dict]:
        """Récupérer les prochains matchs d'une ligue"""
        self.logger.info(f"Récupération prochains matchs ligue {league_id}")

        start_date = datetime.now()
        end_date = start_date + timedelta(days=days_ahead)

        data = self.api_request('fixtures', {
            'league': league_id,
            'season': 2025,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'status': 'NS'  # Not Started
        })

        if data and data.get('response'):
            self.logger.info(f"  {len(data['response'])} matchs trouvés")
            return data['response']

        return []

    def get_fixture_predictions(self, fixture_id: int) -> Optional[Dict]:
        """Récupérer prédictions pour un match spécifique"""
        self.logger.info(f"Récupération prédictions fixture {fixture_id}")

        data = self.api_request('predictions', {
            'fixture': fixture_id
        })

        if data and data.get('response'):
            return data['response'][0] if data['response'] else None

        return None

    def extract_prediction_features(self, prediction_data: Dict) -> Dict:
        """Extraire features des prédictions API Football"""
        features = {
            'api_predictions_available': False,
            'api_home_win_percent': 0.0,
            'api_draw_percent': 0.0,
            'api_away_win_percent': 0.0,
            'api_under_over_under': 0.0,
            'api_under_over_over': 0.0,
            'api_goals_home': 0.0,
            'api_goals_away': 0.0,
            'api_advice': 'none',
            'api_comparison_att_home': 0.0,
            'api_comparison_att_away': 0.0,
            'api_comparison_def_home': 0.0,
            'api_comparison_def_away': 0.0,
            'api_comparison_h2h_home': 0.0,
            'api_comparison_h2h_away': 0.0,
            'api_form_home': 0.0,
            'api_form_away': 0.0
        }

        if not prediction_data:
            return features

        try:
            features['api_predictions_available'] = True

            # Prédictions principales
            predictions = prediction_data.get('predictions', {})
            if predictions:
                winner = predictions.get('winner', {})
                if winner:
                    if winner.get('id') == prediction_data.get('teams', {}).get('home', {}).get('id'):
                        features['api_home_win_percent'] = 0.7
                        features['api_away_win_percent'] = 0.2
                        features['api_draw_percent'] = 0.1
                    elif winner.get('id') == prediction_data.get('teams', {}).get('away', {}).get('id'):
                        features['api_home_win_percent'] = 0.2
                        features['api_away_win_percent'] = 0.7
                        features['api_draw_percent'] = 0.1
                    else:
                        features['api_home_win_percent'] = 0.3
                        features['api_away_win_percent'] = 0.3
                        features['api_draw_percent'] = 0.4

                # Under/Over
                under_over = predictions.get('under_over')
                if under_over:
                    if '2.5' in under_over:
                        if 'over' in under_over.lower():
                            features['api_under_over_over'] = 0.7
                            features['api_under_over_under'] = 0.3
                        else:
                            features['api_under_over_over'] = 0.3
                            features['api_under_over_under'] = 0.7

                # Goals prédits
                goals = predictions.get('goals', {})
                if goals:
                    features['api_goals_home'] = float(goals.get('home', '1.5').replace('-', '1.5'))
                    features['api_goals_away'] = float(goals.get('away', '1.5').replace('-', '1.5'))

                # Advice
                advice = predictions.get('advice', '')
                features['api_advice'] = advice.lower() if advice else 'none'

            # Comparaisons d'équipes
            comparison = prediction_data.get('comparison', {})
            if comparison:
                for key, values in comparison.items():
                    if isinstance(values, dict) and 'home' in values and 'away' in values:
                        home_val = self._parse_percentage(values['home'])
                        away_val = self._parse_percentage(values['away'])

                        if key == 'att':
                            features['api_comparison_att_home'] = home_val
                            features['api_comparison_att_away'] = away_val
                        elif key == 'def':
                            features['api_comparison_def_home'] = home_val
                            features['api_comparison_def_away'] = away_val
                        elif key == 'h2h':
                            features['api_comparison_h2h_home'] = home_val
                            features['api_comparison_h2h_away'] = away_val

            # Forme des équipes
            teams = prediction_data.get('teams', {})
            if teams:
                home_form = teams.get('home', {}).get('league', {}).get('form', '')
                away_form = teams.get('away', {}).get('league', {}).get('form', '')

                features['api_form_home'] = self._calculate_form_score(home_form)
                features['api_form_away'] = self._calculate_form_score(away_form)

        except Exception as e:
            self.logger.warning(f"Erreur extraction features prédictions: {e}")

        return features

    def _parse_percentage(self, value: str) -> float:
        """Parser pourcentage depuis string"""
        try:
            if isinstance(value, str) and '%' in value:
                return float(value.replace('%', '')) / 100.0
            return float(value) if value else 0.0
        except:
            return 0.0

    def _calculate_form_score(self, form: str) -> float:
        """Calculer score de forme depuis string WWLDL"""
        if not form:
            return 0.5

        win_count = form.count('W')
        total = len(form)

        return win_count / total if total > 0 else 0.5

    def collect_predictions_for_league(self, league_id: int, league_name: str):
        """Collecter prédictions pour une ligue"""
        self.logger.info(f"=== COLLECTE PREDICTIONS {league_name.upper()} ===")

        # Récupérer prochains matchs
        fixtures = self.get_upcoming_fixtures(league_id)

        if not fixtures:
            self.logger.warning(f"Aucun match trouvé pour {league_name}")
            return

        predictions_data = []

        for fixture in fixtures[:10]:  # Limiter à 10 matchs pour éviter rate limit
            fixture_id = fixture['fixture']['id']

            # Récupérer prédictions
            prediction = self.get_fixture_predictions(fixture_id)

            if prediction:
                # Extraire features
                features = self.extract_prediction_features(prediction)

                # Ajouter infos du match
                match_data = {
                    'fixture_id': fixture_id,
                    'league_id': league_id,
                    'home_team': fixture['teams']['home']['name'],
                    'away_team': fixture['teams']['away']['name'],
                    'date': fixture['fixture']['date'],
                    'collected_at': datetime.now().isoformat(),
                    **features
                }

                predictions_data.append(match_data)
                self.logger.info(f"  Prédictions collectées: {fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']}")

            # Rate limiting
            time.sleep(1)

        # Sauvegarder
        if predictions_data:
            output_file = self.predictions_dir / f"predictions_{league_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(predictions_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Prédictions sauvées: {output_file} ({len(predictions_data)} matchs)")

    def collect_all_predictions(self):
        """Collecter prédictions pour toutes les ligues"""
        self.logger.info("=== DEBUT COLLECTE PREDICTIONS API FOOTBALL ===")
        start_time = datetime.now()

        total_collected = 0

        for league_id, league_name in self.competitions.items():
            try:
                self.collect_predictions_for_league(league_id, league_name)
                total_collected += 1

                # Pause entre ligues
                time.sleep(2)

            except Exception as e:
                self.logger.error(f"Erreur collecte {league_name}: {e}")
                continue

        duration = datetime.now() - start_time
        self.logger.info(f"=== COLLECTE TERMINEE - {total_collected} ligues - Durée: {duration} ===")

if __name__ == "__main__":
    print("=" * 70)
    print("COLLECTEUR PREDICTIONS API FOOTBALL")
    print("=" * 70)

    collector = APIPredictionsCollector()
    collector.collect_all_predictions()

    print("\nSUCCES! Prédictions API Football collectées pour meta-learning")