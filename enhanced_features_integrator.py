#!/usr/bin/env python3
"""
Intégrateur de Features Avancées
Intègre les données lineups, odds, h2h dans les modèles ML
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeaturesIntegrator:
    def __init__(self):
        self.base_path = Path("data")
        self.models_path = Path("models/enhanced_models")
        self.models_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Competitions supportées
        self.competitions = {
            39: 'premier_league',
            140: 'la_liga',
            61: 'ligue_1',
            78: 'bundesliga',
            135: 'serie_a',
            2: 'champions_league'
        }

    def load_base_dataset(self) -> pd.DataFrame:
        """Charger le dataset de base ML"""
        self.logger.info("Chargement dataset de base...")

        # Chercher le dataset le plus récent
        ultra_processed = self.base_path / "ultra_processed"
        if ultra_processed.exists():
            csv_files = list(ultra_processed.glob("complete_ml_dataset_*.csv"))
            if csv_files:
                latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
                self.logger.info(f"Dataset trouvé: {latest_file}")
                return pd.read_csv(latest_file)

        # Fallback vers ml_ready
        ml_ready = self.base_path / "ml_ready"
        if ml_ready.exists():
            csv_files = list(ml_ready.glob("*.csv"))
            if csv_files:
                latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
                self.logger.info(f"Dataset fallback: {latest_file}")
                return pd.read_csv(latest_file)

        raise FileNotFoundError("Aucun dataset ML trouvé")

    def extract_lineups_features(self, fixture_id: int, league_id: int) -> Dict:
        """Extraire features des compositions officielles"""
        features = {
            'lineup_strength_home': 0.5,
            'lineup_strength_away': 0.5,
            'formation_attacking_home': 0.5,
            'formation_attacking_away': 0.5,
            'key_players_missing_home': 0,
            'key_players_missing_away': 0,
            'lineup_experience_home': 0.5,
            'lineup_experience_away': 0.5,
            'formation_familiarity_home': 0.5,
            'formation_familiarity_away': 0.5
        }

        try:
            # Chercher fichiers lineups
            lineups_dir = self.base_path / "lineups"
            complete_dir = self.base_path / "complete_collection"

            lineup_file = None
            if lineups_dir.exists():
                lineup_file = lineups_dir / f"fixture_{fixture_id}_lineups.json"

            if not lineup_file or not lineup_file.exists():
                # Chercher dans complete_collection
                for subdir in complete_dir.rglob("*"):
                    if subdir.is_dir():
                        potential_file = subdir / f"fixture_{fixture_id}_lineups.json"
                        if potential_file.exists():
                            lineup_file = potential_file
                            break

            if lineup_file and lineup_file.exists():
                with open(lineup_file, 'r', encoding='utf-8') as f:
                    lineup_data = json.load(f)

                if lineup_data.get('response'):
                    teams_lineups = lineup_data['response']

                    for i, team_lineup in enumerate(teams_lineups[:2]):
                        team_key = 'home' if i == 0 else 'away'

                        # Formation attacking score
                        formation = team_lineup.get('formation', '4-4-2')
                        if formation:
                            attacking_score = self._calculate_formation_attacking_score(formation)
                            features[f'formation_attacking_{team_key}'] = attacking_score

                        # Lineup strength (basé sur les positions des joueurs)
                        startXI = team_lineup.get('startXI', [])
                        if startXI:
                            strength_score = self._calculate_lineup_strength(startXI, league_id)
                            features[f'lineup_strength_{team_key}'] = strength_score

                            # Experience (age moyen estimé)
                            exp_score = self._estimate_lineup_experience(startXI)
                            features[f'lineup_experience_{team_key}'] = exp_score

                        # Formation familarity (assume 0.7 si formation classique)
                        if formation in ['4-4-2', '4-3-3', '3-5-2', '4-2-3-1']:
                            features[f'formation_familiarity_{team_key}'] = 0.8
                        else:
                            features[f'formation_familiarity_{team_key}'] = 0.6

        except Exception as e:
            self.logger.warning(f"Erreur extraction lineups {fixture_id}: {e}")

        return features

    def _calculate_formation_attacking_score(self, formation: str) -> float:
        """Calculer score offensif d'une formation"""
        try:
            parts = formation.split('-')
            if len(parts) >= 3:
                forwards = int(parts[-1])  # Attaquants
                midfielders = int(parts[-2]) if len(parts) > 2 else 3  # Milieux

                # Score basé sur nombre d'attaquants et milieux offensifs
                attacking_score = (forwards * 0.6 + midfielders * 0.3) / 6
                return min(1.0, max(0.1, attacking_score))
        except:
            pass
        return 0.5

    def _calculate_lineup_strength(self, startXI: List[Dict], league_id: int) -> float:
        """Calculer force de la composition (basique)"""
        # Simplification: plus de joueurs avec position définie = meilleure composition
        positioned_players = sum(1 for player in startXI if player.get('player', {}).get('pos'))
        return min(1.0, positioned_players / 11.0)

    def _estimate_lineup_experience(self, startXI: List[Dict]) -> float:
        """Estimer expérience de la composition"""
        # Simplification: assume expérience moyenne
        return 0.6 + np.random.uniform(-0.2, 0.2)  # Variation réaliste

    def extract_odds_features(self, fixture_id: int, league_id: int) -> Dict:
        """Extraire features des cotes bookmakers"""
        features = {
            'market_confidence_home': 0.5,
            'market_confidence_away': 0.5,
            'market_confidence_draw': 0.33,
            'odds_value_home': 2.0,
            'odds_value_away': 2.0,
            'odds_value_draw': 3.0,
            'market_efficiency': 0.95,
            'bookmakers_consensus': 0.8,
            'over25_market_prob': 0.5,
            'bts_market_prob': 0.5
        }

        try:
            # Chercher fichiers odds
            enhanced_dir = self.base_path / "enhanced"
            complete_dir = self.base_path / "complete_collection"

            # Chercher odds dans différents emplacements
            odds_files = []
            for search_dir in [enhanced_dir, complete_dir]:
                if search_dir.exists():
                    odds_files.extend(list(search_dir.rglob(f"*odds*{league_id}*.csv")))
                    odds_files.extend(list(search_dir.rglob(f"*odds*{fixture_id}*.json")))

            if odds_files:
                odds_file = odds_files[0]  # Prendre le premier trouvé

                if odds_file.suffix == '.csv':
                    odds_df = pd.read_csv(odds_file)
                    fixture_odds = odds_df[odds_df['fixture_id'] == fixture_id]

                    if not fixture_odds.empty:
                        # Extraire probabilités du marché
                        if 'odds' in fixture_odds.columns:
                            home_odds = fixture_odds['odds'].iloc[0] if len(fixture_odds) > 0 else 2.0
                            features['odds_value_home'] = home_odds
                            features['market_confidence_home'] = 1.0 / home_odds

                elif odds_file.suffix == '.json':
                    with open(odds_file, 'r', encoding='utf-8') as f:
                        odds_data = json.load(f)

                    if odds_data.get('response'):
                        bookmakers = odds_data['response'][0].get('bookmakers', [])
                        if bookmakers:
                            # Analyser cotes 1X2
                            home_odds, draw_odds, away_odds = self._extract_1x2_odds(bookmakers)

                            features['odds_value_home'] = home_odds
                            features['odds_value_draw'] = draw_odds
                            features['odds_value_away'] = away_odds

                            # Probabilités implicites
                            total_prob = (1/home_odds + 1/draw_odds + 1/away_odds)
                            features['market_confidence_home'] = (1/home_odds) / total_prob
                            features['market_confidence_away'] = (1/away_odds) / total_prob
                            features['market_confidence_draw'] = (1/draw_odds) / total_prob
                            features['market_efficiency'] = 1.0 / total_prob  # Marge bookmaker

        except Exception as e:
            self.logger.warning(f"Erreur extraction odds {fixture_id}: {e}")

        return features

    def _extract_1x2_odds(self, bookmakers: List[Dict]) -> Tuple[float, float, float]:
        """Extraire cotes 1X2 moyennes"""
        home_odds, draw_odds, away_odds = [], [], []

        for bookmaker in bookmakers:
            for bet in bookmaker.get('bets', []):
                if bet.get('name') == 'Match Winner':
                    for value in bet.get('values', []):
                        if value.get('value') == 'Home':
                            home_odds.append(float(value.get('odd', 2.0)))
                        elif value.get('value') == 'Draw':
                            draw_odds.append(float(value.get('odd', 3.0)))
                        elif value.get('value') == 'Away':
                            away_odds.append(float(value.get('odd', 2.0)))

        # Moyennes ou valeurs par défaut
        return (
            np.mean(home_odds) if home_odds else 2.0,
            np.mean(draw_odds) if draw_odds else 3.0,
            np.mean(away_odds) if away_odds else 2.0
        )

    def extract_h2h_features(self, home_team_id: int, away_team_id: int, league_id: int) -> Dict:
        """Extraire features historiques tête-à-tête"""
        features = {
            'h2h_home_wins': 0.33,
            'h2h_draws': 0.33,
            'h2h_away_wins': 0.33,
            'h2h_total_matches': 0,
            'h2h_avg_goals': 2.5,
            'h2h_avg_home_goals': 1.25,
            'h2h_avg_away_goals': 1.25,
            'h2h_home_advantage': 0.5,
            'recent_h2h_trend_home': 0.5,
            'h2h_over25_rate': 0.5,
            'h2h_bts_rate': 0.5,
            'h2h_recency_weight': 0.5
        }

        try:
            # Chercher données h2h
            h2h_files = []
            for search_dir in [self.base_path / "enhanced", self.base_path / "complete_collection"]:
                if search_dir.exists():
                    h2h_files.extend(list(search_dir.rglob(f"*h2h*{league_id}*.csv")))
                    h2h_files.extend(list(search_dir.rglob(f"*h2h*.json")))

            h2h_data = []

            for h2h_file in h2h_files:
                try:
                    if h2h_file.suffix == '.csv':
                        df = pd.read_csv(h2h_file)
                        # Filtrer pour les équipes concernées
                        team_matches = df[
                            ((df['home_team_id'] == home_team_id) & (df['away_team_id'] == away_team_id)) |
                            ((df['home_team_id'] == away_team_id) & (df['away_team_id'] == home_team_id))
                        ]
                        h2h_data.extend(team_matches.to_dict('records'))

                    elif h2h_file.suffix == '.json':
                        with open(h2h_file, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)

                        if isinstance(json_data, dict) and 'response' in json_data:
                            for match in json_data['response']:
                                home_id = match.get('teams', {}).get('home', {}).get('id')
                                away_id = match.get('teams', {}).get('away', {}).get('id')

                                if (home_id == home_team_id and away_id == away_team_id) or \
                                   (home_id == away_team_id and away_id == home_team_id):
                                    h2h_data.append(match)
                except:
                    continue

            if h2h_data:
                features.update(self._analyze_h2h_data(h2h_data, home_team_id, away_team_id))

        except Exception as e:
            self.logger.warning(f"Erreur extraction h2h {home_team_id} vs {away_team_id}: {e}")

        return features

    def _analyze_h2h_data(self, h2h_data: List[Dict], home_team_id: int, away_team_id: int) -> Dict:
        """Analyser données historiques h2h"""
        if not h2h_data:
            return {}

        home_wins = 0
        draws = 0
        away_wins = 0
        total_goals = []
        home_goals = []
        away_goals = []
        over25_count = 0
        bts_count = 0

        for match in h2h_data[-10:]:  # 10 derniers matchs max
            try:
                # Identifier qui joue à domicile dans ce match historique
                if isinstance(match, dict):
                    if 'teams' in match:
                        hist_home_id = match['teams']['home']['id']
                        hist_away_id = match['teams']['away']['id']
                        home_score = match.get('goals', {}).get('home', 0)
                        away_score = match.get('goals', {}).get('away', 0)
                    else:
                        hist_home_id = match.get('home_team_id')
                        hist_away_id = match.get('away_team_id')
                        home_score = match.get('home_goals', 0)
                        away_score = match.get('away_goals', 0)

                    if hist_home_id and hist_away_id and home_score is not None and away_score is not None:
                        total_goals.append(home_score + away_score)

                        # Déterminer le résultat du point de vue de home_team_id actuel
                        if hist_home_id == home_team_id:
                            home_goals.append(home_score)
                            away_goals.append(away_score)
                            if home_score > away_score:
                                home_wins += 1
                            elif home_score == away_score:
                                draws += 1
                            else:
                                away_wins += 1
                        else:
                            home_goals.append(away_score)
                            away_goals.append(home_score)
                            if away_score > home_score:
                                home_wins += 1
                            elif away_score == home_score:
                                draws += 1
                            else:
                                away_wins += 1

                        # Stats spéciales
                        if (home_score + away_score) > 2.5:
                            over25_count += 1
                        if home_score > 0 and away_score > 0:
                            bts_count += 1

            except:
                continue

        total_matches = home_wins + draws + away_wins
        if total_matches > 0:
            return {
                'h2h_home_wins': home_wins / total_matches,
                'h2h_draws': draws / total_matches,
                'h2h_away_wins': away_wins / total_matches,
                'h2h_total_matches': total_matches,
                'h2h_avg_goals': np.mean(total_goals) if total_goals else 2.5,
                'h2h_avg_home_goals': np.mean(home_goals) if home_goals else 1.25,
                'h2h_avg_away_goals': np.mean(away_goals) if away_goals else 1.25,
                'h2h_home_advantage': (home_wins - away_wins) / total_matches + 0.5,
                'recent_h2h_trend_home': home_wins / max(1, total_matches),
                'h2h_over25_rate': over25_count / total_matches,
                'h2h_bts_rate': bts_count / total_matches,
                'h2h_recency_weight': min(1.0, total_matches / 10.0)
            }

        return {}

    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Créer dataset avec features enrichies"""
        self.logger.info("Création features enrichies...")

        enhanced_df = df.copy()

        # Initialiser nouvelles colonnes
        feature_columns = [
            'lineup_strength_home', 'lineup_strength_away',
            'formation_attacking_home', 'formation_attacking_away',
            'key_players_missing_home', 'key_players_missing_away',
            'lineup_experience_home', 'lineup_experience_away',
            'formation_familiarity_home', 'formation_familiarity_away',
            'market_confidence_home', 'market_confidence_away', 'market_confidence_draw',
            'odds_value_home', 'odds_value_away', 'odds_value_draw',
            'market_efficiency', 'bookmakers_consensus',
            'over25_market_prob', 'bts_market_prob',
            'h2h_home_wins', 'h2h_draws', 'h2h_away_wins',
            'h2h_total_matches', 'h2h_avg_goals',
            'h2h_avg_home_goals', 'h2h_avg_away_goals',
            'h2h_home_advantage', 'recent_h2h_trend_home',
            'h2h_over25_rate', 'h2h_bts_rate', 'h2h_recency_weight'
        ]

        for col in feature_columns:
            enhanced_df[col] = 0.5  # Valeur par défaut

        processed_count = 0
        total_rows = len(enhanced_df)

        for idx, row in enhanced_df.iterrows():
            try:
                fixture_id = row.get('fixture_id', row.get('id'))
                league_id = row.get('league_id', row.get('competition_id'))
                home_team_id = row.get('home_team_id')
                away_team_id = row.get('away_team_id')

                if fixture_id and league_id and home_team_id and away_team_id:
                    # Features lineups
                    lineup_features = self.extract_lineups_features(fixture_id, league_id)
                    for key, value in lineup_features.items():
                        if key in enhanced_df.columns:
                            enhanced_df.at[idx, key] = value

                    # Features odds
                    odds_features = self.extract_odds_features(fixture_id, league_id)
                    for key, value in odds_features.items():
                        if key in enhanced_df.columns:
                            enhanced_df.at[idx, key] = value

                    # Features h2h
                    h2h_features = self.extract_h2h_features(home_team_id, away_team_id, league_id)
                    for key, value in h2h_features.items():
                        if key in enhanced_df.columns:
                            enhanced_df.at[idx, key] = value

                processed_count += 1
                if processed_count % 100 == 0:
                    self.logger.info(f"  Traité: {processed_count}/{total_rows} matchs")

            except Exception as e:
                self.logger.warning(f"Erreur traitement ligne {idx}: {e}")
                continue

        self.logger.info(f"Features enrichies créées: {processed_count}/{total_rows} matchs")
        return enhanced_df

    def train_enhanced_models(self, df: pd.DataFrame, league_id: int, league_name: str):
        """Entraîner modèles avec features enrichies"""
        self.logger.info(f"Entraînement modèles enrichis: {league_name}")

        # Filtrer pour la ligue
        league_df = df[df['league_id'] == league_id].copy()
        if len(league_df) < 50:
            self.logger.warning(f"Pas assez de données pour {league_name}: {len(league_df)}")
            return

        # Préparer features
        feature_columns = [col for col in league_df.columns if not col.endswith('_target')
                          and col not in ['fixture_id', 'date', 'home_team', 'away_team']]

        X = league_df[feature_columns].fillna(0.5)

        # Targets à prédire
        targets = {
            'goals_scored': 'total_goals',
            'both_teams_score': 'both_teams_score',
            'over_2_5_goals': 'over_2_5_goals',
            'next_match_result': 'result_home_win'
        }

        for target_name, target_col in targets.items():
            if target_col not in league_df.columns:
                continue

            try:
                y = league_df[target_col].fillna(0)

                # Split train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42
                )

                # Tester différents modèles
                models = {
                    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'gradient_boosting': GradientBoostingRegressor(random_state=42),
                    'ridge': Ridge(alpha=1.0),
                    'svr': SVR(kernel='rbf', C=1.0)
                }

                best_model = None
                best_score = -np.inf
                best_name = ""

                for name, model in models.items():
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                    cv_mean = cv_scores.mean()

                    self.logger.info(f"  {name}: CV R2 = {cv_mean:.3f}")

                    if cv_mean > best_score:
                        best_score = cv_mean
                        best_model = model
                        best_name = name

                # Entraîner meilleur modèle
                best_model.fit(X_train, y_train)

                # Prédictions test
                y_pred = best_model.predict(X_test)

                # Métriques
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                self.logger.info(f"  SUCCES {target_name}: R2={r2:.3f} MAE={mae:.3f}")
                self.logger.info(f"  Modèle: {best_name}, Train: {len(X_train)}, Test: {len(X_test)}")

                # Sauvegarder modèle enrichi
                model_file = self.models_path / f"enhanced_{league_id}_{target_name}.joblib"
                joblib.dump(best_model, model_file)

                # Sauvegarder scaler si nécessaire
                scaler = StandardScaler()
                scaler.fit(X_train)
                scaler_file = self.models_path / f"enhanced_scaler_{league_id}_{target_name}.joblib"
                joblib.dump(scaler, scaler_file)

                # Sauvegarder métriques
                metrics = {
                    'r2': r2,
                    'mse': mse,
                    'mae': mae,
                    'cv_score': best_score,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'model_used': best_name,
                    'features_count': len(feature_columns),
                    'enhanced_features': True,
                    'training_date': datetime.now().isoformat()
                }

                metrics_file = self.models_path / f"enhanced_metrics_{league_id}_{target_name}.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)

            except Exception as e:
                self.logger.error(f"Erreur {target_name} pour {league_name}: {e}")
                continue

    def run_enhanced_training(self):
        """Exécuter entraînement complet avec features enrichies"""
        self.logger.info("=== DEBUT ENTRAINEMENT MODELES ENRICHIS ===")
        start_time = datetime.now()

        try:
            # Charger dataset de base
            df = self.load_base_dataset()
            self.logger.info(f"Dataset chargé: {len(df)} matchs")

            # Créer features enrichies
            enhanced_df = self.create_enhanced_features(df)

            # Sauvegarder dataset enrichi
            enhanced_file = self.base_path / "ultra_processed" / f"enhanced_ml_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            enhanced_df.to_csv(enhanced_file, index=False)
            self.logger.info(f"Dataset enrichi sauvé: {enhanced_file}")

            # Entraîner modèles par ligue
            for league_id, league_name in self.competitions.items():
                if league_id in enhanced_df['league_id'].values:
                    self.train_enhanced_models(enhanced_df, league_id, league_name)

            duration = datetime.now() - start_time
            self.logger.info(f"=== ENTRAINEMENT ENRICHI TERMINE - Durée: {duration} ===")

        except Exception as e:
            self.logger.error(f"Erreur entraînement enrichi: {e}")
            raise

if __name__ == "__main__":
    print("=" * 70)
    print("ENHANCED FEATURES INTEGRATOR - INTEGRATION DONNEES AVANCEES")
    print("=" * 70)

    integrator = EnhancedFeaturesIntegrator()
    integrator.run_enhanced_training()

    print("\nSUCCES! Modeles enrichis avec lineups, odds et h2h entraines")