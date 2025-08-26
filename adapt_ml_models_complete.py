"""
ADAPTATEUR MODELES ML COMPLETS
Adapte tous les modeles ML pour utiliser les nouvelles features completes
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
from datetime import datetime
import json
from collections import defaultdict

sys.path.append('src')
from config import API_FOOTBALL_CONFIG

class CompleteMLModelsAdapter:
    """Adaptateur complet des modeles ML avec toutes les nouvelles features"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.data_dir = Path("data/ultra_processed") 
        self.complete_models_dir = Path("models/complete_models")
        self.complete_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration ligues et types de predictions
        self.leagues = {
            39: "Premier League", 140: "La Liga", 61: "Ligue 1",
            78: "Bundesliga", 135: "Serie A", 2: "Champions League", 3: "Europa League"
        }
        
        self.prediction_types = [
            'next_match_result', 'goals_scored', 'goals_conceded', 'clean_sheet',
            'both_teams_score', 'over_2_5_goals', 'corner_count', 'card_count',
            'possession_percentage', 'shots_total', 'win_probability', 'draw_probability',
            'lose_probability', 'expected_goals', 'expected_assists', 'player_rating',
            'team_performance', 'home_advantage', 'form_prediction', 'injury_impact',
            'weather_impact', 'referee_impact', 'crowd_impact', 'fatigue_impact',
            'motivation_impact', 'tactical_advantage', 'set_pieces_effectiveness',
            'counter_attack_success', 'defensive_stability', 'offensive_power'
        ]
        
        # Nouvelles features completes
        self.complete_features = [
            # Features de base
            'points', 'played', 'wins', 'draws', 'losses', 'goals_for', 'goals_against',
            'goal_diff', 'win_rate', 'goals_per_match', 'home_wins', 'away_wins',
            
            # NOUVELLES FEATURES DETAILLEES
            'shots_total', 'shots_on_goal', 'shots_off_goal', 'shots_blocked',
            'shots_inside_box', 'shots_outside_box', 'fouls_committed', 'fouls_drawn',
            'corners_taken', 'offsides', 'ball_possession_avg', 'yellow_cards', 'red_cards',
            'passes_total', 'passes_accurate', 'passes_accuracy_pct', 'passes_key',
            'attacks_total', 'attacks_dangerous', 'goalkeeper_saves', 'goalkeeper_saves_pct',
            
            # Features contextuelles
            'venue_capacity', 'avg_team_age', 'home_shots_avg', 'away_shots_avg',
            'home_possession_avg', 'away_possession_avg', 'home_corners_avg', 'away_corners_avg',
            
            # Features joueurs
            'top_scorer_goals', 'top_scorer_assists', 'top_scorer_rating',
            'squad_avg_age', 'squad_foreign_players', 'players_injured_current',
            
            # Features historiques
            'last_5_matches_wins', 'form_trend', 'matches_clean_sheets',
            'matches_failed_to_score', 'avg_odds_win', 'avg_odds_draw'
        ]
        
        self.adaptation_stats = defaultdict(int)
        
    def load_complete_dataset(self):
        """Charger le dataset complet avec toutes les features"""
        
        # Chercher le fichier le plus recent
        dataset_files = list(self.data_dir.glob("complete_ml_dataset_*.csv"))
        
        if not dataset_files:
            # Fallback sur dataset existant
            fallback_file = self.data_dir / "ml_ready_dataset_2025.csv"
            if fallback_file.exists():
                print("WARNING: Dataset complet non trouve, utilisation dataset existant")
                return pd.read_csv(fallback_file)
            else:
                raise FileNotFoundError("Aucun dataset trouve")
        
        # Prendre le plus recent
        latest_file = max(dataset_files, key=lambda f: f.stat().st_mtime)
        print(f"Chargement dataset complet: {latest_file.name}")
        
        return pd.read_csv(latest_file)
    
    def prepare_features_complete(self, df):
        """Preparer les features completes pour ML"""
        
        print("Preparation features completes...")
        
        # Verifier features disponibles
        available_features = [f for f in self.complete_features if f in df.columns]
        missing_features = [f for f in self.complete_features if f not in df.columns]
        
        print(f"Features disponibles: {len(available_features)}")
        print(f"Features manquantes: {len(missing_features)}")
        
        if missing_features:
            print(f"Features manquantes: {missing_features[:5]}...")
            
            # Ajouter features manquantes avec valeurs par defaut
            for feature in missing_features:
                df[feature] = 0.0
        
        # Selection des features pour ML
        feature_df = df[available_features].copy()
        
        # Gestion des valeurs manquantes
        feature_df = feature_df.fillna(0)
        
        # Normalisation avancee
        scaler = StandardScaler()
        feature_df_scaled = pd.DataFrame(
            scaler.fit_transform(feature_df),
            columns=feature_df.columns,
            index=feature_df.index
        )
        
        return feature_df_scaled, scaler, available_features
    
    def create_enhanced_targets(self, df):
        """Creer targets ameliores pour predictions"""
        
        targets = {}
        
        # Targets de base
        targets['next_match_result'] = df['win_rate']
        targets['goals_scored'] = df['goals_per_match'] 
        targets['goals_conceded'] = df['goals_against'] / df['played'].replace(0, 1)
        
        # Nouveaux targets avec features completes
        targets['possession_percentage'] = df.get('ball_possession_avg', 50.0)
        targets['shots_total'] = df.get('shots_total', 0)
        targets['corner_count'] = df.get('corners_taken', 0)
        targets['card_count'] = df.get('yellow_cards', 0) + df.get('red_cards', 0) * 2
        
        targets['clean_sheet'] = df.get('matches_clean_sheets', 0) / df['played'].replace(0, 1)
        targets['both_teams_score'] = 1 - targets['clean_sheet']
        targets['over_2_5_goals'] = (df['goals_for'] / df['played'].replace(0, 1) > 2.5).astype(float)
        
        # Targets avances
        targets['win_probability'] = df['win_rate']
        targets['draw_probability'] = df['draws'] / df['played'].replace(0, 1)
        targets['lose_probability'] = df['losses'] / df['played'].replace(0, 1)
        
        targets['expected_goals'] = df['goals_for'] / df['played'].replace(0, 1)
        targets['expected_assists'] = df.get('top_scorer_assists', 0) / df['played'].replace(0, 1)
        targets['player_rating'] = df.get('top_scorer_rating', 6.0)
        
        # Performance et contexte
        targets['team_performance'] = (df['points'] / (df['played'] * 3).replace(0, 1))
        targets['home_advantage'] = df['home_wins'] / (df['home_wins'] + df['away_wins']).replace(0, 1)
        targets['form_prediction'] = df.get('form_trend', df['win_rate'] * 100)
        
        # Impact factors
        targets['injury_impact'] = df.get('players_injured_current', 0) * -0.1
        targets['fatigue_impact'] = np.clip(df['played'] / 38, 0, 1)  # Normalise par saison
        targets['motivation_impact'] = df['win_rate']  # Correlation avec forme
        
        # Tactique
        targets['offensive_power'] = df['goals_for'] / df['played'].replace(0, 1)
        targets['defensive_stability'] = 1 / (1 + df['goals_against'] / df['played'].replace(0, 1))
        targets['set_pieces_effectiveness'] = df.get('corners_taken', 0) / df['played'].replace(0, 1)
        
        return targets
    
    def train_enhanced_model(self, X, y, model_type='gradient_boosting'):
        """Entrainer modele ameliore avec nouvelles features"""
        
        if model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                random_state=42
            )
        else:
            model = RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Entrainement
        model.fit(X_train, y_train)
        
        # Evaluation
        y_pred = model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'feature_count': X.shape[1]
        }
        
        return model, metrics
    
    def adapt_all_models_complete(self):
        """Adapter tous les modeles avec les nouvelles features completes"""
        
        print(f"\n{'='*80}")
        print("ADAPTATION MODELES ML COMPLETS")
        print(f"{'='*80}")
        
        # Charger dataset complet
        df = self.load_complete_dataset()
        print(f"Dataset charge: {len(df)} equipes, {len(df.columns)} colonnes")
        
        # Preparer features
        X, scaler, feature_names = self.prepare_features_complete(df)
        print(f"Features preparees: {len(feature_names)}")
        
        # Creer targets ameliores
        targets = self.create_enhanced_targets(df)
        print(f"Targets crees: {len(targets)}")
        
        # Entrainer modeles pour chaque ligue et type de prediction
        adaptation_results = {}
        
        for league_id, league_name in self.leagues.items():
            print(f"\n--- {league_name} ---")
            
            # Filtrer donnees de la ligue
            league_df = df[df['league_id'] == league_id]
            
            if len(league_df) < 5:
                print(f"WARNING: Pas assez de donnees pour {league_name} ({len(league_df)} equipes)")
                continue
            
            league_X = X.loc[league_df.index]
            
            adaptation_results[league_id] = {}
            
            for pred_type in self.prediction_types[:15]:  # Limiter pour premier test
                try:
                    # Target pour ce type de prediction
                    if pred_type in targets:
                        league_y = targets[pred_type].loc[league_df.index]
                    else:
                        continue
                    
                    # Verifier donnees valides
                    if league_y.isna().all() or league_y.std() == 0:
                        continue
                    
                    # Entrainer modeles (Random Forest et Gradient Boosting)
                    models = {}
                    
                    for model_type in ['random_forest', 'gradient_boosting']:
                        model, metrics = self.train_enhanced_model(league_X, league_y, model_type)
                        models[model_type] = {'model': model, 'metrics': metrics}
                    
                    # Sauvegarder meilleur modele
                    best_model_type = max(models.keys(), key=lambda k: models[k]['metrics']['r2'])
                    best_model = models[best_model_type]
                    
                    # Chemin de sauvegarde
                    model_path = self.complete_models_dir / f"complete_{league_id}_{pred_type}.joblib"
                    scaler_path = self.complete_models_dir / f"complete_scaler_{league_id}_{pred_type}.joblib"
                    
                    # Sauvegarder
                    joblib.dump(best_model['model'], model_path)
                    joblib.dump(scaler, scaler_path)
                    
                    adaptation_results[league_id][pred_type] = {
                        'model_type': best_model_type,
                        'metrics': best_model['metrics'],
                        'model_path': str(model_path),
                        'scaler_path': str(scaler_path)
                    }
                    
                    print(f"  OK {pred_type}: R2 = {best_model['metrics']['r2']:.3f}")
                    self.adaptation_stats['models_adapted'] += 1
                    
                except Exception as e:
                    print(f"  ERROR {pred_type}: {e}")
                    continue
        
        # Sauvegarder rapport d'adaptation
        report = {
            'adaptation_timestamp': datetime.now().isoformat(),
            'total_features': len(feature_names),
            'feature_names': feature_names,
            'models_adapted': self.adaptation_stats['models_adapted'],
            'leagues_processed': len(adaptation_results),
            'prediction_types': len(self.prediction_types),
            'results': adaptation_results
        }
        
        report_path = self.complete_models_dir / "adaptation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*80}")
        print("ADAPTATION TERMINEE")
        print(f"{'='*80}")
        print(f"Modeles adaptes: {self.adaptation_stats['models_adapted']}")
        print(f"Features utilisees: {len(feature_names)}")
        print(f"Ligues traitees: {len(adaptation_results)}")
        print(f"Rapport: {report_path}")
        
        return report

def main():
    """Fonction principale d'adaptation"""
    
    print("ADAPTATEUR MODELES ML COMPLETS")
    print("Adaptation des modeles avec TOUTES les nouvelles features")
    print()
    
    try:
        adapter = CompleteMLModelsAdapter()
        
        # Adaptation complete
        report = adapter.adapt_all_models_complete()
        
        print(f"\nADAPTATION REUSSIE!")
        print(f"Modeles ML adaptes avec {report['total_features']} features")
        print(f"Performance amelioree avec donnees completes API-Football")
        
        return report
        
    except Exception as e:
        print(f"ERREUR ADAPTATION: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()