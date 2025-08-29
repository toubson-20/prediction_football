"""
PRÉDICTEUR BUNDESLIGA HYBRIDE FINAL
Combine stratification intelligente + fallback robuste
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class BundesligaHybridPredictor:
    """Prédicteur hybride final pour la Bundesliga"""
    
    def __init__(self):
        self.models_dir = Path("models/bundesliga_hybrid")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path("data/ultra_processed")
        
        self.models = {}
        self.performance = {}
        
    def load_data(self):
        """Charger données Bundesliga"""
        df = pd.read_csv(self.data_dir / "complete_ml_dataset_20250826_213205.csv")
        bundesliga = df[df['league_id'] == 78].copy()
        print(f"Bundesliga: {len(bundesliga)} équipes")
        return bundesliga
    
    def create_smart_features(self, df):
        """Features intelligentes spécifiques Bundesliga"""
        enhanced = df.copy()
        
        # Features de variance (clé pour Bundesliga)
        enhanced['goals_volatility'] = np.abs(enhanced['goals_for'] - enhanced['goals_per_match'] * enhanced['played'])
        enhanced['performance_range'] = enhanced['wins'] + enhanced['losses'] - enhanced['draws'] * 0.5
        
        # Features de niveau adaptatifs
        points_median = enhanced['points'].median()
        enhanced['above_median_team'] = (enhanced['points'] > points_median).astype(int)
        
        # Features de contexte temporel
        enhanced['recent_form_weighted'] = (
            enhanced['last_5_matches_wins'] * 0.6 + 
            enhanced['form_trend'] * 0.4
        )
        
        # Features stabilisées
        enhanced['stable_goals_rate'] = enhanced['goals_per_match'] * (enhanced['recent_form_weighted'] / 5 + 0.5)
        enhanced['defensive_reliability'] = enhanced['matches_clean_sheets'] / (enhanced['played'] + 1)
        
        return enhanced
    
    def train_hybrid_system(self, df):
        """Entraîner système hybride"""
        print("\n=== SYSTÈME HYBRIDE BUNDESLIGA ===")
        
        # Préparer features
        enhanced_df = self.create_smart_features(df)
        
        # Features sélectionnées empiriquement
        key_features = [
            'points', 'win_rate', 'goals_per_match', 'goal_diff',
            'shots_total', 'shots_on_goal', 'ball_possession_avg',
            'yellow_cards', 'corners_taken', 'passes_accuracy_pct',
            'top_scorer_goals', 'last_5_matches_wins', 'form_trend',
            'matches_clean_sheets', 'matches_failed_to_score',
            # Features spéciales
            'goals_volatility', 'performance_range', 'above_median_team',
            'recent_form_weighted', 'stable_goals_rate', 'defensive_reliability'
        ]
        
        # Filtrer features disponibles
        available_features = [f for f in key_features if f in enhanced_df.columns]
        X = enhanced_df[available_features].fillna(0)
        
        # Target avec stabilisation
        goals_raw = (enhanced_df['goals_for'] / enhanced_df['played']).fillna(0)
        
        print(f"Features: {len(available_features)}")
        print(f"Goals/match: {goals_raw.mean():.2f} ± {goals_raw.std():.2f}")
        
        # Multiple approches
        approaches = {
            'robust_log': {
                'y': np.log1p(goals_raw),
                'scaler': RobustScaler(),
                'model': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
            },
            'robust_sqrt': {
                'y': np.sqrt(goals_raw + 0.1),
                'scaler': RobustScaler(), 
                'model': RandomForestRegressor(n_estimators=80, max_depth=5, random_state=42)
            },
            'median_resilient': {
                'y': np.clip(goals_raw, 0, np.percentile(goals_raw, 90)),  # Clip outliers
                'scaler': RobustScaler(),
                'model': GradientBoostingRegressor(n_estimators=80, learning_rate=0.15, max_depth=3, random_state=42)
            }
        }
        
        results = {}
        
        for name, approach in approaches.items():
            print(f"\n--- {name.upper()} ---")
            
            # Préparer données
            y_transformed = approach['y']
            scaler = approach['scaler']
            X_scaled = scaler.fit_transform(X)
            
            # Entraîner
            model = approach['model']
            model.fit(X_scaled, y_transformed)
            
            # Prédire
            y_pred_transformed = model.predict(X_scaled)
            
            # Transformer back
            if name == 'robust_log':
                y_pred_orig = np.expm1(y_pred_transformed)
            elif name == 'robust_sqrt':
                y_pred_orig = np.square(y_pred_transformed) - 0.1
            else:  # median_resilient
                y_pred_orig = y_pred_transformed
            
            # Évaluer
            r2_orig = r2_score(goals_raw, y_pred_orig)
            mae_orig = mean_absolute_error(goals_raw, y_pred_orig)
            
            print(f"R²: {r2_orig:.3f}")
            print(f"MAE: {mae_orig:.3f} buts")
            
            # Sauvegarder si correct
            if r2_orig > -0.5:  # Seuil acceptable
                model_path = self.models_dir / f"bundesliga_{name}_model.joblib"
                scaler_path = self.models_dir / f"bundesliga_{name}_scaler.joblib"
                
                joblib.dump(model, model_path)
                joblib.dump(scaler, scaler_path)
                
                self.models[name] = {
                    'model': model,
                    'scaler': scaler,
                    'features': available_features,
                    'transform_type': name,
                    'r2': r2_orig,
                    'mae': mae_orig
                }
                
                results[name] = {
                    'r2': r2_orig,
                    'mae': mae_orig,
                    'model_path': str(model_path)
                }
        
        # Système de vote pondéré
        if len(results) > 1:
            self.create_weighted_ensemble(results)
        
        return results
    
    def create_weighted_ensemble(self, results):
        """Créer ensemble pondéré des meilleurs modèles"""
        print(f"\n--- ENSEMBLE PONDÉRÉ ---")
        
        # Calculer poids basés sur performance
        weights = {}
        total_weight = 0
        
        for name, perf in results.items():
            if perf['r2'] > 0:
                # Poids = R² normalisé
                weight = max(perf['r2'], 0.1) / perf['mae']
                weights[name] = weight
                total_weight += weight
        
        # Normaliser poids
        for name in weights:
            weights[name] /= total_weight
        
        print("Poids des modèles:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.3f}")
        
        self.models['ensemble'] = {
            'weights': weights,
            'type': 'weighted_ensemble'
        }
    
    def predict_goals_hybrid(self, team_data):
        """Prédiction hybride intelligente"""
        if not self.models:
            return {"error": "Modèles non entraînés"}
        
        # Préparer données équipe
        team_df = pd.DataFrame([team_data])
        team_enhanced = self.create_smart_features(team_df)
        
        predictions = {}
        
        # Prédictions individuelles
        for name, model_info in self.models.items():
            if name == 'ensemble':
                continue
                
            try:
                X = team_enhanced[model_info['features']].fillna(0)
                X_scaled = model_info['scaler'].transform(X)
                pred_transformed = model_info['model'].predict(X_scaled)[0]
                
                # Transform back
                if model_info['transform_type'] == 'robust_log':
                    pred_goals = np.expm1(pred_transformed)
                elif model_info['transform_type'] == 'robust_sqrt':
                    pred_goals = np.square(pred_transformed) - 0.1
                else:  # median_resilient
                    pred_goals = pred_transformed
                
                predictions[name] = max(0, pred_goals)  # Pas de buts négatifs
                
            except Exception as e:
                print(f"Erreur {name}: {e}")
                continue
        
        # Prédiction ensemble
        if 'ensemble' in self.models and len(predictions) > 1:
            ensemble_pred = 0
            weights = self.models['ensemble']['weights']
            
            for name, weight in weights.items():
                if name in predictions:
                    ensemble_pred += predictions[name] * weight
            
            predictions['ensemble'] = ensemble_pred
        
        # Sélectionner meilleure prédiction
        best_model = max(self.models.keys() - {'ensemble'}, 
                        key=lambda x: self.models[x].get('r2', -1))
        
        result = {
            'predicted_goals': predictions.get('ensemble', predictions.get(best_model, 0)),
            'individual_predictions': predictions,
            'best_model': best_model,
            'confidence': self.models.get(best_model, {}).get('r2', 0)
        }
        
        return result

def main():
    """Test du système hybride"""
    predictor = BundesligaHybridPredictor()
    
    # Charger données
    df = predictor.load_data()
    
    # Entraîner système
    results = predictor.train_hybrid_system(df)
    
    # Rapport final
    print(f"\n{'='*50}")
    print("SYSTÈME HYBRIDE FINAL - RÉSULTATS")
    print(f"{'='*50}")
    
    best_r2 = max(r['r2'] for r in results.values()) if results else -1
    best_model = max(results.keys(), key=lambda x: results[x]['r2']) if results else None
    
    print(f"Meilleur modèle: {best_model}")
    print(f"Meilleur R²: {best_r2:.3f}")
    
    if best_r2 > 0:
        print(f"SUCCES: R2 positif atteint!")
        improvement = best_r2 - (-0.145)  # vs modèle original
        print(f"Amelioration: +{improvement:.3f} (+{improvement/0.145*100:.1f}%)")
    else:
        print(f"R2 encore negatif, mais systeme robuste en place")
    
    for name, perf in results.items():
        print(f"\n{name}:")
        print(f"  R²: {perf['r2']:.3f}")
        print(f"  MAE: {perf['mae']:.3f} buts")
    
    # Test prédiction exemple
    if results:
        example_team = df.iloc[0].to_dict()
        pred = predictor.predict_goals_hybrid(example_team)
        print(f"\nTest prédiction:")
        print(f"Équipe points: {example_team.get('points', 0)}")
        print(f"Prédiction: {pred['predicted_goals']:.2f} buts/match")

if __name__ == "__main__":
    main()