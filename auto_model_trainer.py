"""
RE-ENTRAINEMENT AUTOMATIQUE DES MODELES ML
Ré-entraîne tous les modèles avec les nouvelles données API quotidiennes
"""

import asyncio
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import pytz

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from config import Config

class AutoModelTrainer:
    """Système de ré-entraînement automatique des modèles ML"""
    
    def __init__(self):
        self.paris_tz = pytz.timezone(Config.TIMEZONE_PARIS)
        
        # Chemins
        self.ml_data_dir = Path("data/ml_ready")
        self.models_dir = Path("models/complete_models")
        self.backup_dir = Path("models/backups")
        
        # Créer dossiers
        for dir_path in [self.models_dir, self.backup_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Configuration ML
        self.model_types = [
            'next_match_result',
            'goals_scored', 
            'both_teams_score',
            'win_probability',
            'over_2_5_goals'
        ]
        
        self.leagues = Config.TARGET_LEAGUES
        
        # Logging
        self.setup_logging()
    
    def setup_logging(self):
        """Configuration logging"""
        log_file = Path("logs/auto_model_trainer.log")
        log_file.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_latest_dataset(self) -> pd.DataFrame:
        """Charger le dernier dataset ML"""
        # Priorité au dataset statistiques d'équipes
        team_stats_file = self.ml_data_dir / "team_statistics_ml_dataset.csv"
        combined_file = self.ml_data_dir / "combined_ml_dataset.csv"
        latest_file = self.ml_data_dir / "latest_ml_dataset.csv"

        if team_stats_file.exists():
            df = pd.read_csv(team_stats_file)
            self.logger.info(f"Dataset statistiques équipes chargé: {len(df)} lignes, {len(df.columns)} colonnes")
            return df
        elif combined_file.exists():
            df = pd.read_csv(combined_file)
            self.logger.info(f"Dataset combiné chargé: {len(df)} lignes, {len(df.columns)} colonnes")
            return df
        elif latest_file.exists():
            df = pd.read_csv(latest_file)
            self.logger.info(f"Dataset latest chargé: {len(df)} lignes, {len(df.columns)} colonnes")
            return df
        else:
            raise FileNotFoundError(f"Aucun dataset ML trouvé dans {self.ml_data_dir}")

        return df
    
    def backup_existing_models(self):
        """Sauvegarder les modèles existants"""
        backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_session_dir = self.backup_dir / f"backup_{backup_timestamp}"
        backup_session_dir.mkdir(exist_ok=True)
        
        model_files = list(self.models_dir.glob("*.joblib"))
        
        for model_file in model_files:
            backup_file = backup_session_dir / model_file.name
            if model_file.exists():
                import shutil
                shutil.copy2(model_file, backup_file)
        
        self.logger.info(f"Sauvegarde modèles: {len(model_files)} fichiers -> {backup_session_dir}")
        return backup_session_dir
    
    def prepare_training_data(self, df: pd.DataFrame, league_id: int, target_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Préparer données d'entraînement pour un modèle spécifique"""
        # Filtrer par ligue
        league_data = df[df['league_id'] == league_id].copy()
        
        if len(league_data) < 10:
            raise ValueError(f"Pas assez de données pour ligue {league_id}: {len(league_data)} équipes")
        
        # Créer les targets selon le type de modèle
        if target_type == 'next_match_result':
            # Prédire probabilité victoire (0-1)
            league_data['target'] = league_data['win_rate']
            
        elif target_type == 'goals_scored':
            # Prédire buts marqués par match
            league_data['target'] = league_data['avg_goals_for']
            
        elif target_type == 'both_teams_score':
            # Prédire probabilité BTS (basé sur buts pour/contre)
            league_data['target'] = np.minimum(1.0, 
                (league_data['avg_goals_for'] + league_data['avg_goals_against']) / 4.0)
            
        elif target_type == 'win_probability':
            # Même que next_match_result
            league_data['target'] = league_data['win_rate']
            
        elif target_type == 'over_2_5_goals':
            # Prédire probabilité +2.5 buts
            league_data['target'] = np.minimum(1.0, 
                (league_data['avg_goals_for'] + league_data['avg_goals_against']) / 3.0)
        
        # Sélectionner features (exclure metadata et target)
        feature_cols = [col for col in league_data.columns 
                       if col not in ['team_id', 'team_name', 'league_id', 'season', 
                                     'update_date', 'update_timestamp', 'target']]
        
        X = league_data[feature_cols].fillna(0)
        y = league_data['target'].fillna(0)
        
        # Assurer 53 features exactement
        if len(X.columns) < 53:
            # Ajouter features padding
            for i in range(53 - len(X.columns)):
                X[f'padding_feature_{i}'] = 0.0
        elif len(X.columns) > 53:
            # Garder seulement les 53 premières
            X = X.iloc[:, :53]
        
        self.logger.info(f"  Features préparées: {X.shape}, Target: {y.shape}")
        
        return X.values, y.values
    
    def train_model(self, X: np.ndarray, y: np.ndarray, model_type: str) -> Tuple[object, object, Dict]:
        """Entraîner un modèle ML"""
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=Config.RANDOM_STATE
        )
        
        # Scaler
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Choisir algorithme selon type de prédiction
        if model_type in ['next_match_result', 'win_probability', 'both_teams_score', 'over_2_5_goals']:
            # Classification/probabilité
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=Config.RANDOM_STATE
            )
        else:
            # Régression (goals_scored)
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=Config.RANDOM_STATE
            )
        
        # Entraînement
        model.fit(X_train_scaled, y_train)
        
        # Évaluation
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'r2': r2,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'training_date': datetime.now(self.paris_tz).isoformat()
        }
        
        return model, scaler, metrics
    
    async def retrain_league_models(self, df: pd.DataFrame, league_id: int, league_name: str):
        """Ré-entraîner tous les modèles d'une ligue"""
        self.logger.info(f"Ré-entraînement modèles {league_name} (ID: {league_id})")
        
        for model_type in self.model_types:
            try:
                self.logger.info(f"  Modèle: {model_type}")
                
                # Préparer données
                X, y = self.prepare_training_data(df, league_id, model_type)
                
                if len(X) < 5:
                    self.logger.warning(f"    Pas assez de données: {len(X)} échantillons")
                    continue
                
                # Entraîner modèle
                model, scaler, metrics = self.train_model(X, y, model_type)
                
                # Sauvegarder modèle
                model_file = self.models_dir / f"complete_{league_id}_{model_type}.joblib"
                scaler_file = self.models_dir / f"complete_scaler_{league_id}_{model_type}.joblib"
                
                joblib.dump(model, model_file)
                joblib.dump(scaler, scaler_file)
                
                # Sauvegarder métriques
                metrics_file = self.models_dir / f"metrics_{league_id}_{model_type}.json"
                with open(metrics_file, 'w') as f:
                    import json
                    json.dump(metrics, f, indent=2)
                
                self.logger.info(f"    Entraîné: R² = {metrics['r2']:.3f}, MSE = {metrics['mse']:.3f}")
                
            except Exception as e:
                self.logger.error(f"    Erreur {model_type}: {e}")
    
    async def run_full_retraining(self):
        """Ré-entraîner tous les modèles avec données fraîches"""
        start_time = datetime.now(self.paris_tz)
        self.logger.info(f"=== DEBUT RE-ENTRAINEMENT COMPLET - {start_time.strftime('%d/%m/%Y %H:%M')} ===")
        
        try:
            # 1. Sauvegarder modèles existants
            backup_dir = self.backup_existing_models()
            
            # 2. Charger dernier dataset
            df = self.load_latest_dataset()
            
            # 3. Ré-entraîner par ligue
            for league_name, league_id in self.leagues.items():
                await self.retrain_league_models(df, league_id, league_name)
                
                # Petit délai entre ligues
                await asyncio.sleep(1)
            
            end_time = datetime.now(self.paris_tz)
            duration = end_time - start_time
            
            self.logger.info(f"=== RE-ENTRAINEMENT TERMINE - Durée: {duration} ===")
            self.logger.info(f"Modèles sauvegardés dans: {backup_dir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur ré-entraînement: {e}")
            raise


async def main():
    """Fonction principale"""
    trainer = AutoModelTrainer()
    
    try:
        success = await trainer.run_full_retraining()
        if success:
            print("Ré-entraînement terminé avec succès")
        else:
            print("Ré-entraînement échoué")
            
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())