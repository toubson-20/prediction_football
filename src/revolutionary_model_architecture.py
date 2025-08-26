#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REVOLUTIONARY MODEL ARCHITECTURE - PHASE 2 TRANSFORMATION
Architecture modulaire pour 180+ modèles spécialisés football
6 ligues × 30 prédictions avec IA avancée
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import joblib
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Imports ML avancés
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

try:
    import catboost as cb
except ImportError:
    cb = None
    logger.warning("CatBoost non disponible - Utilisation XGBoost/LightGBM seulement")

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionType:
    """Définition d'un type de prédiction"""
    name: str
    category: str  # 'result', 'goals', 'events', 'players', 'special'
    task_type: str  # 'classification', 'regression'
    target_column: str
    classes: Optional[List] = None  # Pour classification
    min_value: Optional[float] = None  # Pour régression
    max_value: Optional[float] = None
    description: str = ""
    priority: int = 1  # 1=haute, 2=moyenne, 3=basse

@dataclass
class ModelPerformance:
    """Performance d'un modèle"""
    model_id: str
    prediction_type: str
    league: str
    algorithm: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    rmse: float = 0.0
    r2_score: float = 0.0
    training_time: float = 0.0
    prediction_time: float = 0.0
    data_points: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

class BaseMLModel(ABC):
    """Classe de base pour tous les modèles ML"""
    
    def __init__(self, model_id: str, prediction_type: PredictionType, league: str):
        self.model_id = model_id
        self.prediction_type = prediction_type
        self.league = league
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.performance = ModelPerformance(
            model_id=model_id,
            prediction_type=prediction_type.name,
            league=league,
            algorithm=self.__class__.__name__
        )
    
    @abstractmethod
    def create_model(self) -> Any:
        """Créer l'instance du modèle ML"""
        pass
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Entraîner le modèle"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Faire des prédictions"""
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Prédictions de probabilité (classification seulement)"""
        if hasattr(self.model, 'predict_proba') and self.prediction_type.task_type == 'classification':
            return self.model.predict_proba(X)
        return None
    
    def save_model(self, path: Path) -> None:
        """Sauvegarder le modèle"""
        model_data = {
            'model': self.model,
            'model_id': self.model_id,
            'prediction_type': self.prediction_type,
            'league': self.league,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'performance': self.performance
        }
        joblib.dump(model_data, path)
        logger.debug(f"💾 Modèle sauvegardé: {path}")
    
    def load_model(self, path: Path) -> bool:
        """Charger un modèle sauvegardé"""
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.is_trained = model_data['is_trained']
            self.feature_names = model_data['feature_names']
            self.performance = model_data['performance']
            return True
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle {path}: {e}")
            return False

class AdvancedRandomForestModel(BaseMLModel):
    """Random Forest optimisé pour football"""
    
    def create_model(self) -> Union[RandomForestClassifier, RandomForestRegressor]:
        """Créer Random Forest optimisé"""
        params = {
            'n_estimators': 200,  # Plus d'arbres pour stabilité
            'max_depth': 15,      # Profondeur contrôlée
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'random_state': config.RANDOM_STATE,
            'n_jobs': -1,         # Parallélisation
            'class_weight': 'balanced' if self.prediction_type.task_type == 'classification' else None
        }
        
        if self.prediction_type.task_type == 'classification':
            return RandomForestClassifier(**{k: v for k, v in params.items() if k != 'class_weight' or v is not None})
        else:
            return RandomForestRegressor(**{k: v for k, v in params.items() if k != 'class_weight'})
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Entraînement Random Forest avec validation"""
        start_time = datetime.now()
        
        self.model = self.create_model()
        self.feature_names = list(X.columns)
        
        # Entraînement
        self.model.fit(X, y)
        
        # Validation croisée
        cv_scores = cross_val_score(self.model, X, y, cv=5, n_jobs=-1)
        
        # Métriques sur set complet (pour référence)
        y_pred = self.model.predict(X)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Mise à jour performance
        self.performance.training_time = training_time
        self.performance.data_points = len(X)
        
        if self.prediction_type.task_type == 'classification':
            self.performance.accuracy = accuracy_score(y, y_pred)
            self.performance.precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            self.performance.recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            self.performance.f1_score = f1_score(y, y_pred, average='weighted', zero_division=0)
        else:
            self.performance.rmse = np.sqrt(mean_squared_error(y, y_pred))
            self.performance.r2_score = r2_score(y, y_pred)
        
        self.is_trained = True
        
        return {
            'algorithm': 'RandomForest',
            'training_time': training_time,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'performance': self.performance
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédictions Random Forest"""
        if not self.is_trained:
            raise ValueError(f"Modèle {self.model_id} non entraîné")
        return self.model.predict(X)

class AdvancedXGBoostModel(BaseMLModel):
    """XGBoost optimisé pour football"""
    
    def create_model(self) -> Union[xgb.XGBClassifier, xgb.XGBRegressor]:
        """Créer XGBoost optimisé"""
        params = {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,      # L1 regularization
            'reg_lambda': 0.1,     # L2 regularization
            'random_state': config.RANDOM_STATE,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        if self.prediction_type.task_type == 'classification':
            return xgb.XGBClassifier(**params)
        else:
            return xgb.XGBRegressor(**params)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Entraînement XGBoost avec early stopping"""
        start_time = datetime.now()
        
        # Split pour early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y if self.prediction_type.task_type == 'classification' else None
        )
        
        self.model = self.create_model()
        self.feature_names = list(X.columns)
        
        # Entraînement avec early stopping
        eval_set = [(X_val, y_val)]
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Prédictions validation
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Mise à jour performance
        self.performance.training_time = training_time
        self.performance.data_points = len(X)
        
        if self.prediction_type.task_type == 'classification':
            self.performance.accuracy = accuracy_score(y_val, y_pred_val)
            self.performance.precision = precision_score(y_val, y_pred_val, average='weighted', zero_division=0)
            self.performance.recall = recall_score(y_val, y_pred_val, average='weighted', zero_division=0)
            self.performance.f1_score = f1_score(y_val, y_pred_val, average='weighted', zero_division=0)
        else:
            self.performance.rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            self.performance.r2_score = r2_score(y_val, y_pred_val)
        
        self.is_trained = True
        
        return {
            'algorithm': 'XGBoost',
            'training_time': training_time,
            'best_iteration': self.model.best_iteration,
            'performance': self.performance,
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédictions XGBoost"""
        if not self.is_trained:
            raise ValueError(f"Modèle {self.model_id} non entraîné")
        return self.model.predict(X)

class AdvancedLightGBMModel(BaseMLModel):
    """LightGBM optimisé pour football"""
    
    def create_model(self) -> Union[lgb.LGBMClassifier, lgb.LGBMRegressor]:
        """Créer LightGBM optimisé"""
        params = {
            'n_estimators': 500,
            'max_depth': 10,
            'learning_rate': 0.05,
            'num_leaves': 64,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': config.RANDOM_STATE,
            'n_jobs': -1,
            'verbosity': -1
        }
        
        if self.prediction_type.task_type == 'classification':
            return lgb.LGBMClassifier(**params)
        else:
            return lgb.LGBMRegressor(**params)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Entraînement LightGBM avec validation"""
        start_time = datetime.now()
        
        # Split pour validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y if self.prediction_type.task_type == 'classification' else None
        )
        
        self.model = self.create_model()
        self.feature_names = list(X.columns)
        
        # Entraînement avec early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Validation
        y_pred_val = self.model.predict(X_val)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Performance
        self.performance.training_time = training_time
        self.performance.data_points = len(X)
        
        if self.prediction_type.task_type == 'classification':
            self.performance.accuracy = accuracy_score(y_val, y_pred_val)
            self.performance.precision = precision_score(y_val, y_pred_val, average='weighted', zero_division=0)
            self.performance.recall = recall_score(y_val, y_pred_val, average='weighted', zero_division=0)
            self.performance.f1_score = f1_score(y_val, y_pred_val, average='weighted', zero_division=0)
        else:
            self.performance.rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            self.performance.r2_score = r2_score(y_val, y_pred_val)
        
        self.is_trained = True
        
        return {
            'algorithm': 'LightGBM',
            'training_time': training_time,
            'best_iteration': self.model.best_iteration_,
            'performance': self.performance
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédictions LightGBM"""
        if not self.is_trained:
            raise ValueError(f"Modèle {self.model_id} non entraîné")
        return self.model.predict(X)

class RevolutionaryModelArchitecture:
    """
    Architecture de modèles révolutionnaire
    6 ligues × 30 prédictions × 3 algorithmes = 540 modèles potentiels
    Sélection intelligente des meilleurs pour chaque contexte
    """
    
    def __init__(self):
        # Configuration 30 types de prédictions révolutionnaires
        self.prediction_types = self._define_prediction_types()
        
        # Ligues supportées
        self.leagues = {
            'Premier_League': 39,
            'La_Liga': 140, 
            'Ligue_1': 61,
            'Bundesliga': 78,
            'Champions_League': 2,
            'Europa_League': 3
        }
        
        # Algorithmes disponibles
        self.available_algorithms = {
            'RandomForest': AdvancedRandomForestModel,
            'XGBoost': AdvancedXGBoostModel,
            'LightGBM': AdvancedLightGBMModel,
        }
        
        # Storage des modèles
        self.models: Dict[str, BaseMLModel] = {}
        self.league_specialists: Dict[str, Dict[str, List[str]]] = {}  # league -> prediction_type -> [model_ids]
        
        # Performance tracking
        self.performance_history: List[ModelPerformance] = []
        
        logger.info(f"🧠 RevolutionaryModelArchitecture initialisée:")
        logger.info(f"   • {len(self.prediction_types)} types de prédictions")
        logger.info(f"   • {len(self.leagues)} ligues")
        logger.info(f"   • {len(self.available_algorithms)} algorithmes")
        logger.info(f"   • Capacité max: {len(self.prediction_types) * len(self.leagues) * len(self.available_algorithms)} modèles")

    def _define_prediction_types(self) -> Dict[str, PredictionType]:
        """Définir les 30 types de prédictions révolutionnaires"""
        
        predictions = {
            # === RÉSULTATS MATCH (9 prédictions) ===
            'match_result_3way': PredictionType(
                name='match_result_3way', category='result', task_type='classification',
                target_column='match_result', classes=['Home', 'Draw', 'Away'],
                description="Résultat 1X2 classique", priority=1
            ),
            'double_chance_1X': PredictionType(
                name='double_chance_1X', category='result', task_type='classification',
                target_column='double_chance_1X', classes=[0, 1],
                description="Double chance domicile ou nul", priority=2
            ),
            'double_chance_12': PredictionType(
                name='double_chance_12', category='result', task_type='classification',
                target_column='double_chance_12', classes=[0, 1],
                description="Double chance domicile ou extérieur", priority=2
            ),
            'double_chance_X2': PredictionType(
                name='double_chance_X2', category='result', task_type='classification',
                target_column='double_chance_X2', classes=[0, 1],
                description="Double chance nul ou extérieur", priority=2
            ),
            'home_win': PredictionType(
                name='home_win', category='result', task_type='classification',
                target_column='home_win', classes=[0, 1],
                description="Victoire équipe domicile", priority=1
            ),
            'away_win': PredictionType(
                name='away_win', category='result', task_type='classification',
                target_column='away_win', classes=[0, 1],
                description="Victoire équipe extérieur", priority=1
            ),
            'draw': PredictionType(
                name='draw', category='result', task_type='classification',
                target_column='draw', classes=[0, 1],
                description="Match nul", priority=2
            ),
            'home_win_margin': PredictionType(
                name='home_win_margin', category='result', task_type='regression',
                target_column='home_win_margin', min_value=-5, max_value=5,
                description="Marge de victoire domicile", priority=3
            ),
            'result_at_halftime': PredictionType(
                name='result_at_halftime', category='result', task_type='classification',
                target_column='result_ht', classes=['Home', 'Draw', 'Away'],
                description="Résultat à la mi-temps", priority=2
            ),
            
            # === BUTS ET SCORES (12 prédictions) ===
            'total_goals': PredictionType(
                name='total_goals', category='goals', task_type='regression',
                target_column='total_goals', min_value=0, max_value=10,
                description="Total buts du match", priority=1
            ),
            'home_goals': PredictionType(
                name='home_goals', category='goals', task_type='regression',
                target_column='home_goals', min_value=0, max_value=8,
                description="Buts équipe domicile", priority=1
            ),
            'away_goals': PredictionType(
                name='away_goals', category='goals', task_type='regression',
                target_column='away_goals', min_value=0, max_value=8,
                description="Buts équipe extérieur", priority=1
            ),
            'over_0_5_goals': PredictionType(
                name='over_0_5_goals', category='goals', task_type='classification',
                target_column='over_0_5', classes=[0, 1],
                description="Plus de 0.5 buts", priority=3
            ),
            'over_1_5_goals': PredictionType(
                name='over_1_5_goals', category='goals', task_type='classification',
                target_column='over_1_5', classes=[0, 1],
                description="Plus de 1.5 buts", priority=2
            ),
            'over_2_5_goals': PredictionType(
                name='over_2_5_goals', category='goals', task_type='classification',
                target_column='over_2_5', classes=[0, 1],
                description="Plus de 2.5 buts", priority=1
            ),
            'over_3_5_goals': PredictionType(
                name='over_3_5_goals', category='goals', task_type='classification',
                target_column='over_3_5', classes=[0, 1],
                description="Plus de 3.5 buts", priority=2
            ),
            'under_2_5_goals': PredictionType(
                name='under_2_5_goals', category='goals', task_type='classification',
                target_column='under_2_5', classes=[0, 1],
                description="Moins de 2.5 buts", priority=2
            ),
            'both_teams_score': PredictionType(
                name='both_teams_score', category='goals', task_type='classification',
                target_column='btts', classes=[0, 1],
                description="Les deux équipes marquent", priority=1
            ),
            'goals_first_half': PredictionType(
                name='goals_first_half', category='goals', task_type='regression',
                target_column='goals_ht', min_value=0, max_value=6,
                description="Buts première mi-temps", priority=2
            ),
            'goals_second_half': PredictionType(
                name='goals_second_half', category='goals', task_type='regression',
                target_column='goals_2h', min_value=0, max_value=8,
                description="Buts seconde mi-temps", priority=3
            ),
            'exact_score_group': PredictionType(
                name='exact_score_group', category='goals', task_type='classification',
                target_column='score_group', classes=['0-0', '1-0', '1-1', '2-0', '2-1', '2-2', '3+'],
                description="Groupe de score exact", priority=3
            ),
            
            # === CARTONS ET ÉVÉNEMENTS (5 prédictions) ===
            'total_cards': PredictionType(
                name='total_cards', category='events', task_type='regression',
                target_column='total_cards', min_value=0, max_value=12,
                description="Total cartons du match", priority=2
            ),
            'over_3_5_cards': PredictionType(
                name='over_3_5_cards', category='events', task_type='classification',
                target_column='over_3_5_cards', classes=[0, 1],
                description="Plus de 3.5 cartons", priority=3
            ),
            'red_card': PredictionType(
                name='red_card', category='events', task_type='classification',
                target_column='red_card', classes=[0, 1],
                description="Carton rouge dans le match", priority=3
            ),
            'penalty_awarded': PredictionType(
                name='penalty_awarded', category='events', task_type='classification',
                target_column='penalty', classes=[0, 1],
                description="Penalty accordé", priority=3
            ),
            'first_goal_time': PredictionType(
                name='first_goal_time', category='events', task_type='regression',
                target_column='first_goal_minute', min_value=1, max_value=90,
                description="Minute du premier but", priority=3
            ),
            
            # === CORNERS ET STATS (4 prédictions) ===
            'total_corners': PredictionType(
                name='total_corners', category='events', task_type='regression',
                target_column='total_corners', min_value=0, max_value=20,
                description="Total corners du match", priority=2
            ),
            'over_9_5_corners': PredictionType(
                name='over_9_5_corners', category='events', task_type='classification',
                target_column='over_9_5_corners', classes=[0, 1],
                description="Plus de 9.5 corners", priority=3
            ),
            'corner_handicap': PredictionType(
                name='corner_handicap', category='events', task_type='classification',
                target_column='corner_handicap', classes=['Home', 'Away'],
                description="Handicap corners", priority=3
            ),
            'clean_sheet_home': PredictionType(
                name='clean_sheet_home', category='events', task_type='classification',
                target_column='clean_sheet_home', classes=[0, 1],
                description="Clean sheet équipe domicile", priority=2
            )
        }
        
        return predictions

    def create_league_specialist(self, league: str, prediction_type: str, algorithm: str = 'auto') -> str:
        """
        Créer un modèle spécialisé pour une ligue et type de prédiction
        RÉVOLUTIONNAIRE: Sélection automatique du meilleur algorithme
        """
        if league not in self.leagues:
            raise ValueError(f"Ligue non supportée: {league}")
        
        if prediction_type not in self.prediction_types:
            raise ValueError(f"Type prédiction non supporté: {prediction_type}")
        
        # Sélection algorithme automatique ou manuel
        if algorithm == 'auto':
            algorithm = self._select_best_algorithm(prediction_type, league)
        
        if algorithm not in self.available_algorithms:
            raise ValueError(f"Algorithme non supporté: {algorithm}")
        
        # Créer ID unique du modèle
        model_id = f"{league}_{prediction_type}_{algorithm}"
        
        # Créer l'instance du modèle
        model_class = self.available_algorithms[algorithm]
        pred_type = self.prediction_types[prediction_type]
        
        model = model_class(model_id, pred_type, league)
        
        # Stocker le modèle
        self.models[model_id] = model
        
        # Mettre à jour les spécialistes de ligue
        if league not in self.league_specialists:
            self.league_specialists[league] = {}
        if prediction_type not in self.league_specialists[league]:
            self.league_specialists[league][prediction_type] = []
        
        self.league_specialists[league][prediction_type].append(model_id)
        
        logger.info(f"🎯 Modèle spécialisé créé: {model_id}")
        return model_id

    def _select_best_algorithm(self, prediction_type: str, league: str) -> str:
        """
        Sélection intelligente du meilleur algorithme
        INNOVATION: Basée sur le type de prédiction et contexte ligue
        """
        pred_type = self.prediction_types[prediction_type]
        
        # Stratégies par catégorie
        if pred_type.category == 'result':
            if pred_type.task_type == 'classification':
                return 'XGBoost'  # Excellent pour classification multiclasse
            else:
                return 'LightGBM'  # Rapide pour régression
        
        elif pred_type.category == 'goals':
            if 'over_' in prediction_type or 'under_' in prediction_type:
                return 'XGBoost'  # Optimal pour seuils
            elif pred_type.task_type == 'regression':
                return 'LightGBM'  # Précis pour prédiction buts
            else:
                return 'RandomForest'  # Robuste pour classification binaire
        
        elif pred_type.category == 'events':
            return 'RandomForest'  # Bon avec features éparses
        
        else:  # Défaut
            return 'XGBoost'

    def create_full_league_ensemble(self, league: str, priority_only: bool = True) -> List[str]:
        """
        Créer ensemble complet de modèles pour une ligue
        RÉVOLUTIONNAIRE: 30 modèles spécialisés par ligue
        """
        logger.info(f"🏗️ Création ensemble complet pour {league}")
        
        created_models = []
        predictions_to_create = self.prediction_types.items()
        
        # Filtrer par priorité si demandé
        if priority_only:
            predictions_to_create = [(name, pred) for name, pred in predictions_to_create if pred.priority == 1]
            logger.info(f"📊 Création priorité 1 seulement: {len(predictions_to_create)} prédictions")
        
        for pred_name, pred_type in predictions_to_create:
            try:
                model_id = self.create_league_specialist(league, pred_name, 'auto')
                created_models.append(model_id)
            except Exception as e:
                logger.error(f"❌ Erreur création {league}_{pred_name}: {e}")
        
        logger.info(f"✅ {len(created_models)} modèles créés pour {league}")
        return created_models

    def create_all_leagues_ensemble(self, priority_only: bool = True) -> Dict[str, List[str]]:
        """
        Créer tous les ensembles de modèles pour toutes les ligues
        ARCHITECTURE COMPLÈTE: 6 ligues × 30 prédictions
        """
        logger.info("🚀 CRÉATION ARCHITECTURE COMPLÈTE - TOUS MODÈLES")
        
        all_models = {}
        total_created = 0
        
        for league in self.leagues.keys():
            logger.info(f"🏆 Traitement ligue: {league}")
            league_models = self.create_full_league_ensemble(league, priority_only)
            all_models[league] = league_models
            total_created += len(league_models)
        
        logger.info(f"🎉 ARCHITECTURE TERMINÉE:")
        logger.info(f"   • {len(all_models)} ligues équipées")
        logger.info(f"   • {total_created} modèles créés au total")
        logger.info(f"   • Prêt pour entraînement massif")
        
        return all_models

    def get_model(self, model_id: str) -> Optional[BaseMLModel]:
        """Récupérer un modèle par ID"""
        return self.models.get(model_id)

    def get_league_specialists(self, league: str, prediction_type: str) -> List[BaseMLModel]:
        """Récupérer tous les modèles spécialisés pour une ligue et prédiction"""
        model_ids = self.league_specialists.get(league, {}).get(prediction_type, [])
        return [self.models[mid] for mid in model_ids if mid in self.models]

    def get_architecture_summary(self) -> Dict:
        """Résumé complet de l'architecture"""
        # Statistiques par ligue
        league_stats = {}
        for league, specialists in self.league_specialists.items():
            total_models = sum(len(models) for models in specialists.values())
            trained_models = sum(
                1 for model_list in specialists.values() 
                for model_id in model_list 
                if self.models[model_id].is_trained
            )
            league_stats[league] = {
                'total_models': total_models,
                'trained_models': trained_models,
                'predictions_covered': len(specialists)
            }
        
        # Statistiques par algorithme
        algo_stats = {}
        for model_id, model in self.models.items():
            algo = model.__class__.__name__
            if algo not in algo_stats:
                algo_stats[algo] = {'total': 0, 'trained': 0}
            algo_stats[algo]['total'] += 1
            if model.is_trained:
                algo_stats[algo]['trained'] += 1
        
        return {
            'architecture_version': '2.0.0-Revolutionary',
            'total_models': len(self.models),
            'total_leagues': len(self.leagues),
            'total_prediction_types': len(self.prediction_types),
            'trained_models': sum(1 for m in self.models.values() if m.is_trained),
            'league_statistics': league_stats,
            'algorithm_statistics': algo_stats,
            'available_algorithms': list(self.available_algorithms.keys()),
            'supported_leagues': list(self.leagues.keys()),
            'prediction_categories': {
                cat: len([p for p in self.prediction_types.values() if p.category == cat])
                for cat in set(p.category for p in self.prediction_types.values())
            }
        }

    def save_architecture(self, output_dir: Path = None) -> None:
        """Sauvegarder toute l'architecture"""
        if output_dir is None:
            output_dir = config.MODELS_DIR / "revolutionary_architecture"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder chaque modèle
        for model_id, model in self.models.items():
            model_file = output_dir / f"{model_id}.joblib"
            model.save_model(model_file)
        
        # Sauvegarder métadonnées architecture
        metadata = {
            'architecture_summary': self.get_architecture_summary(),
            'league_specialists': self.league_specialists,
            'prediction_types': {name: {
                'name': pt.name,
                'category': pt.category,
                'task_type': pt.task_type,
                'target_column': pt.target_column,
                'classes': pt.classes,
                'description': pt.description,
                'priority': pt.priority
            } for name, pt in self.prediction_types.items()},
            'leagues': self.leagues,
            'save_timestamp': datetime.now().isoformat()
        }
        
        metadata_file = output_dir / "architecture_metadata.joblib"
        joblib.dump(metadata, metadata_file)
        
        logger.info(f"💾 Architecture sauvegardée dans {output_dir}")
        logger.info(f"   • {len(self.models)} modèles")
        logger.info(f"   • Métadonnées complètes")

# Factory et utilitaires
def create_revolutionary_architecture() -> RevolutionaryModelArchitecture:
    """Factory pour créer l'architecture révolutionnaire"""
    return RevolutionaryModelArchitecture()

if __name__ == "__main__":
    # Démonstration architecture
    logger.info("🧪 Test RevolutionaryModelArchitecture")
    
    # Créer architecture
    arch = create_revolutionary_architecture()
    
    # Créer modèles pour une ligue (test)
    test_models = arch.create_full_league_ensemble('Premier_League', priority_only=True)
    
    logger.info(f"✅ Test réussi:")
    logger.info(f"   • Architecture créée")
    logger.info(f"   • {len(test_models)} modèles Premier League")
    
    # Résumé
    summary = arch.get_architecture_summary()
    logger.info(f"📊 Résumé: {summary['total_models']} modèles, {summary['total_prediction_types']} types")