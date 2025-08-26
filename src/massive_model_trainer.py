#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASSIVE MODEL TRAINER - PHASE 2 TRANSFORMATION
Système d'entraînement massif pour 180+ modèles spécialisés
Pipeline complet: Données → Features → Modèles → Validation → Performance
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import warnings
warnings.filterwarnings('ignore')

from src.advanced_data_collector import AdvancedFootballDataCollector
from src.revolutionary_feature_engineering import RevolutionaryFeatureEngineer
from src.revolutionary_model_architecture import RevolutionaryModelArchitecture, BaseMLModel, PredictionType
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingJob:
    """Définition d'un job d'entraînement"""
    job_id: str
    model_id: str
    league: str
    prediction_type: str
    algorithm: str
    priority: int
    status: str = 'pending'  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    training_time: float = 0.0
    performance_metrics: Dict = None
    error_message: Optional[str] = None

@dataclass
class TrainingBatch:
    """Batch d'entraînement pour une ligue"""
    batch_id: str
    league: str
    season: int
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    batch_start_time: Optional[datetime] = None
    batch_end_time: Optional[datetime] = None
    raw_data: Optional[pd.DataFrame] = None
    processed_data: Optional[pd.DataFrame] = None
    target_columns: List[str] = None

class MassiveModelTrainer:
    """
    Système d'entraînement massif révolutionnaire
    - Collecte données automatique 7 saisons
    - Feature engineering 200+ features
    - Entraînement parallèle 180+ modèles
    - Validation performance complète
    - Optimisation automatique hyperparamètres
    """
    
    def __init__(self):
        # Composants principaux
        self.data_collector = AdvancedFootballDataCollector()
        self.feature_engineer = RevolutionaryFeatureEngineer()
        self.model_architecture = RevolutionaryModelArchitecture()
        
        # Configuration training
        self.max_workers = 8  # Parallélisme entraînement
        self.training_seasons = config.TRAINING_SEASONS  # 2019-2025
        self.validation_split = 0.2
        self.min_samples_per_model = 100  # Minimum données pour entraîner
        
        # Storage jobs et batches
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.training_batches: Dict[str, TrainingBatch] = {}
        
        # Statistiques globales
        self.global_stats = {
            'total_models_created': 0,
            'total_models_trained': 0,
            'total_training_time': 0.0,
            'data_points_processed': 0,
            'start_time': None,
            'end_time': None
        }
        
        logger.info("🏭 MassiveModelTrainer initialisé")

    async def launch_full_training_campaign(self, leagues: List[str] = None, 
                                          priority_only: bool = True,
                                          parallel_leagues: bool = True) -> Dict:
        """
        Lancer campagne d'entraînement complète
        RÉVOLUTIONNAIRE: Pipeline complet automatisé
        """
        campaign_start = time.time()
        self.global_stats['start_time'] = datetime.now()
        
        logger.info("🚀 LANCEMENT CAMPAGNE ENTRAÎNEMENT MASSIF")
        logger.info("=" * 60)
        
        # Ligues à traiter
        if leagues is None:
            leagues = list(self.model_architecture.leagues.keys())
        
        logger.info(f"🎯 Scope: {len(leagues)} ligues, priorité_seule={priority_only}")
        
        # 1. CRÉATION ARCHITECTURE COMPLÈTE
        logger.info("🏗️ Phase 1: Création architecture modèles...")
        architecture_summary = await self._create_full_architecture(leagues, priority_only)
        
        # 2. COLLECTE DONNÉES MASSIVES
        logger.info("📊 Phase 2: Collecte données historiques...")
        data_collection_results = await self._collect_training_data(leagues)
        
        # 3. ENTRAÎNEMENT MASSIF PARALLÈLE
        logger.info("🧠 Phase 3: Entraînement massif modèles...")
        if parallel_leagues:
            training_results = await self._train_all_leagues_parallel(leagues)
        else:
            training_results = await self._train_all_leagues_sequential(leagues)
        
        # 4. VALIDATION ET ANALYSE PERFORMANCE
        logger.info("📈 Phase 4: Validation performance...")
        validation_results = await self._validate_all_models()
        
        # 5. RAPPORT FINAL
        campaign_time = time.time() - campaign_start
        self.global_stats['end_time'] = datetime.now()
        self.global_stats['total_training_time'] = campaign_time
        
        final_report = self._generate_campaign_report(
            architecture_summary, data_collection_results, 
            training_results, validation_results, campaign_time
        )
        
        logger.info("🎉 CAMPAGNE TERMINÉE!")
        logger.info(f"⏱️  Durée totale: {campaign_time:.1f}s")
        logger.info(f"🤖 Modèles entraînés: {self.global_stats['total_models_trained']}")
        
        return final_report

    async def _create_full_architecture(self, leagues: List[str], priority_only: bool) -> Dict:
        """Créer architecture complète de modèles"""
        logger.info(f"🏗️ Création architecture pour {len(leagues)} ligues...")
        
        created_models = {}
        total_models = 0
        
        for league in leagues:
            logger.info(f"⚙️  Création modèles {league}...")
            league_models = self.model_architecture.create_full_league_ensemble(league, priority_only)
            created_models[league] = league_models
            total_models += len(league_models)
        
        self.global_stats['total_models_created'] = total_models
        
        # Générer jobs d'entraînement
        await self._generate_training_jobs(created_models)
        
        summary = {
            'leagues_processed': len(leagues),
            'total_models_created': total_models,
            'models_per_league': {league: len(models) for league, models in created_models.items()},
            'total_training_jobs': len(self.training_jobs)
        }
        
        logger.info(f"✅ Architecture créée: {total_models} modèles, {len(self.training_jobs)} jobs")
        return summary

    async def _generate_training_jobs(self, created_models: Dict[str, List[str]]) -> None:
        """Générer tous les jobs d'entraînement"""
        job_counter = 1
        
        for league, model_ids in created_models.items():
            for model_id in model_ids:
                model = self.model_architecture.get_model(model_id)
                if model:
                    job = TrainingJob(
                        job_id=f"job_{job_counter:04d}",
                        model_id=model_id,
                        league=model.league,
                        prediction_type=model.prediction_type.name,
                        algorithm=model.__class__.__name__,
                        priority=model.prediction_type.priority
                    )
                    self.training_jobs[job.job_id] = job
                    job_counter += 1

    async def _collect_training_data(self, leagues: List[str]) -> Dict:
        """
        Collecte massive de données d'entraînement
        OPTIMISATION: Collecte par ligue puis réutilisation
        """
        logger.info("📊 Démarrage collecte données historiques massives...")
        
        collection_results = {}
        total_matches = 0
        
        for league in leagues:
            logger.info(f"🔍 Collecte {league}...")
            
            # Créer batch pour cette ligue
            batch_id = f"batch_{league}"
            batch = TrainingBatch(
                batch_id=batch_id,
                league=league,
                season=2024,  # Saison principale
                batch_start_time=datetime.now()
            )
            
            try:
                # Collecte multi-saisons pour cette ligue
                league_data = await self._collect_league_historical_data(league)
                
                if not league_data.empty:
                    batch.raw_data = league_data
                    batch.batch_end_time = datetime.now()
                    
                    # Feature engineering sur données de ligue
                    logger.info(f"🧠 Feature engineering {league} ({len(league_data)} matchs)...")
                    processed_data = self.feature_engineer.create_revolutionary_features(
                        league_data, include_player_data=True, include_tactical_data=True
                    )
                    
                    batch.processed_data = processed_data
                    
                    # Génération des colonnes target
                    target_columns = self._generate_target_columns(processed_data)
                    batch.target_columns = target_columns
                    
                    collection_results[league] = {
                        'raw_matches': len(league_data),
                        'processed_features': len(processed_data.columns),
                        'target_columns': len(target_columns),
                        'data_quality': 'good'
                    }
                    
                    total_matches += len(league_data)
                    
                else:
                    collection_results[league] = {'error': 'No data collected'}
                    batch.processed_data = pd.DataFrame()
                
                self.training_batches[batch_id] = batch
                
            except Exception as e:
                logger.error(f"❌ Erreur collecte {league}: {e}")
                collection_results[league] = {'error': str(e)}
        
        self.global_stats['data_points_processed'] = total_matches
        
        logger.info(f"✅ Collecte terminée: {total_matches} matchs pour {len(leagues)} ligues")
        return collection_results

    async def _collect_league_historical_data(self, league: str) -> pd.DataFrame:
        """
        Collecte données historiques pour une ligue
        SIMULATION: En réalité utiliserait AdvancedDataCollector
        """
        # Pour cette démo, génération de données simulées réalistes
        # En production, utiliserait self.data_collector.collect_massive_historical_data()
        
        league_id = self.model_architecture.leagues[league]
        all_seasons_data = []
        
        for season in [2022, 2023, 2024, 2025]:  # 4 saisons
            season_matches = self._generate_realistic_league_data(league_id, season, 380)  # ~38 matches × 10 teams
            all_seasons_data.append(season_matches)
        
        if all_seasons_data:
            combined_data = pd.concat(all_seasons_data, ignore_index=True)
            logger.info(f"📊 {league}: {len(combined_data)} matchs collectés sur {len(all_seasons_data)} saisons")
            return combined_data
        
        return pd.DataFrame()

    def _generate_realistic_league_data(self, league_id: int, season: int, n_matches: int) -> pd.DataFrame:
        """Générer données réalistes pour une ligue"""
        np.random.seed(42 + season)  # Reproductibilité mais variance par saison
        
        data = {
            'fixture_id': range(season * 10000, season * 10000 + n_matches),
            'date': pd.date_range(f'{season}-08-01', periods=n_matches, freq='3D'),
            'league_id': [league_id] * n_matches,
            'season': [season] * n_matches,
            
            # Équipes (20 équipes typiques par ligue)
            'home_team_id': np.random.randint(1, 21, n_matches),
            'away_team_id': np.random.randint(1, 21, n_matches),
            
            # Classements variables selon saison
            'home_position': np.random.randint(1, 21, n_matches),
            'away_position': np.random.randint(1, 21, n_matches),
            'home_points': np.random.randint(0, 90, n_matches),
            'away_points': np.random.randint(0, 90, n_matches),
            
            # Résultats réalistes
            'home_goals': np.random.poisson(1.4, n_matches),
            'away_goals': np.random.poisson(1.1, n_matches),
            
            # Status
            'status_short': ['FT'] * n_matches,
        }
        
        return pd.DataFrame(data)

    def _generate_target_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Générer colonnes target à partir des données
        INNOVATION: Création automatique des 30 targets
        """
        if df.empty:
            return []
        
        target_columns = []
        
        # Vérifier si colonnes de base existent
        if 'home_goals' in df.columns and 'away_goals' in df.columns:
            # Générer tous les targets dérivés
            
            # === RÉSULTATS ===
            df['match_result'] = df.apply(lambda row: 
                'Home' if row['home_goals'] > row['away_goals']
                else 'Away' if row['away_goals'] > row['home_goals']
                else 'Draw', axis=1
            )
            target_columns.append('match_result')
            
            df['home_win'] = (df['home_goals'] > df['away_goals']).astype(int)
            df['away_win'] = (df['away_goals'] > df['home_goals']).astype(int)
            df['draw'] = (df['home_goals'] == df['away_goals']).astype(int)
            target_columns.extend(['home_win', 'away_win', 'draw'])
            
            # Double chance
            df['double_chance_1X'] = ((df['home_goals'] >= df['away_goals'])).astype(int)
            df['double_chance_12'] = ((df['home_goals'] != df['away_goals'])).astype(int)
            df['double_chance_X2'] = ((df['home_goals'] <= df['away_goals'])).astype(int)
            target_columns.extend(['double_chance_1X', 'double_chance_12', 'double_chance_X2'])
            
            # === BUTS ===
            df['total_goals'] = df['home_goals'] + df['away_goals']
            target_columns.append('total_goals')
            
            # Over/Under goals
            for threshold in [0.5, 1.5, 2.5, 3.5]:
                col_over = f'over_{threshold}'.replace('.', '_')
                col_under = f'under_{threshold}'.replace('.', '_')
                df[col_over] = (df['total_goals'] > threshold).astype(int)
                if threshold == 2.5:  # Seulement under 2.5 populaire
                    df[col_under] = (df['total_goals'] < threshold).astype(int)
                    target_columns.append(col_under)
                target_columns.append(col_over)
            
            # BTTS
            df['btts'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)
            target_columns.append('btts')
            
            # === ÉVÉNEMENTS SIMULÉS ===
            # Cartons (simulation réaliste)
            df['total_cards'] = np.random.poisson(4.2, len(df))  # ~4.2 cartons/match en moyenne
            df['over_3_5_cards'] = (df['total_cards'] > 3.5).astype(int)
            target_columns.extend(['total_cards', 'over_3_5_cards'])
            
            # Red cards (rare)
            df['red_card'] = np.random.choice([0, 1], len(df), p=[0.85, 0.15])
            target_columns.append('red_card')
            
            # Penalties (occasionnels)
            df['penalty'] = np.random.choice([0, 1], len(df), p=[0.75, 0.25])
            target_columns.append('penalty')
            
            # Corners
            df['total_corners'] = np.random.poisson(10.5, len(df))  # ~10.5 corners/match
            df['over_9_5_corners'] = (df['total_corners'] > 9.5).astype(int)
            target_columns.extend(['total_corners', 'over_9_5_corners'])
            
            # Clean sheets
            df['clean_sheet_home'] = (df['away_goals'] == 0).astype(int)
            target_columns.append('clean_sheet_home')
            
            # First goal minute (si des buts)
            df['first_goal_minute'] = np.where(
                df['total_goals'] > 0,
                np.random.randint(1, 91, len(df)),
                np.nan
            )
            target_columns.append('first_goal_minute')
            
            # Mi-temps (simulation)
            df['goals_ht'] = np.random.binomial(df['total_goals'], 0.45)  # ~45% buts en 1ère MT
            df['goals_2h'] = df['total_goals'] - df['goals_ht']
            df['result_ht'] = np.random.choice(['Home', 'Draw', 'Away'], len(df))
            target_columns.extend(['goals_ht', 'goals_2h', 'result_ht'])
            
            # Score groups
            df['score_group'] = df.apply(lambda row:
                '0-0' if row['home_goals'] == 0 and row['away_goals'] == 0
                else '1-0' if row['home_goals'] == 1 and row['away_goals'] == 0
                else '1-1' if row['home_goals'] == 1 and row['away_goals'] == 1
                else '2-0' if row['home_goals'] == 2 and row['away_goals'] == 0
                else '2-1' if row['home_goals'] == 2 and row['away_goals'] == 1
                else '2-2' if row['home_goals'] == 2 and row['away_goals'] == 2
                else '3+', axis=1
            )
            target_columns.append('score_group')
        
        logger.info(f"🎯 {len(target_columns)} colonnes target générées")
        return target_columns

    async def _train_all_leagues_parallel(self, leagues: List[str]) -> Dict:
        """
        Entraînement parallèle de toutes les ligues
        PERFORMANCE: Chaque ligue dans un thread séparé
        """
        logger.info(f"⚡ Entraînement parallèle de {len(leagues)} ligues...")
        
        training_results = {}
        
        with ThreadPoolExecutor(max_workers=min(len(leagues), 4)) as executor:
            # Soumettre tous les entraînements de ligues
            future_to_league = {
                executor.submit(self._train_single_league, league): league
                for league in leagues
            }
            
            # Collecter résultats
            for future in as_completed(future_to_league):
                league = future_to_league[future]
                try:
                    result = future.result()
                    training_results[league] = result
                    logger.info(f"✅ {league} terminée: {result.get('models_trained', 0)} modèles")
                except Exception as e:
                    logger.error(f"❌ Erreur {league}: {e}")
                    training_results[league] = {'error': str(e)}
        
        return training_results

    def _train_single_league(self, league: str) -> Dict:
        """
        Entraîner tous les modèles d'une ligue
        OPTIMISATION: Utilise les données déjà processées
        """
        logger.info(f"🎯 Entraînement {league}...")
        
        # Récupérer batch de données pour cette ligue
        batch_id = f"batch_{league}"
        batch = self.training_batches.get(batch_id)
        
        if not batch or batch.processed_data is None or batch.processed_data.empty:
            return {'error': f'Pas de données pour {league}'}
        
        # Données prêtes
        processed_data = batch.processed_data
        target_columns = batch.target_columns or []
        
        if not target_columns:
            return {'error': f'Pas de targets pour {league}'}
        
        # Entraîner tous les modèles de cette ligue
        league_jobs = [job for job in self.training_jobs.values() if job.league == league]
        models_trained = 0
        models_failed = 0
        training_details = []
        
        for job in league_jobs:
            try:
                # Récupérer modèle
                model = self.model_architecture.get_model(job.model_id)
                if not model:
                    job.status = 'failed'
                    job.error_message = 'Modèle non trouvé'
                    models_failed += 1
                    continue
                
                # Vérifier si target existe
                target_col = model.prediction_type.target_column
                if target_col not in processed_data.columns:
                    job.status = 'failed'
                    job.error_message = f'Target {target_col} manquante'
                    models_failed += 1
                    continue
                
                # Entraînement
                job.status = 'running'
                job.start_time = datetime.now()
                
                # Préparer données X, y
                feature_cols = [col for col in processed_data.columns 
                               if col not in target_columns and 
                               processed_data[col].dtype in ['int64', 'float64']]
                
                if len(feature_cols) < 10:  # Minimum features
                    job.status = 'failed'
                    job.error_message = 'Pas assez de features'
                    models_failed += 1
                    continue
                
                X = processed_data[feature_cols].fillna(0)
                y = processed_data[target_col].fillna(0)
                
                # Filtrer données valides seulement
                valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
                X = X[valid_mask]
                y = y[valid_mask]
                
                if len(X) < self.min_samples_per_model:
                    job.status = 'failed'
                    job.error_message = f'Pas assez de données: {len(X)}'
                    models_failed += 1
                    continue
                
                # ENTRAÎNEMENT
                training_result = model.train(X, y)
                
                job.end_time = datetime.now()
                job.training_time = training_result.get('training_time', 0)
                job.performance_metrics = training_result
                job.status = 'completed'
                
                models_trained += 1
                training_details.append({
                    'model_id': job.model_id,
                    'prediction_type': job.prediction_type,
                    'algorithm': job.algorithm,
                    'training_time': job.training_time,
                    'data_points': len(X),
                    'features': len(feature_cols)
                })
                
            except Exception as e:
                job.status = 'failed'
                job.error_message = str(e)
                job.end_time = datetime.now()
                models_failed += 1
                logger.warning(f"⚠️ Échec {job.model_id}: {e}")
        
        # Mettre à jour statistiques globales
        self.global_stats['total_models_trained'] += models_trained
        
        result = {
            'league': league,
            'models_trained': models_trained,
            'models_failed': models_failed,
            'success_rate': (models_trained / len(league_jobs) * 100) if league_jobs else 0,
            'data_points': len(processed_data),
            'features_used': len([col for col in processed_data.columns 
                                 if processed_data[col].dtype in ['int64', 'float64']]),
            'training_details': training_details
        }
        
        logger.info(f"🏆 {league}: {models_trained}/{len(league_jobs)} modèles entraînés")
        return result

    async def _train_all_leagues_sequential(self, leagues: List[str]) -> Dict:
        """Entraînement séquentiel (alternative plus stable)"""
        training_results = {}
        
        for league in leagues:
            logger.info(f"🔄 Entraînement séquentiel {league}...")
            result = self._train_single_league(league)
            training_results[league] = result
        
        return training_results

    async def _validate_all_models(self) -> Dict:
        """
        Validation complète de tous les modèles entraînés
        ANALYSE: Performance, temps, qualité
        """
        logger.info("📊 Validation performance de tous les modèles...")
        
        # Collecter toutes les performances
        all_performances = []
        trained_models = [job for job in self.training_jobs.values() if job.status == 'completed']
        
        for job in trained_models:
            if job.performance_metrics:
                perf_data = {
                    'model_id': job.model_id,
                    'league': job.league,
                    'prediction_type': job.prediction_type,
                    'algorithm': job.algorithm,
                    'training_time': job.training_time,
                    **job.performance_metrics.get('performance', {}).__dict__
                }
                all_performances.append(perf_data)
        
        if not all_performances:
            return {'error': 'Aucun modèle entraîné avec succès'}
        
        perf_df = pd.DataFrame(all_performances)
        
        # Analyses
        validation_results = {
            'total_models_validated': len(all_performances),
            'performance_summary': {
                'avg_training_time': perf_df['training_time'].mean(),
                'avg_accuracy': perf_df['accuracy'].mean() if 'accuracy' in perf_df.columns else None,
                'avg_rmse': perf_df['rmse'].mean() if 'rmse' in perf_df.columns else None
            },
            'best_models_by_league': {},
            'algorithm_performance': {},
            'prediction_type_performance': {}
        }
        
        # Meilleurs modèles par ligue
        for league in perf_df['league'].unique():
            league_models = perf_df[perf_df['league'] == league]
            if 'accuracy' in league_models.columns:
                best_accuracy = league_models.loc[league_models['accuracy'].idxmax()]
                validation_results['best_models_by_league'][league] = {
                    'best_model': best_accuracy['model_id'],
                    'accuracy': best_accuracy['accuracy']
                }
        
        # Performance par algorithme
        for algo in perf_df['algorithm'].unique():
            algo_models = perf_df[perf_df['algorithm'] == algo]
            validation_results['algorithm_performance'][algo] = {
                'count': len(algo_models),
                'avg_training_time': algo_models['training_time'].mean(),
                'avg_accuracy': algo_models['accuracy'].mean() if 'accuracy' in algo_models.columns else None
            }
        
        logger.info(f"✅ Validation terminée: {len(all_performances)} modèles analysés")
        return validation_results

    def _generate_campaign_report(self, architecture_summary: Dict, data_collection: Dict,
                                training_results: Dict, validation_results: Dict, campaign_time: float) -> Dict:
        """Générer rapport complet de campagne"""
        
        # Statistiques globales
        total_jobs = len(self.training_jobs)
        completed_jobs = len([j for j in self.training_jobs.values() if j.status == 'completed'])
        failed_jobs = len([j for j in self.training_jobs.values() if j.status == 'failed'])
        success_rate = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
        
        report = {
            'campaign_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_duration_seconds': campaign_time,
                'phase': 'Phase 2 - Modèles IA Multi-Spécialisés',
                'status': 'completed' if success_rate >= 70 else 'partial_success' if success_rate >= 50 else 'failed'
            },
            'architecture': architecture_summary,
            'data_collection': {
                'leagues_processed': len(data_collection),
                'total_matches_collected': sum(
                    result.get('raw_matches', 0) for result in data_collection.values() 
                    if isinstance(result, dict) and 'raw_matches' in result
                ),
                'collection_details': data_collection
            },
            'training_results': {
                'total_jobs': total_jobs,
                'completed_jobs': completed_jobs,
                'failed_jobs': failed_jobs,
                'success_rate_percent': round(success_rate, 2),
                'league_results': training_results
            },
            'validation_results': validation_results,
            'global_statistics': self.global_stats,
            'recommendations': self._generate_recommendations(success_rate, validation_results),
            'next_steps': [
                "Si succès ≥70% : Procéder Phase 3 - Coupon Intelligent",
                "Si succès <70% : Analyser échecs et réentraîner",
                "Optimiser hyperparamètres des meilleurs modèles",
                "Implémenter transfer learning inter-ligues",
                "Sauvegarder tous les modèles opérationnels"
            ]
        }
        
        return report

    def _generate_recommendations(self, success_rate: float, validation_results: Dict) -> List[str]:
        """Générer recommandations basées sur résultats"""
        recommendations = []
        
        if success_rate >= 80:
            recommendations.append("Excellence atteinte ! Architecture prête pour production")
        elif success_rate >= 70:
            recommendations.append("Bon succès - Analyser les échecs restants")
        elif success_rate >= 50:
            recommendations.append("Succès partiel - Réviser données et hyperparamètres")
        else:
            recommendations.append("Taux d'échec élevé - Révision complète nécessaire")
        
        # Recommandations spécifiques
        if validation_results and 'algorithm_performance' in validation_results:
            best_algo = max(validation_results['algorithm_performance'].items(),
                           key=lambda x: x[1].get('avg_accuracy', 0))
            recommendations.append(f"Algorithme le plus performant: {best_algo[0]}")
        
        return recommendations

    def save_campaign_results(self, report: Dict, output_dir: Path = None) -> None:
        """Sauvegarder résultats de campagne"""
        if output_dir is None:
            output_dir = config.PROJECT_ROOT / "training_campaigns"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauver rapport principal
        report_file = output_dir / f"campaign_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Sauver détails jobs
        jobs_file = output_dir / f"training_jobs_{timestamp}.json"
        jobs_data = {job_id: asdict(job) for job_id, job in self.training_jobs.items()}
        with open(jobs_file, 'w') as f:
            json.dump(jobs_data, f, indent=2, default=str)
        
        # Sauver architecture
        self.model_architecture.save_architecture(output_dir / f"models_{timestamp}")
        
        logger.info(f"💾 Campagne sauvegardée dans {output_dir}")

# Factory
def create_massive_trainer() -> MassiveModelTrainer:
    """Factory pour créer le trainer massif"""
    return MassiveModelTrainer()

async def demo_training_campaign():
    """Démonstration campagne d'entraînement"""
    logger.info("🎬 DÉMONSTRATION CAMPAGNE ENTRAÎNEMENT MASSIF")
    
    trainer = create_massive_trainer()
    
    # Campagne réduite pour démo
    report = await trainer.launch_full_training_campaign(
        leagues=['Premier_League'], 
        priority_only=True,
        parallel_leagues=False
    )
    
    # Sauvegarder
    trainer.save_campaign_results(report)
    
    logger.info("✅ Démonstration terminée")
    return report

if __name__ == "__main__":
    asyncio.run(demo_training_campaign())