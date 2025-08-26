#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REALTIME DATA PIPELINE - PHASE 1 TRANSFORMATION
Pipeline intelligent pour données pré-match temps réel
Intégration AdvancedDataCollector + RevolutionaryFeatureEngineering
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time

from src.advanced_data_collector import AdvancedFootballDataCollector
from src.revolutionary_feature_engineering import RevolutionaryFeatureEngineer
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PreMatchData:
    """Structure pour données pré-match"""
    fixture_id: int
    collection_time: datetime
    minutes_before_kickoff: int
    
    # Données de base
    home_team_id: int
    away_team_id: int
    league_id: int
    season: int
    kickoff_time: datetime
    
    # Données enrichies
    lineups_confirmed: bool = False
    weather_data: Dict = None
    betting_odds: Dict = None
    injury_updates: Dict = None
    team_news: Dict = None
    
    # Features générées
    features_generated: bool = False
    feature_vector: Dict = None
    data_quality_score: float = 0.0

@dataclass 
class PipelineStats:
    """Statistiques du pipeline"""
    total_fixtures_processed: int = 0
    successful_collections: int = 0
    failed_collections: int = 0
    avg_processing_time_ms: float = 0.0
    features_generated_count: int = 0
    last_update: datetime = None

class RealtimeDataPipeline:
    """
    Pipeline de données temps réel révolutionnaire
    - Collecte automatique selon timeline pré-match
    - Feature engineering intelligent temps réel
    - Validation qualité continue
    - Callbacks pour déclenchement prédictions
    """
    
    def __init__(self):
        # Composants principaux
        self.data_collector = AdvancedFootballDataCollector()
        self.feature_engineer = RevolutionaryFeatureEngineer()
        
        # Configuration pipeline
        self.collection_timelines = {
            90: "early_data",      # 90+ min : données préliminaires
            60: "confirmed_lineups", # 60+ min : compositions confirmées
            30: "betting_odds",    # 30+ min : cotes finales
            15: "final_updates",   # 15+ min : dernières mises à jour
            0: "kickoff_ready"     # 0 min : prêt pour prédiction
        }
        
        # Storage et cache
        self.active_fixtures: Dict[int, PreMatchData] = {}
        self.processed_data: Dict[int, Dict] = {}
        self.callbacks: List[Callable] = []
        
        # Statistiques
        self.stats = PipelineStats()
        
        # Configuration temps réel
        self.monitoring_interval = 300  # 5 minutes
        self.is_monitoring = False
        
        logger.info("⚡ RealtimeDataPipeline initialisé")

    def register_callback(self, callback: Callable[[int, Dict], None]) -> None:
        """
        Enregistrer un callback pour notifications temps réel
        Le callback sera appelé avec (fixture_id, processed_data)
        """
        self.callbacks.append(callback)
        logger.info(f"📞 Callback enregistré - Total: {len(self.callbacks)}")

    async def start_monitoring(self, fixtures_to_monitor: List[Dict]) -> None:
        """
        Démarrer le monitoring temps réel pour les fixtures
        RÉVOLUTIONNAIRE: Surveillance automatique multi-fixtures
        """
        logger.info(f"🎯 Démarrage monitoring pour {len(fixtures_to_monitor)} fixtures")
        
        # Initialiser les fixtures à surveiller
        for fixture_info in fixtures_to_monitor:
            fixture_id = fixture_info['fixture_id']
            kickoff_time = pd.to_datetime(fixture_info['kickoff_time'])
            
            pre_match_data = PreMatchData(
                fixture_id=fixture_id,
                collection_time=datetime.now(),
                minutes_before_kickoff=self._calculate_minutes_before(kickoff_time),
                home_team_id=fixture_info['home_team_id'],
                away_team_id=fixture_info['away_team_id'],
                league_id=fixture_info['league_id'],
                season=fixture_info['season'],
                kickoff_time=kickoff_time
            )
            
            self.active_fixtures[fixture_id] = pre_match_data
        
        # Lancer la surveillance en boucle
        self.is_monitoring = True
        await self._monitoring_loop()

    async def _monitoring_loop(self) -> None:
        """Boucle principale de monitoring"""
        logger.info("🔄 Boucle monitoring démarrée")
        
        while self.is_monitoring:
            current_time = datetime.now()
            fixtures_to_remove = []
            
            for fixture_id, pre_match_data in self.active_fixtures.items():
                try:
                    # Calculer temps restant
                    minutes_before = self._calculate_minutes_before(pre_match_data.kickoff_time)
                    
                    # Si match déjà commencé, arrêter surveillance
                    if minutes_before < 0:
                        fixtures_to_remove.append(fixture_id)
                        continue
                    
                    # Vérifier si besoin de collecter selon timeline
                    should_collect = self._should_collect_now(minutes_before, pre_match_data)
                    
                    if should_collect:
                        logger.info(f"📊 Collection données fixture {fixture_id} ({minutes_before}min avant)")
                        await self._process_fixture(fixture_id, minutes_before)
                
                except Exception as e:
                    logger.error(f"❌ Erreur monitoring fixture {fixture_id}: {e}")
            
            # Nettoyer fixtures terminés
            for fixture_id in fixtures_to_remove:
                del self.active_fixtures[fixture_id]
                logger.info(f"✅ Monitoring terminé pour fixture {fixture_id}")
            
            # Si plus de fixtures, arrêter monitoring
            if not self.active_fixtures:
                self.is_monitoring = False
                logger.info("🏁 Monitoring terminé - Plus de fixtures actives")
                break
            
            # Attendre avant prochaine vérification
            await asyncio.sleep(self.monitoring_interval)

    def _calculate_minutes_before(self, kickoff_time: datetime) -> int:
        """Calculer minutes avant coup d'envoi"""
        now = datetime.now()
        delta = kickoff_time - now
        return int(delta.total_seconds() / 60)

    def _should_collect_now(self, minutes_before: int, pre_match_data: PreMatchData) -> bool:
        """
        Déterminer si on doit collecter maintenant selon timeline
        INTELLIGENT: Évite collectes redondantes
        """
        # Identifier la timeline actuelle
        current_timeline = None
        for threshold in sorted(self.collection_timelines.keys(), reverse=True):
            if minutes_before >= threshold:
                current_timeline = threshold
                break
        
        if current_timeline is None:
            return False
        
        # Vérifier si on a déjà collecté pour cette timeline
        last_collection_minutes = getattr(pre_match_data, 'last_collection_minutes', 999)
        
        # Collecter si on entre dans une nouvelle timeline
        return minutes_before <= current_timeline and last_collection_minutes > current_timeline

    async def _process_fixture(self, fixture_id: int, minutes_before: int) -> None:
        """
        Traitement complet d'un fixture
        PIPELINE: Collecte → Features → Validation → Callbacks
        """
        start_time = time.time()
        pre_match_data = self.active_fixtures[fixture_id]
        
        try:
            # 1. COLLECTE DONNÉES TEMPS RÉEL
            logger.debug(f"📥 Collecte données fixture {fixture_id}")
            raw_data = self.data_collector.get_pre_match_data(fixture_id, minutes_before)
            
            if not raw_data:
                logger.warning(f"⚠️ Aucune donnée pour fixture {fixture_id}")
                return
            
            # 2. ENRICHISSEMENT AVEC DONNÉES HISTORIQUES
            logger.debug(f"🔍 Enrichissement données historiques")
            enriched_data = await self._enrich_with_historical_data(raw_data, pre_match_data)
            
            # 3. GÉNÉRATION FEATURES RÉVOLUTIONNAIRES  
            logger.debug(f"🧠 Génération features")
            features_df = self._generate_realtime_features(enriched_data, pre_match_data)
            
            # 4. VALIDATION QUALITÉ
            logger.debug(f"✅ Validation qualité")
            quality_score = self._validate_data_quality(features_df)
            
            # 5. MISE À JOUR STRUCTURE
            pre_match_data.collection_time = datetime.now()
            pre_match_data.minutes_before_kickoff = minutes_before
            pre_match_data.features_generated = True
            pre_match_data.feature_vector = features_df.to_dict('records')[0] if not features_df.empty else {}
            pre_match_data.data_quality_score = quality_score
            pre_match_data.last_collection_minutes = minutes_before  # Track dernière collecte
            
            # 6. SAUVEGARDE
            self.processed_data[fixture_id] = {
                'fixture_id': fixture_id,
                'processing_time': datetime.now().isoformat(),
                'minutes_before': minutes_before,
                'raw_data': raw_data,
                'features': pre_match_data.feature_vector,
                'quality_score': quality_score
            }
            
            # 7. CALLBACKS NOTIFICATIONS
            await self._notify_callbacks(fixture_id, self.processed_data[fixture_id])
            
            # 8. STATISTIQUES
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_stats(processing_time_ms, success=True)
            
            logger.info(f"✅ Fixture {fixture_id} traité en {processing_time_ms:.1f}ms (qualité: {quality_score:.2f})")
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement fixture {fixture_id}: {e}")
            self._update_stats(0, success=False)

    async def _enrich_with_historical_data(self, raw_data: Dict, pre_match_data: PreMatchData) -> pd.DataFrame:
        """
        Enrichir avec données historiques nécessaires aux features
        INNOVATION: Collecte intelligente données manquantes
        """
        # Créer DataFrame de base
        match_row = {
            'fixture_id': pre_match_data.fixture_id,
            'home_team_id': pre_match_data.home_team_id,
            'away_team_id': pre_match_data.away_team_id,
            'league_id': pre_match_data.league_id,
            'season': pre_match_data.season,
            'date': pre_match_data.kickoff_time.isoformat(),
            'minutes_before_kickoff': pre_match_data.minutes_before_kickoff
        }
        
        # Ajouter données temps réel collectées
        match_row.update(raw_data)
        
        # Enrichir avec données historiques si nécessaire
        try:
            # Classements actuels
            home_standings = self.data_collector.get_standings(pre_match_data.league_id, pre_match_data.season)
            away_standings = home_standings  # Même ligue
            
            if not home_standings.empty:
                home_team_standing = home_standings[home_standings['team_id'] == pre_match_data.home_team_id]
                away_team_standing = home_standings[home_standings['team_id'] == pre_match_data.away_team_id]
                
                # Ajouter stats classement
                for prefix, standing in [('home', home_team_standing), ('away', away_team_standing)]:
                    if not standing.empty:
                        for col in ['position', 'points', 'played', 'won', 'draw', 'lost', 'goalsFor', 'goalsAgainst']:
                            if col in standing.columns:
                                match_row[f'{prefix}_{col}'] = standing[col].iloc[0]
            
            # Statistiques équipes complètes  
            home_stats = self.data_collector.get_team_comprehensive_stats(
                pre_match_data.home_team_id, pre_match_data.league_id, pre_match_data.season
            )
            away_stats = self.data_collector.get_team_comprehensive_stats(
                pre_match_data.away_team_id, pre_match_data.league_id, pre_match_data.season
            )
            
            # Ajouter stats équipes
            for prefix, stats in [('home', home_stats), ('away', away_stats)]:
                for key, value in stats.items():
                    if not key.endswith('_id'):  # Éviter doublons IDs
                        match_row[f'{prefix}_{key}'] = value
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur enrichissement historique: {e}")
        
        return pd.DataFrame([match_row])

    def _generate_realtime_features(self, enriched_data: pd.DataFrame, pre_match_data: PreMatchData) -> pd.DataFrame:
        """
        Générer features avec le RevolutionaryFeatureEngineer
        TEMPS RÉEL: Features optimisées pour prédictions live
        """
        try:
            # Génération features complètes
            features_df = self.feature_engineer.create_revolutionary_features(
                enriched_data,
                include_player_data=True,
                include_tactical_data=True
            )
            
            # Ajout de features spécifiques temps réel
            features_df['realtime_collection'] = True
            features_df['minutes_before_kickoff'] = pre_match_data.minutes_before_kickoff
            features_df['data_freshness_score'] = self._calculate_data_freshness(pre_match_data)
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Erreur génération features: {e}")
            return enriched_data  # Fallback

    def _calculate_data_freshness(self, pre_match_data: PreMatchData) -> float:
        """
        Calculer score de fraîcheur des données
        INNOVATION: Score basé sur recence des données
        """
        minutes_before = pre_match_data.minutes_before_kickoff
        
        # Score plus élevé = données plus fraîches/fiables
        if minutes_before <= 15:
            return 1.0  # Données très fraîches
        elif minutes_before <= 30:
            return 0.9
        elif minutes_before <= 60:
            return 0.8
        elif minutes_before <= 90:
            return 0.7
        else:
            return 0.6  # Données préliminaires

    def _validate_data_quality(self, df: pd.DataFrame) -> float:
        """
        Validation qualité pour données temps réel
        SCORE: 0-1 basé sur complétude et cohérence
        """
        if df.empty:
            return 0.0
        
        # Métriques qualité
        total_cols = len(df.columns)
        missing_values = df.isnull().sum().sum()
        completeness = 1 - (missing_values / (len(df) * total_cols))
        
        # Vérifications cohérence basiques
        coherence_score = 1.0
        
        # Scores négatifs impossibles
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        negative_scores = df[numeric_cols].lt(0).sum().sum()
        if negative_scores > 0:
            coherence_score -= 0.1
        
        # Score final pondéré
        quality_score = (completeness * 0.7) + (coherence_score * 0.3)
        
        return min(max(quality_score, 0.0), 1.0)

    async def _notify_callbacks(self, fixture_id: int, processed_data: Dict) -> None:
        """Notifier tous les callbacks enregistrés"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(fixture_id, processed_data)
                else:
                    callback(fixture_id, processed_data)
            except Exception as e:
                logger.error(f"❌ Erreur callback: {e}")

    def _update_stats(self, processing_time_ms: float, success: bool) -> None:
        """Mettre à jour statistiques pipeline"""
        self.stats.total_fixtures_processed += 1
        
        if success:
            self.stats.successful_collections += 1
            self.stats.features_generated_count += 1
        else:
            self.stats.failed_collections += 1
        
        # Moyenne mobile du temps de traitement
        if self.stats.avg_processing_time_ms == 0:
            self.stats.avg_processing_time_ms = processing_time_ms
        else:
            # Moyenne pondérée (nouveau = 30%, ancien = 70%)
            self.stats.avg_processing_time_ms = (
                self.stats.avg_processing_time_ms * 0.7 + processing_time_ms * 0.3
            )
        
        self.stats.last_update = datetime.now()

    def get_fixture_status(self, fixture_id: int) -> Optional[Dict]:
        """Obtenir le statut d'un fixture"""
        if fixture_id not in self.active_fixtures:
            return None
        
        pre_match_data = self.active_fixtures[fixture_id]
        return {
            'fixture_id': fixture_id,
            'minutes_before_kickoff': self._calculate_minutes_before(pre_match_data.kickoff_time),
            'data_quality_score': pre_match_data.data_quality_score,
            'features_generated': pre_match_data.features_generated,
            'last_collection': pre_match_data.collection_time.isoformat() if pre_match_data.collection_time else None
        }

    def get_processed_data(self, fixture_id: int) -> Optional[Dict]:
        """Récupérer données traitées pour un fixture"""
        return self.processed_data.get(fixture_id)

    def get_pipeline_stats(self) -> Dict:
        """Statistiques du pipeline"""
        return asdict(self.stats)

    async def stop_monitoring(self) -> None:
        """Arrêter le monitoring"""
        logger.info("🛑 Arrêt monitoring demandé")
        self.is_monitoring = False

    def save_processed_data(self, output_dir: Path = None) -> None:
        """Sauvegarder toutes les données traitées"""
        if output_dir is None:
            output_dir = config.PROCESSED_DATA_DIR / "realtime"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for fixture_id, data in self.processed_data.items():
            file_path = output_dir / f"fixture_{fixture_id}_processed.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        logger.info(f"💾 {len(self.processed_data)} fixtures sauvegardés dans {output_dir}")

# Factory et utilitaires
def create_realtime_pipeline() -> RealtimeDataPipeline:
    """Factory pour créer le pipeline temps réel"""
    return RealtimeDataPipeline()

async def demo_pipeline():
    """Démonstration du pipeline temps réel"""
    logger.info("🎬 Démonstration Pipeline Temps Réel")
    
    # Fixtures de test
    test_fixtures = [
        {
            'fixture_id': 12345,
            'home_team_id': 33,  # Manchester United
            'away_team_id': 34,  # Arsenal
            'league_id': 39,     # Premier League
            'season': 2025,
            'kickoff_time': (datetime.now() + timedelta(minutes=75)).isoformat()
        }
    ]
    
    # Callback de test
    def test_callback(fixture_id: int, processed_data: Dict):
        logger.info(f"📞 Callback: Fixture {fixture_id} traité (qualité: {processed_data.get('quality_score', 0):.2f})")
    
    # Créer et démarrer pipeline
    pipeline = create_realtime_pipeline()
    pipeline.register_callback(test_callback)
    
    # Monitoring (arrêt automatique après démo)
    await pipeline.start_monitoring(test_fixtures)
    
    logger.info("✅ Démonstration terminée")

if __name__ == "__main__":
    # Test du pipeline
    asyncio.run(demo_pipeline())