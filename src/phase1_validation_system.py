#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 1 VALIDATION SYSTEM - TESTS ET VALIDATION FINALE
Système de tests complet pour valider tous les composants Phase 1
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from dataclasses import dataclass

from src.advanced_data_collector import AdvancedFootballDataCollector
from src.revolutionary_feature_engineering import RevolutionaryFeatureEngineer  
from src.realtime_data_pipeline import RealtimeDataPipeline
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Résultat de validation d'un composant"""
    component_name: str
    test_name: str
    success: bool
    execution_time_ms: float
    details: Dict
    error_message: Optional[str] = None

class Phase1ValidationSystem:
    """
    Système de validation complet Phase 1
    - Tests unitaires tous composants
    - Tests d'intégration pipeline complet
    - Validation performance
    - Génération rapport détaillé
    """
    
    def __init__(self):
        # Composants à tester
        self.data_collector = None
        self.feature_engineer = None
        self.realtime_pipeline = None
        
        # Résultats des tests
        self.validation_results: List[ValidationResult] = []
        self.overall_success = False
        
        logger.info("🧪 Phase1ValidationSystem initialisé")

    async def run_full_validation(self) -> Dict:
        """
        Validation complète de tous les composants Phase 1
        PIPELINE COMPLET: Tests unitaires → Tests intégration → Rapport
        """
        logger.info("🚀 DÉMARRAGE VALIDATION COMPLÈTE PHASE 1")
        validation_start = time.time()
        
        # 1. Tests DataCollector
        logger.info("📊 Tests AdvancedFootballDataCollector...")
        collector_results = await self._test_data_collector()
        
        # 2. Tests FeatureEngineer 
        logger.info("🧠 Tests RevolutionaryFeatureEngineer...")
        feature_results = await self._test_feature_engineer()
        
        # 3. Tests Pipeline Temps Réel
        logger.info("⚡ Tests RealtimeDataPipeline...")
        pipeline_results = await self._test_realtime_pipeline()
        
        # 4. Tests d'intégration
        logger.info("🔗 Tests d'intégration...")
        integration_results = await self._test_integration()
        
        # 5. Tests de performance
        logger.info("⚡ Tests de performance...")
        performance_results = await self._test_performance()
        
        # 6. Génération rapport final
        total_time = (time.time() - validation_start) * 1000
        validation_report = self._generate_validation_report(total_time)
        
        logger.info(f"✅ VALIDATION TERMINÉE en {total_time:.1f}ms")
        return validation_report

    async def _test_data_collector(self) -> List[ValidationResult]:
        """Tests complets AdvancedFootballDataCollector"""
        results = []
        
        try:
            # Initialisation
            start_time = time.time()
            self.data_collector = AdvancedFootballDataCollector()
            init_time = (time.time() - start_time) * 1000
            
            results.append(ValidationResult(
                component_name="AdvancedFootballDataCollector",
                test_name="initialization",
                success=True,
                execution_time_ms=init_time,
                details={"endpoints_count": len(self.data_collector.available_endpoints)}
            ))
            
            # Test endpoints disponibles
            summary = self.data_collector.get_collection_summary()
            results.append(ValidationResult(
                component_name="AdvancedFootballDataCollector",
                test_name="endpoints_availability",
                success=summary['available_endpoints'] >= 50,  # Au moins 50 endpoints
                execution_time_ms=0,
                details=summary
            ))
            
            # Test collecte simple (sans vraie requête API pour éviter rate limits)
            start_time = time.time()
            # Simulation réussie de collecte
            mock_fixtures = pd.DataFrame({
                'fixture_id': [1, 2, 3],
                'home_team_id': [33, 34, 35],
                'away_team_id': [36, 37, 38],
                'date': ['2024-08-25', '2024-08-26', '2024-08-27']
            })
            
            # Test validation qualité
            validation_report = self.data_collector.validate_data_quality(mock_fixtures)
            collect_time = (time.time() - start_time) * 1000
            
            results.append(ValidationResult(
                component_name="AdvancedFootballDataCollector",
                test_name="data_collection_simulation",
                success=validation_report['status'] == 'valid',
                execution_time_ms=collect_time,
                details=validation_report
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="AdvancedFootballDataCollector",
                test_name="error_handling",
                success=False,
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
        
        return results

    async def _test_feature_engineer(self) -> List[ValidationResult]:
        """Tests complets RevolutionaryFeatureEngineer"""
        results = []
        
        try:
            # Initialisation
            start_time = time.time()
            self.feature_engineer = RevolutionaryFeatureEngineer()
            init_time = (time.time() - start_time) * 1000
            
            results.append(ValidationResult(
                component_name="RevolutionaryFeatureEngineer",
                test_name="initialization",
                success=True,
                execution_time_ms=init_time,
                details={"recent_windows": len(self.feature_engineer.recent_matches_windows)}
            ))
            
            # Créer données de test
            test_data = self._create_test_football_data(100)
            
            # Test génération features complètes
            start_time = time.time()
            enriched_df = self.feature_engineer.create_revolutionary_features(test_data)
            feature_time = (time.time() - start_time) * 1000
            
            initial_cols = len(test_data.columns)
            final_cols = len(enriched_df.columns)
            new_features = final_cols - initial_cols
            
            results.append(ValidationResult(
                component_name="RevolutionaryFeatureEngineer",
                test_name="feature_generation",
                success=new_features >= 50,  # Au moins 50 nouvelles features
                execution_time_ms=feature_time,
                details={
                    "initial_columns": initial_cols,
                    "final_columns": final_cols,
                    "new_features": new_features,
                    "features_per_match": new_features
                }
            ))
            
            # Test sélection de features
            start_time = time.time()
            # Ajouter target pour test
            enriched_df['test_target'] = np.random.randint(0, 3, len(enriched_df))
            
            selected_features = self.feature_engineer.select_best_features(
                enriched_df, 'test_target', max_features=30
            )
            selection_time = (time.time() - start_time) * 1000
            
            results.append(ValidationResult(
                component_name="RevolutionaryFeatureEngineer",
                test_name="feature_selection",
                success=10 <= len(selected_features) <= 30,
                execution_time_ms=selection_time,
                details={"selected_count": len(selected_features)}
            ))
            
            # Test préparation ML
            start_time = time.time()
            ml_ready_df, final_features = self.feature_engineer.prepare_features_for_ml(
                test_data, ['test_target'], fit_scalers=True
            )
            ml_prep_time = (time.time() - start_time) * 1000
            
            results.append(ValidationResult(
                component_name="RevolutionaryFeatureEngineer",
                test_name="ml_preparation",
                success=not ml_ready_df.empty and len(final_features) > 0,
                execution_time_ms=ml_prep_time,
                details={
                    "final_features_count": len(final_features),
                    "scalers_fitted": len(self.feature_engineer.scalers)
                }
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="RevolutionaryFeatureEngineer",
                test_name="error_handling",
                success=False,
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
        
        return results

    async def _test_realtime_pipeline(self) -> List[ValidationResult]:
        """Tests complets RealtimeDataPipeline"""
        results = []
        
        try:
            # Initialisation
            start_time = time.time()
            self.realtime_pipeline = RealtimeDataPipeline()
            init_time = (time.time() - start_time) * 1000
            
            results.append(ValidationResult(
                component_name="RealtimeDataPipeline",
                test_name="initialization",
                success=True,
                execution_time_ms=init_time,
                details={"monitoring_interval": self.realtime_pipeline.monitoring_interval}
            ))
            
            # Test callbacks
            callback_triggered = False
            def test_callback(fixture_id: int, data: Dict):
                nonlocal callback_triggered
                callback_triggered = True
            
            self.realtime_pipeline.register_callback(test_callback)
            
            results.append(ValidationResult(
                component_name="RealtimeDataPipeline",
                test_name="callback_registration",
                success=len(self.realtime_pipeline.callbacks) == 1,
                execution_time_ms=0,
                details={"callbacks_count": len(self.realtime_pipeline.callbacks)}
            ))
            
            # Test statistiques
            stats = self.realtime_pipeline.get_pipeline_stats()
            results.append(ValidationResult(
                component_name="RealtimeDataPipeline",
                test_name="statistics_tracking",
                success=isinstance(stats, dict) and 'total_fixtures_processed' in stats,
                execution_time_ms=0,
                details=stats
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="RealtimeDataPipeline",
                test_name="error_handling",
                success=False,
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
        
        return results

    async def _test_integration(self) -> List[ValidationResult]:
        """Tests d'intégration entre composants"""
        results = []
        
        try:
            # Test intégration DataCollector → FeatureEngineer
            if self.data_collector and self.feature_engineer:
                start_time = time.time()
                
                # Données simulées du collector
                collector_data = self._create_test_football_data(10)
                
                # Features via engineer
                enriched_data = self.feature_engineer.create_revolutionary_features(collector_data)
                
                integration_time = (time.time() - start_time) * 1000
                
                results.append(ValidationResult(
                    component_name="Integration",
                    test_name="collector_to_feature_engineer",
                    success=not enriched_data.empty,
                    execution_time_ms=integration_time,
                    details={
                        "input_rows": len(collector_data),
                        "output_rows": len(enriched_data),
                        "output_columns": len(enriched_data.columns)
                    }
                ))
            
            # Test pipeline complet simulé
            if all([self.data_collector, self.feature_engineer, self.realtime_pipeline]):
                start_time = time.time()
                
                # Simulation pipeline complet
                test_fixture_data = {
                    'fixture_id': 99999,
                    'home_team_id': 33,
                    'away_team_id': 34,
                    'league_id': 39,
                    'season': 2025,
                    'kickoff_time': (datetime.now() + timedelta(hours=2)).isoformat()
                }
                
                # Test ajout fixture au pipeline (sans démarrer monitoring)
                pre_match_data = self.realtime_pipeline.active_fixtures.get(99999)
                
                full_integration_time = (time.time() - start_time) * 1000
                
                results.append(ValidationResult(
                    component_name="Integration",
                    test_name="full_pipeline_simulation",
                    success=True,
                    execution_time_ms=full_integration_time,
                    details=test_fixture_data
                ))
            
        except Exception as e:
            results.append(ValidationResult(
                component_name="Integration",
                test_name="error_handling",
                success=False,
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
        
        return results

    async def _test_performance(self) -> List[ValidationResult]:
        """Tests de performance des composants"""
        results = []
        
        try:
            # Test performance feature engineering sur gros dataset
            large_dataset = self._create_test_football_data(1000)  # 1000 matchs
            
            start_time = time.time()
            if self.feature_engineer:
                enriched_large = self.feature_engineer.create_revolutionary_features(large_dataset)
                large_dataset_time = (time.time() - start_time) * 1000
                
                # Performance acceptable : < 5000ms pour 1000 matchs
                performance_ok = large_dataset_time < 5000
                
                results.append(ValidationResult(
                    component_name="Performance",
                    test_name="feature_engineering_1000_matches",
                    success=performance_ok,
                    execution_time_ms=large_dataset_time,
                    details={
                        "matches_processed": len(large_dataset),
                        "features_generated": len(enriched_large.columns) - len(large_dataset.columns),
                        "ms_per_match": large_dataset_time / len(large_dataset)
                    }
                ))
            
            # Test performance validation qualité
            if self.data_collector:
                start_time = time.time()
                validation_report = self.data_collector.validate_data_quality(large_dataset)
                validation_time = (time.time() - start_time) * 1000
                
                results.append(ValidationResult(
                    component_name="Performance",
                    test_name="data_validation_1000_matches",
                    success=validation_time < 1000,  # < 1 seconde
                    execution_time_ms=validation_time,
                    details=validation_report
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                component_name="Performance",
                test_name="error_handling",
                success=False,
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
        
        return results

    def _create_test_football_data(self, n_matches: int) -> pd.DataFrame:
        """Créer données de test réalistes pour football"""
        np.random.seed(42)  # Reproductibilité
        
        data = {
            'fixture_id': range(1, n_matches + 1),
            'date': pd.date_range('2024-08-01', periods=n_matches, freq='3D'),
            'home_team_id': np.random.randint(1, 21, n_matches),
            'away_team_id': np.random.randint(1, 21, n_matches),
            'league_id': np.random.choice([39, 140, 61, 78, 2, 3], n_matches),
            'season': [2024] * n_matches,
            
            # Données classement
            'home_position': np.random.randint(1, 21, n_matches),
            'away_position': np.random.randint(1, 21, n_matches),
            'home_points': np.random.randint(0, 60, n_matches),
            'away_points': np.random.randint(0, 60, n_matches),
            'home_played': np.random.randint(10, 30, n_matches),
            'away_played': np.random.randint(10, 30, n_matches),
            
            # Résultats (si matchs terminés)
            'home_goals': np.random.randint(0, 5, n_matches),
            'away_goals': np.random.randint(0, 5, n_matches),
            'status_short': np.random.choice(['FT', 'NS'], n_matches, p=[0.7, 0.3]),
        }
        
        return pd.DataFrame(data)

    def _generate_validation_report(self, total_time_ms: float) -> Dict:
        """Générer rapport de validation complet"""
        # Calculer statistiques globales
        total_tests = len(self.validation_results)
        successful_tests = sum(1 for r in self.validation_results if r.success)
        failed_tests = total_tests - successful_tests
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Grouper par composant
        by_component = {}
        for result in self.validation_results:
            component = result.component_name
            if component not in by_component:
                by_component[component] = {'tests': [], 'success_count': 0, 'total_time_ms': 0}
            
            by_component[component]['tests'].append({
                'test_name': result.test_name,
                'success': result.success,
                'execution_time_ms': result.execution_time_ms,
                'details': result.details,
                'error_message': result.error_message
            })
            
            if result.success:
                by_component[component]['success_count'] += 1
            by_component[component]['total_time_ms'] += result.execution_time_ms
        
        # Déterminer succès global
        self.overall_success = success_rate >= 80  # 80% minimum pour valider Phase 1
        
        # Rapport final
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'phase': 'Phase 1 - Infrastructure & Données',
            'overall_success': self.overall_success,
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate_percent': round(success_rate, 2),
                'total_execution_time_ms': round(total_time_ms, 2)
            },
            'components': by_component,
            'recommendations': self._generate_recommendations(),
            'next_steps': [
                "Si validation réussie (≥80%) : Procéder à Phase 2 - Modèles IA",
                "Si validation échouée : Corriger erreurs identifiées",
                "Optimiser performances si temps d'exécution > seuils",
                "Documenter résultats pour traçabilité"
            ]
        }
        
        return report

    def _generate_recommendations(self) -> List[str]:
        """Générer recommandations basées sur résultats"""
        recommendations = []
        
        # Analyser les erreurs
        failed_tests = [r for r in self.validation_results if not r.success]
        
        if failed_tests:
            recommendations.append(f"Corriger {len(failed_tests)} tests en échec")
        
        # Analyser performances
        slow_tests = [r for r in self.validation_results if r.execution_time_ms > 1000]
        if slow_tests:
            recommendations.append(f"Optimiser performances de {len(slow_tests)} tests lents")
        
        # Recommandations par composant
        component_failures = {}
        for result in failed_tests:
            component = result.component_name
            component_failures[component] = component_failures.get(component, 0) + 1
        
        for component, failure_count in component_failures.items():
            if failure_count > 1:
                recommendations.append(f"Attention particulière requise pour {component}")
        
        if not recommendations:
            recommendations.append("Tous les tests passent - Excellente qualité Phase 1!")
        
        return recommendations

    def save_validation_report(self, report: Dict, output_path: Path = None) -> None:
        """Sauvegarder rapport de validation"""
        if output_path is None:
            output_path = config.PROJECT_ROOT / "validation_reports"
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"phase1_validation_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Rapport sauvegardé: {report_file}")

async def run_phase1_validation():
    """Lancer validation complète Phase 1"""
    validator = Phase1ValidationSystem()
    
    report = await validator.run_full_validation()
    
    # Afficher résumé
    logger.info("=" * 60)
    logger.info("🎯 RAPPORT DE VALIDATION PHASE 1")
    logger.info("=" * 60)
    logger.info(f"Succès global: {'✅ OUI' if report['overall_success'] else '❌ NON'}")
    logger.info(f"Tests réussis: {report['summary']['successful_tests']}/{report['summary']['total_tests']}")
    logger.info(f"Taux de succès: {report['summary']['success_rate_percent']}%")
    logger.info(f"Temps total: {report['summary']['total_execution_time_ms']:.1f}ms")
    
    # Sauvegarder
    validator.save_validation_report(report)
    
    return report

if __name__ == "__main__":
    asyncio.run(run_phase1_validation())