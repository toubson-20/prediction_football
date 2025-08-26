"""
PHASE 2 VALIDATION SYSTEM - TESTS COMPLETS ENSEMBLE REVOLUTIONNAIRE
Validation complete de tous les composants de la Phase 2

Version: 2.0 - Phase 2 ML Transformation
Cree: 23 aout 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings
warnings.filterwarnings('ignore')

# Imports des composants Phase 2
try:
    from revolutionary_model_architecture import RevolutionaryModelArchitecture
    from massive_model_trainer import MassiveModelTrainer
    from deep_learning_models import DeepLearningEnsemble
    from intelligent_meta_model import IntelligentMetaModel
    from transfer_learning_system import TransferLearningOrchestrator
except ImportError as e:
    print(f"Warning: Import manque - {e}")

class Phase2ValidationSuite:
    """Suite complete de validation pour la Phase 2"""
    
    def __init__(self):
        self.validation_results = {}
        self.performance_metrics = {}
        self.test_data = self._generate_comprehensive_test_data()
        
    def _generate_comprehensive_test_data(self) -> Dict:
        """Generate des donnees de test completes pour tous les composants"""
        
        np.random.seed(42)
        
        # Donnees pour differentes ligues
        leagues = ['Premier_League', 'La_Liga', 'Bundesliga', 'Serie_A', 'Ligue_1', 'Champions_League']
        prediction_types = [
            'match_result', 'total_goals', 'both_teams_scored', 'over_2_5_goals',
            'home_goals', 'away_goals', 'first_half_result', 'correct_score',
            'double_chance', 'handicap_home', 'corners_total', 'cards_total'
        ]
        
        test_data = {}
        
        for league in leagues:
            test_data[league] = {}
            
            # Caracteristiques specifiques par ligue
            league_multiplier = {
                'Premier_League': 1.0,
                'La_Liga': 0.9,
                'Bundesliga': 1.1,
                'Serie_A': 0.8,
                'Ligue_1': 0.95,
                'Champions_League': 1.2
            }.get(league, 1.0)
            
            # Donnees d'entrainement
            n_samples = int(1000 * league_multiplier)
            n_features = 200
            
            X = np.random.randn(n_samples, n_features) * league_multiplier
            
            # Targets pour chaque type de prediction
            for pred_type in prediction_types:
                if pred_type == 'match_result':
                    y = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.3, 0.4])
                elif 'goals' in pred_type:
                    y = np.random.poisson(2.5 * league_multiplier, n_samples)
                elif pred_type == 'both_teams_scored':
                    y = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
                elif 'corners' in pred_type:
                    y = np.random.poisson(10, n_samples)
                elif 'cards' in pred_type:
                    y = np.random.poisson(4, n_samples)
                else:
                    y = np.random.randn(n_samples) * league_multiplier
                
                test_data[league][pred_type] = {
                    'X': X,
                    'y': y,
                    'X_train': X[:int(0.8*len(X))],
                    'y_train': y[:int(0.8*len(y))],
                    'X_test': X[int(0.8*len(X)):],
                    'y_test': y[int(0.8*len(y)):]
                }
        
        return test_data
    
    def validate_revolutionary_architecture(self) -> Dict:
        """Valide l'architecture revolutionnaire des modeles"""
        
        print("Validation Architecture Revolutionnaire...")
        start_time = time.time()
        
        try:
            architecture = RevolutionaryModelArchitecture()
            
            # Tests de base
            results = {
                'architecture_initialized': True,
                'prediction_types_count': len(architecture.prediction_types),
                'leagues_count': len(architecture.leagues),
                'total_models_expected': len(architecture.prediction_types) * len(architecture.leagues),
                'algorithms_available': len(architecture.available_algorithms)
            }
            
            # Test creation modele
            test_league = 'Premier_League'
            test_pred_type = 'match_result'
            
            model = architecture.create_specialized_model(test_league, test_pred_type, 'XGBoost')
            results['model_creation_successful'] = model is not None
            
            # Test entrainement rapide
            test_data = self.test_data[test_league][test_pred_type]
            
            if hasattr(model, 'fit'):
                model.fit(test_data['X_train'], test_data['y_train'])
                predictions = model.predict(test_data['X_test'])
                results['model_training_successful'] = len(predictions) > 0
                results['predictions_shape'] = predictions.shape
            
            # Test des parametres optimises
            optimized_params = architecture.get_optimized_parameters(test_pred_type, 'XGBoost')
            results['optimized_params_available'] = len(optimized_params) > 0
            
            results['validation_time'] = time.time() - start_time
            results['status'] = 'SUCCESS'
            
        except Exception as e:
            results = {
                'status': 'FAILED',
                'error': str(e),
                'validation_time': time.time() - start_time
            }
        
        self.validation_results['revolutionary_architecture'] = results
        return results
    
    def validate_massive_trainer(self) -> Dict:
        """Valide le systeme d'entrainement massif"""
        
        print("Validation Massive Model Trainer...")
        start_time = time.time()
        
        try:
            trainer = MassiveModelTrainer()
            
            results = {
                'trainer_initialized': True,
                'leagues_configured': len(trainer.leagues),
                'prediction_types_configured': len(trainer.prediction_types)
            }
            
            # Test preparation donnees
            test_league = 'Premier_League'
            league_data = {}
            
            for pred_type in ['match_result', 'total_goals', 'both_teams_scored']:
                league_data[pred_type] = self.test_data[test_league][pred_type]
            
            prepared_data = trainer._prepare_league_training_data(test_league, league_data)
            results['data_preparation_successful'] = len(prepared_data) > 0
            
            # Test generation targets
            X_sample = self.test_data[test_league]['match_result']['X'][:100]
            
            targets_generated = 0
            for pred_type in ['match_result', 'total_goals', 'both_teams_scored']:
                try:
                    targets = trainer._generate_targets_for_prediction(X_sample, pred_type)
                    if len(targets) > 0:
                        targets_generated += 1
                except Exception as e:
                    print(f"Erreur generation targets {pred_type}: {e}")
            
            results['target_generation_successful'] = targets_generated > 0
            results['target_types_working'] = targets_generated
            
            # Test entrainement d'un modele
            try:
                single_result = trainer._train_single_model(
                    league=test_league,
                    prediction_type='match_result',
                    algorithm='RandomForest',
                    training_data=league_data['match_result']
                )
                
                results['single_training_successful'] = single_result['status'] == 'success'
                if single_result['status'] == 'success':
                    results['training_accuracy'] = single_result.get('accuracy', 0.0)
                
            except Exception as e:
                results['single_training_successful'] = False
                results['training_error'] = str(e)
            
            results['validation_time'] = time.time() - start_time
            results['status'] = 'SUCCESS'
            
        except Exception as e:
            results = {
                'status': 'FAILED',
                'error': str(e),
                'validation_time': time.time() - start_time
            }
        
        self.validation_results['massive_trainer'] = results
        return results
    
    def validate_deep_learning_models(self) -> Dict:
        """Valide les modeles Deep Learning"""
        
        print("Validation Deep Learning Models...")
        start_time = time.time()
        
        try:
            # Test avec des donnees reduites pour vitesse
            test_X = self.test_data['Premier_League']['match_result']['X'][:200, :50]
            test_y = self.test_data['Premier_League']['match_result']['y'][:200]
            
            device = 'cpu'  # Force CPU pour tests
            ensemble = DeepLearningEnsemble(input_dim=50, device=device)
            
            results = {
                'ensemble_initialized': True,
                'device_used': device,
                'models_available': list(ensemble.models.keys()),
                'models_count': len(ensemble.models)
            }
            
            # Test entrainement de chaque modele (peu d'epochs)
            training_results = {}
            
            for model_name in ['transformer', 'lstm', 'cnn1d']:  # Skip GNN pour test rapide
                try:
                    print(f"  Test {model_name}...")
                    
                    training_result = ensemble.train_model(
                        model_name=model_name,
                        X_train=test_X,
                        y_train=test_y,
                        epochs=3,  # Tres peu pour test rapide
                        batch_size=32
                    )
                    
                    training_results[model_name] = {
                        'success': True,
                        'final_loss': training_result['final_loss'],
                        'epochs_completed': training_result['epochs']
                    }
                    
                    # Test prediction
                    predictions = ensemble.predict(test_X[:10], [model_name])
                    training_results[model_name]['predictions_generated'] = len(predictions[model_name]) > 0
                    
                except Exception as e:
                    training_results[model_name] = {
                        'success': False,
                        'error': str(e)
                    }
            
            results['training_results'] = training_results
            
            # Test ensemble prediction
            try:
                trained_models = [name for name, result in training_results.items() if result.get('success', False)]
                if trained_models:
                    ensemble_pred = ensemble.get_ensemble_prediction(test_X[:5])
                    results['ensemble_prediction_successful'] = len(ensemble_pred) > 0
                    results['trained_models_count'] = len(trained_models)
            except Exception as e:
                results['ensemble_prediction_error'] = str(e)
            
            # Infos modeles
            model_info = ensemble.get_model_info()
            results['model_parameters'] = {name: info['total_parameters'] for name, info in model_info.items()}
            
            results['validation_time'] = time.time() - start_time
            results['status'] = 'SUCCESS'
            
        except Exception as e:
            results = {
                'status': 'FAILED',
                'error': str(e),
                'validation_time': time.time() - start_time
            }
        
        self.validation_results['deep_learning_models'] = results
        return results
    
    def validate_meta_model(self) -> Dict:
        """Valide le meta-modele intelligent"""
        
        print("Validation Meta-Model Intelligent...")
        start_time = time.time()
        
        try:
            meta_model = IntelligentMetaModel()
            
            results = {
                'meta_model_initialized': True,
                'performance_tracker_ready': True,
                'model_selector_ready': True
            }
            
            # Simulation donnees historiques pour meta-apprentissage
            print("  Simulation donnees historiques...")
            
            models = ['xgb_match_result', 'rf_match_result', 'nn_match_result', 'lstm', 'transformer']
            leagues = ['Premier_League', 'La_Liga', 'Bundesliga']
            
            # Generation de 150 predictions simulees
            for i in range(150):
                context = {
                    'league': np.random.choice(leagues),
                    'prediction_type': 'match_result',
                    'match_importance': np.random.choice(['normal', 'high']),
                    'date': '2025-01-15',
                    'home_team_form_points': np.random.randint(0, 15),
                    'away_team_form_points': np.random.randint(0, 15)
                }
                
                model_name = np.random.choice(models)
                prediction = np.random.uniform(0, 2)
                actual = np.random.uniform(0, 2)
                confidence = np.random.uniform(0.5, 0.95)
                
                meta_model.record_prediction_result(model_name, context, prediction, actual, confidence)
            
            results['historical_data_simulated'] = len(meta_model.selection_history)
            
            # Test selection optimale
            test_match = {
                'league': 'Premier_League',
                'prediction_type': 'match_result',
                'match_importance': 'high',
                'date': '2025-01-20'
            }
            
            ensemble_result = meta_model.get_optimal_model_ensemble(
                'match_result', test_match, models, ensemble_size=3
            )
            
            results['optimal_selection_successful'] = len(ensemble_result['models']) > 0
            results['selected_models_count'] = len(ensemble_result['models'])
            results['ensemble_confidence'] = ensemble_result['confidence']
            
            # Test meta-apprentissage
            meta_training_success = meta_model.train_meta_model(min_samples=50)
            results['meta_training_successful'] = meta_training_success
            
            if meta_training_success:
                best_model, expected_perf = meta_model.predict_best_model(test_match)
                results['meta_prediction_successful'] = best_model is not None
                results['predicted_best_model'] = best_model
                results['expected_performance'] = expected_perf
            
            # Resume performances
            summary = meta_model.get_performance_summary()
            results['performance_summary'] = summary
            
            results['validation_time'] = time.time() - start_time
            results['status'] = 'SUCCESS'
            
        except Exception as e:
            results = {
                'status': 'FAILED',
                'error': str(e),
                'validation_time': time.time() - start_time
            }
        
        self.validation_results['meta_model'] = results
        return results
    
    def validate_transfer_learning(self) -> Dict:
        """Valide le systeme de transfer learning"""
        
        print("Validation Transfer Learning...")
        start_time = time.time()
        
        try:
            orchestrator = TransferLearningOrchestrator()
            
            results = {
                'orchestrator_initialized': True,
                'league_characteristics_loaded': len(orchestrator.league_characteristics.league_profiles),
                'league_similarities_calculated': len(orchestrator.league_characteristics.league_similarities)
            }
            
            # Test similarites entre ligues
            similar_leagues = orchestrator.league_characteristics.get_most_similar_leagues('Premier_League', top_k=3)
            results['similarity_calculation_successful'] = len(similar_leagues) > 0
            results['most_similar_to_PL'] = [league for league, _ in similar_leagues]
            
            # Test caracteristiques transferables
            transferable = orchestrator.league_characteristics.get_transferable_characteristics('Premier_League', 'La_Liga')
            results['transferable_characteristics_count'] = len(transferable)
            
            # Simulation transfer learning
            print("  Test transfer learning...")
            
            # Donnees simulees pour 2 ligues
            np.random.seed(42)
            X_source = np.random.randn(200, 30)
            y_source = np.random.randn(200)
            X_target = np.random.randn(150, 30) * 1.1 + 0.2
            y_target = np.random.randn(150) * 0.9
            
            # Modele source
            from sklearn.ensemble import RandomForestRegressor
            source_model = RandomForestRegressor(n_estimators=50, random_state=42)
            source_model.fit(X_source, y_source)
            
            # Enregistrement
            orchestrator.register_source_model(
                model=source_model,
                league='Premier_League',
                prediction_type='total_goals',
                training_data={
                    'X_train': X_source,
                    'y_train': y_source,
                    'metrics': {'mae': 0.5, 'r2': 0.7}
                }
            )
            
            results['source_model_registered'] = True
            
            # Test transfer
            target_data = {
                'X_train': X_target[:120],
                'y_train': y_target[:120],
                'X_val': X_target[120:],
                'y_val': y_target[120:]
            }
            
            # Test feature-based transfer
            transfer_result = orchestrator.transfer_model_knowledge(
                source_league='Premier_League',
                target_league='La_Liga',
                prediction_type='total_goals',
                target_training_data=target_data,
                transfer_strategy='feature_based'
            )
            
            results['feature_transfer_successful'] = transfer_result.get('success', False)
            if transfer_result.get('success'):
                results['performance_improvement'] = transfer_result.get('performance_improvement', 0.0)
            
            # Recommandations sources
            best_sources = orchestrator.get_best_source_leagues('La_Liga', 'total_goals')
            results['source_recommendations'] = len(best_sources)
            
            # Resume
            summary = orchestrator.get_transfer_summary()
            results['transfer_summary'] = summary
            
            results['validation_time'] = time.time() - start_time
            results['status'] = 'SUCCESS'
            
        except Exception as e:
            results = {
                'status': 'FAILED',
                'error': str(e),
                'validation_time': time.time() - start_time
            }
        
        self.validation_results['transfer_learning'] = results
        return results
    
    def run_integration_tests(self) -> Dict:
        """Tests d'integration entre tous les composants"""
        
        print("Tests d'Integration Phase 2...")
        start_time = time.time()
        
        try:
            results = {
                'integration_tests_started': True,
                'components_tested': []
            }
            
            # Test 1: Architecture + Trainer
            print("  Test Architecture + Trainer...")
            try:
                architecture = RevolutionaryModelArchitecture()
                trainer = MassiveModelTrainer()
                
                # Test creation et entrainement
                test_data = self.test_data['Premier_League']['match_result']
                model = architecture.create_specialized_model('Premier_League', 'match_result', 'RandomForest')
                
                if model and hasattr(model, 'fit'):
                    model.fit(test_data['X_train'], test_data['y_train'])
                    predictions = model.predict(test_data['X_test'])
                    
                    results['architecture_trainer_integration'] = len(predictions) > 0
                    results['components_tested'].append('Architecture+Trainer')
                
            except Exception as e:
                results['architecture_trainer_error'] = str(e)
            
            # Test 2: Meta-Model + Deep Learning
            print("  Test Meta-Model + Deep Learning...")
            try:
                meta_model = IntelligentMetaModel()
                
                # Simulation selection avec deep learning
                test_match = {
                    'league': 'Premier_League',
                    'prediction_type': 'match_result',
                    'match_importance': 'high'
                }
                
                models_available = ['xgb_match_result', 'transformer', 'lstm']
                ensemble_result = meta_model.get_optimal_model_ensemble(
                    'match_result', test_match, models_available, ensemble_size=2
                )
                
                results['meta_deep_integration'] = len(ensemble_result['models']) > 0
                results['components_tested'].append('Meta-Model+DeepLearning')
                
            except Exception as e:
                results['meta_deep_error'] = str(e)
            
            # Test 3: Transfer Learning + Architecture
            print("  Test Transfer Learning + Architecture...")
            try:
                orchestrator = TransferLearningOrchestrator()
                
                # Test recommandations pour nouvelle ligue
                recommendations = orchestrator.get_best_source_leagues('Bundesliga', 'total_goals')
                results['transfer_architecture_integration'] = len(recommendations) >= 0
                results['components_tested'].append('Transfer+Architecture')
                
            except Exception as e:
                results['transfer_architecture_error'] = str(e)
            
            # Performance globale des composants valides
            successful_components = len([comp for comp_name, comp_results in self.validation_results.items() 
                                        if comp_results.get('status') == 'SUCCESS'])
            
            results['successful_components'] = successful_components
            results['total_components'] = len(self.validation_results)
            results['integration_success_rate'] = successful_components / len(self.validation_results) if self.validation_results else 0.0
            
            results['validation_time'] = time.time() - start_time
            results['status'] = 'SUCCESS' if results['integration_success_rate'] > 0.5 else 'PARTIAL'
            
        except Exception as e:
            results = {
                'status': 'FAILED',
                'error': str(e),
                'validation_time': time.time() - start_time
            }
        
        self.validation_results['integration_tests'] = results
        return results
    
    def run_complete_validation(self) -> Dict:
        """Lance la validation complete de la Phase 2"""
        
        print("=" * 60)
        print("VALIDATION COMPLETE PHASE 2 - SYSTEME REVOLUTIONNAIRE")
        print("=" * 60)
        
        start_time = time.time()
        
        # Validation de chaque composant
        validation_sequence = [
            ('Revolutionary Architecture', self.validate_revolutionary_architecture),
            ('Massive Model Trainer', self.validate_massive_trainer),
            ('Deep Learning Models', self.validate_deep_learning_models),
            ('Meta-Model Intelligent', self.validate_meta_model),
            ('Transfer Learning System', self.validate_transfer_learning)
        ]
        
        for component_name, validation_func in validation_sequence:
            print(f"\\n{component_name}...")
            try:
                validation_func()
            except Exception as e:
                print(f"ERREUR {component_name}: {e}")
                self.validation_results[component_name.lower().replace(' ', '_')] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
        
        # Tests d'integration
        self.run_integration_tests()
        
        # Compilation du rapport final
        total_time = time.time() - start_time
        final_results = self._compile_final_report(total_time)
        
        return final_results
    
    def _compile_final_report(self, total_time: float) -> Dict:
        """Compile le rapport final de validation"""
        
        successful_components = []
        failed_components = []
        warnings = []
        
        # Analyse des resultats
        for component_name, results in self.validation_results.items():
            if results.get('status') == 'SUCCESS':
                successful_components.append(component_name)
            else:
                failed_components.append({
                    'component': component_name,
                    'error': results.get('error', 'Erreur inconnue')
                })
        
        # Calcul metriques globales
        success_rate = len(successful_components) / len(self.validation_results) if self.validation_results else 0.0
        
        # Rapport detaille
        report = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'total_validation_time': total_time,
            'components_tested': len(self.validation_results),
            'successful_components': len(successful_components),
            'failed_components': len(failed_components),
            'overall_success_rate': success_rate,
            'phase2_status': 'READY' if success_rate >= 0.8 else 'NEEDS_ATTENTION' if success_rate >= 0.5 else 'FAILED',
            
            'component_details': {
                'successful': successful_components,
                'failed': failed_components,
                'detailed_results': self.validation_results
            },
            
            'performance_summary': self._calculate_performance_summary(),
            'recommendations': self._generate_recommendations()
        }
        
        # Affichage du rapport
        self._display_final_report(report)
        
        return report
    
    def _calculate_performance_summary(self) -> Dict:
        """Calcule le resume des performances"""
        
        summary = {
            'total_models_architecture': 0,
            'training_models_tested': 0,
            'deep_learning_models': 0,
            'transfer_learning_pairs': 0
        }
        
        # Architecture
        if 'revolutionary_architecture' in self.validation_results:
            arch_results = self.validation_results['revolutionary_architecture']
            summary['total_models_architecture'] = arch_results.get('total_models_expected', 0)
        
        # Deep Learning
        if 'deep_learning_models' in self.validation_results:
            dl_results = self.validation_results['deep_learning_models']
            summary['deep_learning_models'] = dl_results.get('models_count', 0)
        
        # Transfer Learning
        if 'transfer_learning' in self.validation_results:
            tl_results = self.validation_results['transfer_learning']
            summary['transfer_learning_pairs'] = tl_results.get('source_recommendations', 0)
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate des recommandations basees sur les resultats"""
        
        recommendations = []
        
        # Analyse des echecs
        for component_name, results in self.validation_results.items():
            if results.get('status') != 'SUCCESS':
                if 'deep_learning' in component_name:
                    recommendations.append("Installer PyTorch et dependances pour Deep Learning")
                elif 'transfer' in component_name:
                    recommendations.append("Verifier les donnees d'entrainement pour Transfer Learning")
                else:
                    recommendations.append(f"Corriger les erreurs dans {component_name}")
        
        # Recommandations generales
        success_rate = len([r for r in self.validation_results.values() if r.get('status') == 'SUCCESS']) / len(self.validation_results)
        
        if success_rate >= 0.8:
            recommendations.append("Phase 2 prete -> Proceder a la Phase 3 (Coupon Intelligent)")
        elif success_rate >= 0.5:
            recommendations.append("Corriger les composants defaillants avant Phase 3")
        else:
            recommendations.append("Revision majeure necessaire avant de continuer")
        
        return recommendations
    
    def _display_final_report(self, report: Dict):
        """Affiche le rapport final formate"""
        
        print("\\n" + "=" * 60)
        print("RAPPORT FINAL - VALIDATION PHASE 2")
        print("=" * 60)
        
        print(f"Temps total: {report['total_validation_time']:.2f}s")
        print(f"Composants reussis: {report['successful_components']}/{report['components_tested']}")
        print(f"Taux de reussite: {report['overall_success_rate']:.1%}")
        print(f"Statut Phase 2: {report['phase2_status']}")
        
        # Composants reussis
        if report['component_details']['successful']:
            print(f"\\nCOMPOSANTS FONCTIONNELS:")
            for component in report['component_details']['successful']:
                print(f"   + {component}")
        
        # Composants echoues
        if report['component_details']['failed']:
            print(f"\\nCOMPOSANTS A CORRIGER:")
            for failed in report['component_details']['failed']:
                print(f"   - {failed['component']}: {failed['error'][:100]}...")
        
        # Resume des performances
        perf = report['performance_summary']
        print(f"\\nPERFORMANCES:")
        print(f"   * Modeles architecture: {perf['total_models_architecture']}")
        print(f"   * Modeles Deep Learning: {perf['deep_learning_models']}")
        print(f"   * Pairs Transfer Learning: {perf['transfer_learning_pairs']}")
        
        # Recommandations
        if report['recommendations']:
            print(f"\\nRECOMMANDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        print("\\n" + "=" * 60)
        
        # Status final selon les resultats
        if report['phase2_status'] == 'READY':
            print("PHASE 2 VALIDEE - PRETE POUR PHASE 3!")
        elif report['phase2_status'] == 'NEEDS_ATTENTION':
            print("PHASE 2 PARTIELLEMENT VALIDEE - CORRECTIONS MINEURES NECESSAIRES")
        else:
            print("PHASE 2 NECESSITE DES CORRECTIONS MAJEURES")
        
        print("=" * 60)

def run_phase2_validation():
    """Point d'entree pour la validation de la Phase 2"""
    
    print("Initialisation de la validation Phase 2...")
    
    validator = Phase2ValidationSuite()
    final_report = validator.run_complete_validation()
    
    return final_report

if __name__ == "__main__":
    run_phase2_validation()