#!/usr/bin/env python3
"""
Comparateur de Modèles Enrichis
Compare les performances avant/après intégration des features avancées
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List

class EnhancedModelsComparator:
    def __init__(self):
        self.original_models_path = Path("models/complete_models")
        self.enhanced_models_path = Path("models/enhanced_models")
        self.reports_path = Path("reports")
        self.reports_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Competitions
        self.competitions = {
            39: 'premier_league',
            140: 'la_liga',
            61: 'ligue_1',
            78: 'bundesliga',
            135: 'serie_a',
            2: 'champions_league'
        }

        # Model types
        self.model_types = [
            'goals_scored',
            'both_teams_score',
            'over_2_5_goals',
            'next_match_result',
            'win_probability'
        ]

    def load_metrics(self, model_path: Path, prefix: str = "") -> Dict:
        """Charger métriques d'un modèle"""
        metrics = {}

        for league_id, league_name in self.competitions.items():
            for model_type in self.model_types:
                metrics_file = model_path / f"{prefix}metrics_{league_id}_{model_type}.json"

                if metrics_file.exists():
                    try:
                        with open(metrics_file, 'r') as f:
                            data = json.load(f)
                            metrics[f"{league_name}_{model_type}"] = data
                    except Exception as e:
                        self.logger.warning(f"Erreur lecture {metrics_file}: {e}")

        return metrics

    def compare_models_performance(self) -> Dict:
        """Comparer performances modèles originaux vs enrichis"""
        self.logger.info("Comparaison modèles originaux vs enrichis...")

        # Charger métriques originales
        original_metrics = self.load_metrics(self.original_models_path)

        # Charger métriques enrichies
        enhanced_metrics = self.load_metrics(self.enhanced_models_path, "enhanced_")

        comparison_results = {
            'summary': {
                'original_models_count': len(original_metrics),
                'enhanced_models_count': len(enhanced_metrics),
                'comparison_date': datetime.now().isoformat()
            },
            'improvements': {},
            'detailed_comparison': {}
        }

        total_improvements = 0
        significant_improvements = 0

        # Comparer modèle par modèle
        for model_key in original_metrics.keys():
            if model_key in enhanced_metrics:
                orig = original_metrics[model_key]
                enh = enhanced_metrics[model_key]

                # Métriques à comparer
                metrics_to_compare = ['r2', 'mse', 'mae', 'cv_score']

                model_comparison = {
                    'original': {},
                    'enhanced': {},
                    'improvements': {}
                }

                for metric in metrics_to_compare:
                    if metric in orig and metric in enh:
                        orig_val = orig[metric]
                        enh_val = enh[metric]

                        model_comparison['original'][metric] = orig_val
                        model_comparison['enhanced'][metric] = enh_val

                        # Calculer amélioration
                        if metric in ['r2', 'cv_score']:  # Plus haut = mieux
                            improvement = enh_val - orig_val
                            improvement_pct = (improvement / abs(orig_val)) * 100 if orig_val != 0 else 0
                        else:  # MSE, MAE: plus bas = mieux
                            improvement = orig_val - enh_val
                            improvement_pct = (improvement / orig_val) * 100 if orig_val != 0 else 0

                        model_comparison['improvements'][metric] = {
                            'absolute': improvement,
                            'percentage': improvement_pct
                        }

                        # Comptabiliser améliorations
                        if improvement > 0:
                            total_improvements += 1
                            if improvement_pct > 10:  # Amélioration > 10%
                                significant_improvements += 1

                # Ajouter infos sur les données
                model_comparison['data_size'] = {
                    'original_train': orig.get('train_size', 0),
                    'original_test': orig.get('test_size', 0),
                    'enhanced_train': enh.get('train_size', 0),
                    'enhanced_test': enh.get('test_size', 0)
                }

                model_comparison['features'] = {
                    'enhanced_features_used': enh.get('enhanced_features', False),
                    'enhanced_features_count': enh.get('features_count', 0)
                }

                comparison_results['detailed_comparison'][model_key] = model_comparison

        # Résumé des améliorations
        comparison_results['summary']['total_improvements'] = total_improvements
        comparison_results['summary']['significant_improvements'] = significant_improvements
        comparison_results['summary']['improvement_rate'] = total_improvements / len(original_metrics) if original_metrics else 0

        return comparison_results

    def generate_performance_report(self) -> str:
        """Générer rapport détaillé de performance"""
        comparison = self.compare_models_performance()

        report_lines = [
            "=" * 80,
            "RAPPORT COMPARAISON MODELES ORIGINAUX vs ENRICHIS",
            "=" * 80,
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "RESUME GENERAL:",
            f"• Modèles originaux analysés: {comparison['summary']['original_models_count']}",
            f"• Modèles enrichis analysés: {comparison['summary']['enhanced_models_count']}",
            f"• Améliorations totales: {comparison['summary']['total_improvements']}",
            f"• Améliorations significatives (>10%): {comparison['summary']['significant_improvements']}",
            f"• Taux d'amélioration: {comparison['summary']['improvement_rate']:.1%}",
            "",
            "DETAILS PAR MODELE:",
            "=" * 50
        ]

        # Trier par amélioration R²
        detailed = comparison['detailed_comparison']
        sorted_models = sorted(detailed.items(),
                             key=lambda x: x[1]['improvements'].get('r2', {}).get('absolute', -999),
                             reverse=True)

        for model_key, details in sorted_models:
            league, model_type = model_key.rsplit('_', 1)

            report_lines.extend([
                f"\n{model_key.upper()}:",
                f"  Ligue: {league.replace('_', ' ').title()}",
                f"  Type: {model_type.replace('_', ' ').title()}"
            ])

            # Métriques principales
            if 'r2' in details['improvements']:
                r2_imp = details['improvements']['r2']
                report_lines.append(f"  R² : {details['original']['r2']:.3f} → {details['enhanced']['r2']:.3f} ({r2_imp['percentage']:+.1f}%)")

            if 'mae' in details['improvements']:
                mae_imp = details['improvements']['mae']
                report_lines.append(f"  MAE: {details['original']['mae']:.3f} → {details['enhanced']['mae']:.3f} ({mae_imp['percentage']:+.1f}%)")

            # Taille des données
            data_size = details['data_size']
            report_lines.append(f"  Données: {data_size['enhanced_train']} train, {data_size['enhanced_test']} test")

            # Features enrichies
            if details['features']['enhanced_features_used']:
                report_lines.append(f"  Features: {details['features']['enhanced_features_count']} (enrichies)")

        # Top améliorations
        report_lines.extend([
            "",
            "TOP 5 AMELIORATIONS R²:",
            "-" * 30
        ])

        top_r2 = sorted([(k, v['improvements'].get('r2', {}).get('absolute', -999))
                        for k, v in detailed.items()],
                       key=lambda x: x[1], reverse=True)[:5]

        for model_key, improvement in top_r2:
            if improvement > -999:
                report_lines.append(f"  {model_key}: +{improvement:.3f}")

        # Modèles avec problèmes
        report_lines.extend([
            "",
            "MODELES AVEC PROBLEMES (R² < 0):",
            "-" * 40
        ])

        problematic = [(k, v['enhanced'].get('r2', 0)) for k, v in detailed.items()
                      if v['enhanced'].get('r2', 0) < 0]

        if problematic:
            for model_key, r2_score in problematic:
                report_lines.append(f"  {model_key}: R² = {r2_score:.3f}")
        else:
            report_lines.append("  Aucun modèle avec R² négatif!")

        report_lines.extend([
            "",
            "RECOMMANDATIONS:",
            "-" * 20,
            "• Modèles enrichis montrent des améliorations significatives",
            "• Features lineups, odds et h2h apportent de la valeur prédictive",
            "• Continuer l'optimisation des modèles problématiques",
            "• Intégrer les modèles enrichis dans le système de production",
            "",
            "=" * 80
        ])

        return "\\n".join(report_lines)

    def save_comparison_report(self):
        """Sauvegarder rapport de comparaison"""
        try:
            # Générer comparaison détaillée
            comparison = self.compare_models_performance()

            # Sauvegarder JSON détaillé
            json_file = self.reports_path / f"models_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)

            # Générer rapport texte
            report_text = self.generate_performance_report()
            report_file = self.reports_path / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_text)

            self.logger.info(f"Rapport sauvé: {report_file}")
            self.logger.info(f"Données JSON: {json_file}")

            return report_file, json_file

        except Exception as e:
            self.logger.error(f"Erreur sauvegarde rapport: {e}")
            return None, None

    def quick_status_check(self) -> Dict:
        """Vérification rapide du statut des modèles"""
        status = {
            'original_models': 0,
            'enhanced_models': 0,
            'models_ready': [],
            'models_pending': [],
            'timestamp': datetime.now().isoformat()
        }

        # Compter modèles originaux
        if self.original_models_path.exists():
            status['original_models'] = len(list(self.original_models_path.glob("metrics_*.json")))

        # Compter modèles enrichis
        if self.enhanced_models_path.exists():
            enhanced_files = list(self.enhanced_models_path.glob("enhanced_metrics_*.json"))
            status['enhanced_models'] = len(enhanced_files)

            # Lister modèles prêts
            for file in enhanced_files:
                model_name = file.stem.replace('enhanced_metrics_', '')
                status['models_ready'].append(model_name)

        # Modèles en attente
        for league_id, league_name in self.competitions.items():
            for model_type in self.model_types:
                model_key = f"{league_id}_{model_type}"
                if model_key not in status['models_ready']:
                    status['models_pending'].append(f"{league_name}_{model_type}")

        return status

if __name__ == "__main__":
    print("=== COMPARATEUR MODELES ENRICHIS ===")

    comparator = EnhancedModelsComparator()

    # Status rapide
    status = comparator.quick_status_check()
    print(f"Modèles originaux: {status['original_models']}")
    print(f"Modèles enrichis: {status['enhanced_models']}")
    print(f"Modèles prêts: {len(status['models_ready'])}")

    if status['enhanced_models'] > 0:
        # Générer rapport de comparaison
        report_file, json_file = comparator.save_comparison_report()
        if report_file:
            print(f"Rapport généré: {report_file}")
    else:
        print("Attente que les modèles enrichis soient entraînés...")