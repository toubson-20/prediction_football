"""
SYSTEME DE VALIDATION DES DONNEES V2.0
Valide l'intégrité et la cohérence des données collectées
Détecte les problèmes comme ceux qu'on a eus avec Haaland
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

class DataValidationSystem:
    """Système de validation exhaustif pour les nouvelles données."""
    
    def __init__(self):
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'warnings': [],
            'errors': [],
            'critical_errors': [],
            'validation_details': {}
        }
        
        # Seuils de validation
        self.thresholds = {
            'max_goals_per_player_start_season': 10,  # Max buts début saison
            'max_events_per_match': 150,             # Max événements par match
            'min_matches_per_season': 300,           # Min matchs par saison
            'max_matches_per_season': 450,           # Max matchs par saison
            'current_season': 2025                   # Saison actuelle
        }
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging pour validation."""
        log_dir = Path("logs/validation")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_season_consistency(self, competition_path: Path) -> Dict:
        """Valider la cohérence des saisons."""
        self.logger.info(f"Validation saisons pour {competition_path.name}")
        
        validation = {
            'check_name': 'Season Consistency',
            'status': 'passed',
            'details': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            # Vérifier que les 5 saisons existent
            expected_seasons = [2021, 2022, 2023, 2024, 2025]
            seasons_path = competition_path / "seasons"
            
            if not seasons_path.exists():
                validation['status'] = 'failed'
                validation['errors'].append("Dossier seasons manquant")
                return validation
            
            existing_seasons = [int(d.name) for d in seasons_path.iterdir() if d.is_dir() and d.name.isdigit()]
            missing_seasons = set(expected_seasons) - set(existing_seasons)
            extra_seasons = set(existing_seasons) - set(expected_seasons)
            
            if missing_seasons:
                validation['warnings'].append(f"Saisons manquantes: {missing_seasons}")
            
            if extra_seasons:
                validation['warnings'].append(f"Saisons inattendues: {extra_seasons}")
            
            # Vérifier que saison 2025 = saison actuelle
            if 2025 in existing_seasons:
                validation['details'].append("✓ Saison actuelle 2025 (2025-2026) présente")
            else:
                validation['status'] = 'failed'
                validation['errors'].append("Saison actuelle 2025 manquante")
            
            validation['details'].append(f"Saisons trouvées: {sorted(existing_seasons)}")
            
        except Exception as e:
            validation['status'] = 'failed'
            validation['errors'].append(f"Erreur validation saisons: {e}")
        
        return validation
    
    def validate_fixtures_data(self, season_path: Path, season_year: int) -> Dict:
        """Valider les données de matchs."""
        validation = {
            'check_name': f'Fixtures Data {season_year}',
            'status': 'passed',
            'details': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            fixtures_file = season_path / "fixtures/fixtures.json"
            
            if not fixtures_file.exists():
                validation['status'] = 'failed'
                validation['errors'].append("Fichier fixtures.json manquant")
                return validation
            
            with open(fixtures_file, 'r', encoding='utf-8') as f:
                fixtures_data = json.load(f)
            
            if 'response' not in fixtures_data:
                validation['status'] = 'failed'
                validation['errors'].append("Structure fixtures invalide - pas de 'response'")
                return validation
            
            fixtures = fixtures_data['response']
            num_fixtures = len(fixtures)
            
            # Vérifier nombre de matchs cohérent
            if num_fixtures < self.thresholds['min_matches_per_season']:
                validation['warnings'].append(f"Peu de matchs: {num_fixtures}")
            elif num_fixtures > self.thresholds['max_matches_per_season']:
                validation['warnings'].append(f"Beaucoup de matchs: {num_fixtures}")
            else:
                validation['details'].append(f"✓ Nombre de matchs cohérent: {num_fixtures}")
            
            # Vérifier les dates des matchs
            future_matches = 0
            past_matches = 0
            now = datetime.now()
            
            for fixture in fixtures[:10]:  # Échantillon
                match_date_str = fixture['fixture']['date']
                match_date = datetime.fromisoformat(match_date_str.replace('Z', '+00:00'))
                
                if match_date > now:
                    future_matches += 1
                else:
                    past_matches += 1
            
            # Pour saison actuelle, il devrait y avoir des matchs futurs
            if season_year == self.thresholds['current_season']:
                if future_matches == 0:
                    validation['warnings'].append("Aucun match futur en saison actuelle")
                else:
                    validation['details'].append(f"✓ Matchs futurs présents en saison actuelle")
            
            validation['details'].append(f"Matchs analysés: {len(fixtures)}")
            
        except Exception as e:
            validation['status'] = 'failed'
            validation['errors'].append(f"Erreur validation fixtures: {e}")
        
        return validation
    
    def validate_events_data(self, season_path: Path, season_year: int) -> Dict:
        """Valider les événements de matchs (là où était le bug Haaland)."""
        validation = {
            'check_name': f'Events Data {season_year}',
            'status': 'passed',
            'details': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            events_dir = season_path / "events"
            
            if not events_dir.exists():
                validation['warnings'].append("Pas de dossier events")
                return validation
            
            event_files = list(events_dir.glob("*.json"))
            
            if len(event_files) == 0:
                validation['warnings'].append("Aucun fichier d'événements")
                return validation
            
            # Analyser échantillon de fichiers d'événements
            sample_files = event_files[:20]  # 20 fichiers max
            total_events = 0
            goal_events = 0
            haaland_goals = 0
            players_with_many_goals = {}
            
            for event_file in sample_files:
                try:
                    with open(event_file, 'r', encoding='utf-8') as f:
                        events_data = json.load(f)
                    
                    if 'response' in events_data and events_data['response']:
                        events = events_data['response']
                        total_events += len(events)
                        
                        # Analyser les événements
                        for event in events:
                            if event.get('type') == 'Goal':
                                goal_events += 1
                                player_name = event.get('player', {}).get('name', 'Unknown')
                                
                                # Traquer spécifiquement Haaland
                                if 'Haaland' in player_name:
                                    haaland_goals += 1
                                
                                # Compter buts par joueur
                                if player_name in players_with_many_goals:
                                    players_with_many_goals[player_name] += 1
                                else:
                                    players_with_many_goals[player_name] = 1
                        
                        # Vérifier nombre d'événements par match cohérent
                        if len(events) > self.thresholds['max_events_per_match']:
                            validation['warnings'].append(f"Match {event_file.stem}: {len(events)} événements (élevé)")
                
                except json.JSONDecodeError:
                    validation['errors'].append(f"Fichier JSON corrompu: {event_file.name}")
                except Exception as e:
                    validation['warnings'].append(f"Erreur lecture {event_file.name}: {e}")
            
            # Analyse des statistiques
            validation['details'].append(f"✓ Fichiers événements analysés: {len(sample_files)}")
            validation['details'].append(f"✓ Total événements: {total_events}")
            validation['details'].append(f"✓ Buts totaux: {goal_events}")
            
            # VERIFICATION CRITIQUE: Haaland
            if season_year == self.thresholds['current_season']:
                if haaland_goals > self.thresholds['max_goals_per_player_start_season']:
                    validation['status'] = 'failed' 
                    validation['critical_errors'].append(
                        f"ALERTE HAALAND: {haaland_goals} buts détectés (max attendu: {self.thresholds['max_goals_per_player_start_season']})"
                    )
                elif haaland_goals > 0:
                    validation['details'].append(f"✓ Haaland: {haaland_goals} buts (cohérent)")
            
            # Vérifier joueurs avec beaucoup de buts
            suspicious_players = {name: count for name, count in players_with_many_goals.items() 
                                if count > self.thresholds['max_goals_per_player_start_season']}
            
            if suspicious_players and season_year == self.thresholds['current_season']:
                validation['warnings'].append(f"Joueurs avec beaucoup de buts: {suspicious_players}")
            
        except Exception as e:
            validation['status'] = 'failed'
            validation['errors'].append(f"Erreur validation events: {e}")
        
        return validation
    
    def validate_players_data(self, season_path: Path, season_year: int) -> Dict:
        """Valider les données joueurs."""
        validation = {
            'check_name': f'Players Data {season_year}',
            'status': 'passed',
            'details': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            players_file = season_path / "players/players.json"
            
            if not players_file.exists():
                validation['warnings'].append("Fichier players.json manquant")
                return validation
            
            with open(players_file, 'r', encoding='utf-8') as f:
                players_data = json.load(f)
            
            if 'response' not in players_data:
                validation['errors'].append("Structure players invalide")
                return validation
            
            players = players_data['response']
            validation['details'].append(f"✓ {len(players)} joueurs trouvés")
            
            # Vérifier statistiques de quelques joueurs
            haaland_found = False
            for player in players[:50]:  # Échantillon
                player_info = player.get('player', {})
                stats = player.get('statistics', [])
                
                if 'Haaland' in player_info.get('name', ''):
                    haaland_found = True
                    
                    # Analyser stats Haaland
                    for stat in stats:
                        goals = stat.get('goals', {}).get('total', 0) or 0
                        
                        if season_year == self.thresholds['current_season']:
                            if goals > self.thresholds['max_goals_per_player_start_season']:
                                validation['status'] = 'failed'
                                validation['critical_errors'].append(
                                    f"HAALAND STATS ABERRANTES: {goals} buts en saison {season_year}"
                                )
                            else:
                                validation['details'].append(f"✓ Haaland: {goals} buts (stats cohérentes)")
            
            if season_year == self.thresholds['current_season'] and not haaland_found:
                validation['warnings'].append("Haaland non trouvé dans les stats de la saison actuelle")
            
        except Exception as e:
            validation['status'] = 'failed'
            validation['errors'].append(f"Erreur validation players: {e}")
        
        return validation
    
    def validate_competition(self, competition_path: Path) -> Dict:
        """Validation complète d'une compétition."""
        comp_name = competition_path.name
        self.logger.info(f"🔍 Validation complète: {comp_name}")
        
        competition_validation = {
            'competition_name': comp_name,
            'overall_status': 'passed',
            'checks': [],
            'summary': {
                'total_checks': 0,
                'passed': 0,
                'warnings': 0,
                'errors': 0,
                'critical_errors': 0
            }
        }
        
        # 1. Validation structure saisons
        season_check = self.validate_season_consistency(competition_path)
        competition_validation['checks'].append(season_check)
        
        # 2. Validation de chaque saison
        seasons_path = competition_path / "seasons"
        if seasons_path.exists():
            for season_dir in seasons_path.iterdir():
                if season_dir.is_dir() and season_dir.name.isdigit():
                    season_year = int(season_dir.name)
                    
                    # Validation fixtures
                    fixtures_check = self.validate_fixtures_data(season_dir, season_year)
                    competition_validation['checks'].append(fixtures_check)
                    
                    # Validation events (critique pour Haaland)
                    events_check = self.validate_events_data(season_dir, season_year)
                    competition_validation['checks'].append(events_check)
                    
                    # Validation players
                    players_check = self.validate_players_data(season_dir, season_year)
                    competition_validation['checks'].append(players_check)
        
        # Calculer résumé
        for check in competition_validation['checks']:
            competition_validation['summary']['total_checks'] += 1
            
            if check['status'] == 'passed':
                competition_validation['summary']['passed'] += 1
            elif check['status'] == 'failed':
                competition_validation['summary']['errors'] += 1
                competition_validation['overall_status'] = 'failed'
            
            competition_validation['summary']['warnings'] += len(check.get('warnings', []))
            
            if 'critical_errors' in check:
                competition_validation['summary']['critical_errors'] += len(check['critical_errors'])
                if check['critical_errors']:
                    competition_validation['overall_status'] = 'critical_failure'
        
        return competition_validation
    
    def run_complete_validation(self) -> Dict:
        """Lancer validation complète de toutes les données."""
        self.logger.info("🚀 DEBUT VALIDATION COMPLETE DES DONNEES")
        
        competitions_path = Path("data/competitions")
        
        if not competitions_path.exists():
            self.validation_results['critical_errors'].append("Dossier competitions manquant")
            return self.validation_results
        
        # Valider chaque compétition
        for comp_dir in competitions_path.iterdir():
            if comp_dir.is_dir():
                try:
                    comp_validation = self.validate_competition(comp_dir)
                    self.validation_results['validation_details'][comp_dir.name] = comp_validation
                    
                    # Mettre à jour statistiques globales
                    self.validation_results['total_checks'] += comp_validation['summary']['total_checks']
                    self.validation_results['passed_checks'] += comp_validation['summary']['passed']
                    self.validation_results['failed_checks'] += comp_validation['summary']['errors']
                    self.validation_results['warnings'].extend([f"{comp_dir.name}: {w}" for check in comp_validation['checks'] for w in check.get('warnings', [])])
                    self.validation_results['errors'].extend([f"{comp_dir.name}: {e}" for check in comp_validation['checks'] for e in check.get('errors', [])])
                    
                    # Erreurs critiques
                    for check in comp_validation['checks']:
                        if 'critical_errors' in check:
                            self.validation_results['critical_errors'].extend([f"{comp_dir.name}: {ce}" for ce in check['critical_errors']])
                    
                except Exception as e:
                    self.logger.error(f"❌ Erreur validation {comp_dir.name}: {e}")
                    self.validation_results['critical_errors'].append(f"{comp_dir.name}: {e}")
        
        # Résumé final
        total_checks = self.validation_results['total_checks']
        passed_checks = self.validation_results['passed_checks']
        success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        self.logger.info("✅ VALIDATION TERMINEE")
        self.logger.info(f"📊 Résultats: {passed_checks}/{total_checks} checks passés ({success_rate:.1f}%)")
        self.logger.info(f"⚠️ Warnings: {len(self.validation_results['warnings'])}")
        self.logger.info(f"❌ Errors: {len(self.validation_results['errors'])}")
        self.logger.info(f"🚨 Critical Errors: {len(self.validation_results['critical_errors'])}")
        
        # Sauvegarder résultats
        results_path = Path("data/validation_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
        
        return self.validation_results

def main():
    """Fonction principale de validation."""
    print("="*70)
    print("SYSTEME DE VALIDATION DES DONNEES V2.0")
    print("="*70)
    print("Objectif: Valider integrite des nouvelles donnees")
    print("Detection: Anomalies type Haaland 18 buts")
    print("Verification: Coherence saisons et statistiques")
    print("="*70)
    
    validator = DataValidationSystem()
    
    try:
        results = validator.run_complete_validation()
        
        # Afficher résumé
        print(f"\n📊 RESULTATS VALIDATION:")
        print(f"Checks totaux: {results['total_checks']}")
        print(f"Succès: {results['passed_checks']}")
        print(f"Warnings: {len(results['warnings'])}")
        print(f"Erreurs: {len(results['errors'])}")
        print(f"Erreurs critiques: {len(results['critical_errors'])}")
        
        if results['critical_errors']:
            print(f"\nERREURS CRITIQUES:")
            for error in results['critical_errors']:
                print(f"  - {error}")
        
        if results['critical_errors']:
            print(f"\nVALIDATION ECHOUEE - Donnees non fiables")
        elif results['errors']:
            print(f"\nVALIDATION PARTIELLE - Erreurs mineures detectees")
        else:
            print(f"\nVALIDATION REUSSIE - Donnees fiables")
        
    except Exception as e:
        print(f"ERREUR CRITIQUE VALIDATION: {e}")

if __name__ == "__main__":
    main()