"""
⚡ REALTIME RECALIBRATION ENGINE - RECALIBRAGE TEMPS RÉEL INTELLIGENT
Système de recalibrage automatique des prédictions selon événements pré-match

Version: 3.0 - Phase 3 ML Transformation
Créé: 23 août 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
from datetime import datetime, timedelta
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# Import des composants précédents
try:
    from intelligent_betting_coupon import IntelligentBettingCoupon, BettingPrediction
    from confidence_scoring_engine import AdvancedConfidenceScorer
except ImportError as e:
    print(f"Warning: Import manque - {e}")

class EventImpactCalculator:
    """Calculateur d'impact des événements sur les prédictions"""
    
    def __init__(self):
        # Matrice d'impact des événements par type de prédiction
        self.impact_matrix = self._initialize_impact_matrix()
        self.player_importance_cache = {}
        
    def _initialize_impact_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialise la matrice d'impact événement -> prédiction"""
        
        return {
            # Changements de composition
            'key_player_injured': {
                'match_result': -0.15,
                'total_goals': -0.10,
                'both_teams_scored': -0.08,
                'over_2_5_goals': -0.12,
                'first_half_result': -0.10,
                'corners_total': -0.05,
                'cards_total': 0.05  # Plus de cartons si remplaçant nerveux
            },
            'goalkeeper_change': {
                'match_result': -0.20,
                'total_goals': -0.15,
                'both_teams_scored': -0.18,
                'clean_sheet_home': -0.25,
                'clean_sheet_away': -0.25,
                'over_2_5_goals': 0.12
            },
            'striker_injured': {
                'match_result': -0.12,
                'total_goals': -0.18,
                'both_teams_scored': -0.15,
                'over_2_5_goals': -0.20,
                'home_goals': -0.25,
                'away_goals': -0.25
            },
            'defender_injured': {
                'match_result': -0.08,
                'total_goals': 0.08,
                'both_teams_scored': 0.12,
                'over_2_5_goals': 0.15,
                'clean_sheet_home': -0.30,
                'clean_sheet_away': -0.30
            },
            
            # Conditions météorologiques
            'heavy_rain': {
                'match_result': -0.05,
                'total_goals': -0.15,
                'both_teams_scored': -0.12,
                'over_2_5_goals': -0.18,
                'corners_total': -0.10,
                'cards_total': 0.08
            },
            'strong_wind': {
                'total_goals': -0.10,
                'over_2_5_goals': -0.12,
                'corners_total': 0.15,  # Plus de corners avec vent
                'both_teams_scored': -0.08
            },
            'extreme_cold': {
                'total_goals': -0.08,
                'cards_total': 0.10,  # Plus de fautes par froid
                'both_teams_scored': -0.05
            },
            'extreme_heat': {
                'total_goals': -0.12,
                'over_2_5_goals': -0.15,
                'cards_total': 0.12  # Plus de cartons par fatigue
            },
            
            # Changements tactiques
            'formation_change': {
                'match_result': -0.08,
                'total_goals': -0.10,
                'both_teams_scored': -0.08,
                'corners_total': -0.05
            },
            'defensive_lineup': {
                'total_goals': -0.20,
                'over_2_5_goals': -0.25,
                'both_teams_scored': -0.18,
                'clean_sheet_home': 0.15,
                'clean_sheet_away': 0.15
            },
            'attacking_lineup': {
                'total_goals': 0.15,
                'over_2_5_goals': 0.20,
                'both_teams_scored': 0.12,
                'clean_sheet_home': -0.20,
                'clean_sheet_away': -0.20
            },
            
            # Événements externes
            'referee_change': {
                'cards_total': -0.15,  # Impact variable selon referee
                'match_result': -0.05
            },
            'crowd_restrictions': {
                'match_result': -0.08,  # Moins d'avantage à domicile
                'total_goals': -0.05
            },
            'media_pressure': {
                'match_result': -0.10,
                'cards_total': 0.08
            },
            'stadium_issue': {
                'match_result': -0.12,
                'total_goals': -0.08
            }
        }
    
    def calculate_event_impact(self, event_type: str, event_details: Dict, 
                             prediction_type: str, team_affected: str = 'home') -> float:
        """Calcule l'impact d'un événement sur un type de prédiction"""
        
        # Impact de base selon la matrice
        base_impact = self.impact_matrix.get(event_type, {}).get(prediction_type, 0.0)
        
        # Modulation selon les détails de l'événement
        impact_modifier = 1.0
        
        # Importance du joueur affecté
        if 'player_name' in event_details:
            player_importance = self._get_player_importance(
                event_details['player_name'], event_details.get('team', team_affected)
            )
            impact_modifier *= (0.5 + player_importance)  # 0.5 à 1.5x
        
        # Sévérité de l'événement
        if 'severity' in event_details:
            severity_multipliers = {'low': 0.5, 'medium': 1.0, 'high': 1.5, 'critical': 2.0}
            impact_modifier *= severity_multipliers.get(event_details['severity'], 1.0)
        
        # Timing de l'événement (plus proche = plus d'impact)
        if 'minutes_before_kickoff' in event_details:
            minutes = event_details['minutes_before_kickoff']
            # Impact maximum si moins de 30 min, diminue progressivement
            time_factor = max(0.3, 1.0 - (minutes - 30) / 120) if minutes > 30 else 1.0
            impact_modifier *= time_factor
        
        # Impact directionnel (positif/négatif selon équipe)
        final_impact = base_impact * impact_modifier
        
        # Inversion si événement affecte l'équipe visiteur et prédiction home-specific
        if team_affected == 'away' and prediction_type in ['home_goals', 'clean_sheet_home']:
            final_impact = -final_impact
        
        return final_impact
    
    def _get_player_importance(self, player_name: str, team: str) -> float:
        """Évalue l'importance d'un joueur (0.0 à 1.0)"""
        
        cache_key = f"{team}_{player_name}"
        
        if cache_key in self.player_importance_cache:
            return self.player_importance_cache[cache_key]
        
        # Simulation basée sur des patterns de noms (en réalité, base de données)
        importance = 0.5  # Valeur par défaut
        
        # Patterns pour joueurs importants (simulation)
        important_patterns = ['captain', 'star', 'key', 'main']
        if any(pattern in player_name.lower() for pattern in important_patterns):
            importance = 0.9
        
        # Patterns pour remplaçants
        bench_patterns = ['sub', 'reserve', 'backup']
        if any(pattern in player_name.lower() for pattern in bench_patterns):
            importance = 0.2
        
        # Cache du résultat
        self.player_importance_cache[cache_key] = importance
        
        return importance

class RealtimeDataMonitor:
    """Moniteur de données temps réel pré-match"""
    
    def __init__(self, callback_function: Callable = None):
        self.is_monitoring = False
        self.monitored_matches = {}
        self.data_sources = []
        self.callback_function = callback_function
        self.monitor_thread = None
        self.update_intervals = {
            'lineups': 300,      # 5 minutes
            'weather': 600,      # 10 minutes  
            'odds': 60,          # 1 minute
            'news': 180,         # 3 minutes
            'referee': 1800      # 30 minutes
        }
        
    def add_match_monitoring(self, match_id: str, match_data: Dict, 
                           monitoring_timeline: Dict = None):
        """Ajoute un match à la surveillance temps réel"""
        
        if monitoring_timeline is None:
            monitoring_timeline = {
                120: ['lineups', 'weather', 'odds'],      # 2h avant
                90: ['lineups', 'weather', 'odds', 'news'], # 1h30 avant
                60: ['lineups', 'weather', 'odds', 'news'], # 1h avant
                30: ['lineups', 'weather', 'odds'],         # 30 min avant
                15: ['lineups', 'weather'],                 # 15 min avant
                0: ['final_check']                          # Coup d'envoi
            }
        
        self.monitored_matches[match_id] = {
            'match_data': match_data,
            'timeline': monitoring_timeline,
            'last_updates': {},
            'active_monitoring': True,
            'kickoff_time': pd.to_datetime(match_data.get('kickoff_time', '2025-01-25 15:00:00'))
        }
        
        print(f"Match {match_id} ajouté à la surveillance temps réel")
    
    def start_monitoring(self):
        """Démarre la surveillance temps réel"""
        
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        print("Surveillance temps réel démarrée")
    
    def stop_monitoring(self):
        """Arrête la surveillance temps réel"""
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        print("Surveillance temps réel arrêtée")
    
    def _monitoring_loop(self):
        """Boucle principale de surveillance"""
        
        while self.is_monitoring:
            current_time = pd.Timestamp.now()
            
            for match_id, match_info in self.monitored_matches.items():
                if not match_info['active_monitoring']:
                    continue
                
                kickoff_time = match_info['kickoff_time']
                minutes_until_kickoff = (kickoff_time - current_time).total_seconds() / 60
                
                # Vérifier les jalons de surveillance
                for timeline_minutes in match_info['timeline']:
                    if (timeline_minutes - 5) <= minutes_until_kickoff <= (timeline_minutes + 5):
                        data_types = match_info['timeline'][timeline_minutes]
                        
                        for data_type in data_types:
                            self._check_data_source(match_id, data_type, match_info)
                
                # Désactiver si match commencé
                if minutes_until_kickoff < -10:  # 10 min après début
                    match_info['active_monitoring'] = False
            
            time.sleep(30)  # Vérification toutes les 30 secondes
    
    def _check_data_source(self, match_id: str, data_type: str, match_info: Dict):
        """Vérifie une source de données spécifique"""
        
        # Simulation de collecte de données (remplacer par vrais APIs)
        updates = self._simulate_data_collection(data_type, match_info['match_data'])
        
        if updates:
            print(f"Mise à jour {data_type} détectée pour match {match_id}")
            
            # Appel du callback si configuré
            if self.callback_function:
                self.callback_function(match_id, data_type, updates)
            
            # Sauvegarde de la dernière mise à jour
            match_info['last_updates'][data_type] = {
                'timestamp': pd.Timestamp.now(),
                'data': updates
            }
    
    def _simulate_data_collection(self, data_type: str, match_data: Dict) -> Optional[Dict]:
        """Simule la collecte de données depuis différentes sources"""
        
        # Probabilité de changement selon le type
        change_probabilities = {
            'lineups': 0.3,
            'weather': 0.2,
            'odds': 0.8,
            'news': 0.1,
            'referee': 0.05,
            'final_check': 0.1
        }
        
        if np.random.random() > change_probabilities.get(data_type, 0.1):
            return None  # Pas de changement
        
        # Simulation de données selon le type
        if data_type == 'lineups':
            return self._simulate_lineup_changes()
        elif data_type == 'weather':
            return self._simulate_weather_update()
        elif data_type == 'odds':
            return self._simulate_odds_changes()
        elif data_type == 'news':
            return self._simulate_news_update()
        elif data_type == 'referee':
            return self._simulate_referee_change()
        
        return None
    
    def _simulate_lineup_changes(self) -> Dict:
        """Simule des changements de composition"""
        
        change_types = ['key_player_injured', 'goalkeeper_change', 'striker_injured', 'defender_injured']
        change_type = np.random.choice(change_types)
        
        return {
            'event_type': change_type,
            'player_name': f'Player_{np.random.randint(1, 23)}',
            'team': np.random.choice(['home', 'away']),
            'severity': np.random.choice(['medium', 'high']),
            'minutes_before_kickoff': np.random.randint(15, 90),
            'replacement_player': f'Sub_Player_{np.random.randint(1, 10)}',
            'reason': 'injury' if 'injured' in change_type else 'tactical'
        }
    
    def _simulate_weather_update(self) -> Dict:
        """Simule des mises à jour météo"""
        
        weather_events = ['heavy_rain', 'strong_wind', 'extreme_cold', 'extreme_heat']
        event = np.random.choice(weather_events)
        
        return {
            'event_type': event,
            'severity': np.random.choice(['medium', 'high']),
            'temperature': np.random.randint(-5, 35),
            'humidity': np.random.randint(40, 90),
            'wind_speed': np.random.randint(5, 35),
            'precipitation': np.random.randint(0, 100),
            'minutes_before_kickoff': np.random.randint(30, 120)
        }
    
    def _simulate_odds_changes(self) -> Dict:
        """Simule des changements de cotes"""
        
        return {
            'event_type': 'odds_movement',
            'changes': {
                'match_result_home': np.random.uniform(-0.3, 0.3),
                'total_goals_over': np.random.uniform(-0.2, 0.2),
                'both_teams_scored': np.random.uniform(-0.1, 0.1)
            },
            'volume_spike': np.random.choice([True, False]),
            'minutes_before_kickoff': np.random.randint(5, 120)
        }
    
    def _simulate_news_update(self) -> Dict:
        """Simule des mises à jour d'actualités"""
        
        news_types = ['media_pressure', 'crowd_restrictions', 'stadium_issue']
        news_type = np.random.choice(news_types)
        
        return {
            'event_type': news_type,
            'severity': np.random.choice(['low', 'medium']),
            'description': f'Simulated {news_type} event',
            'minutes_before_kickoff': np.random.randint(60, 180)
        }
    
    def _simulate_referee_change(self) -> Dict:
        """Simule un changement d'arbitre"""
        
        return {
            'event_type': 'referee_change',
            'original_referee': 'Referee_A',
            'new_referee': 'Referee_B',
            'reason': 'illness',
            'severity': 'medium',
            'minutes_before_kickoff': np.random.randint(120, 300)
        }

class RealtimeRecalibrationEngine:
    """Engine principal de recalibrage temps réel"""
    
    def __init__(self):
        self.impact_calculator = EventImpactCalculator()
        self.data_monitor = RealtimeDataMonitor(callback_function=self._handle_data_update)
        self.confidence_scorer = None
        self.coupon_system = None
        
        # Configuration
        self.recalibration_config = {
            'min_impact_threshold': 0.05,  # Impact minimum pour déclencher recalibrage
            'max_confidence_adjustment': 20.0,  # Ajustement maximum en points
            'batch_update_interval': 60,  # Secondes entre mises à jour groupées
            'auto_approve_threshold': 0.10  # Auto-approuve si impact < 10%
        }
        
        # Historique et cache
        self.recalibration_history = []
        self.pending_recalibrations = {}
        
    def initialize_components(self):
        """Initialise les composants nécessaires"""
        
        try:
            self.confidence_scorer = AdvancedConfidenceScorer()
            self.coupon_system = IntelligentBettingCoupon()
            self.coupon_system.initialize_components()
            print("Composants de recalibrage initialisés")
        except Exception as e:
            print(f"Erreur initialisation: {e}")
    
    def start_realtime_monitoring(self, coupons_to_monitor: List[str]):
        """Démarre la surveillance temps réel pour les coupons actifs"""
        
        print(f"Démarrage surveillance pour {len(coupons_to_monitor)} coupons")
        
        # Configuration surveillance pour chaque coupon
        for coupon_id in coupons_to_monitor:
            coupon = self._get_coupon_by_id(coupon_id)
            if coupon:
                matches_in_coupon = self._extract_matches_from_coupon(coupon)
                
                for match_id, match_data in matches_in_coupon.items():
                    self.data_monitor.add_match_monitoring(match_id, match_data)
        
        # Démarrage du monitoring
        self.data_monitor.start_monitoring()
    
    def _handle_data_update(self, match_id: str, data_type: str, update_data: Dict):
        """Gère les mises à jour de données en temps réel"""
        
        print(f"Traitement mise à jour {data_type} pour match {match_id}")
        
        # Calcul impact sur tous les coupons concernés
        affected_coupons = self._find_coupons_with_match(match_id)
        
        for coupon_id in affected_coupons:
            impact_analysis = self._analyze_update_impact(coupon_id, match_id, update_data)
            
            if impact_analysis['requires_recalibration']:
                self._queue_recalibration(coupon_id, match_id, update_data, impact_analysis)
    
    def _analyze_update_impact(self, coupon_id: str, match_id: str, update_data: Dict) -> Dict:
        """Analyse l'impact d'une mise à jour sur un coupon"""
        
        coupon = self._get_coupon_by_id(coupon_id)
        if not coupon:
            return {'requires_recalibration': False}
        
        total_impact = 0.0
        prediction_impacts = []
        
        # Analyse impact sur chaque prédiction du coupon
        for pred in coupon.get('predictions', []):
            # Filtre prédictions du match concerné
            if not self._prediction_matches_event(pred, match_id, update_data):
                continue
            
            impact = self.impact_calculator.calculate_event_impact(
                update_data['event_type'],
                update_data,
                pred['prediction_type'],
                update_data.get('team', 'home')
            )
            
            if abs(impact) > self.recalibration_config['min_impact_threshold']:
                prediction_impacts.append({
                    'prediction_type': pred['prediction_type'],
                    'original_confidence': pred['confidence_score'],
                    'impact': impact,
                    'new_confidence_estimate': max(30, min(95, pred['confidence_score'] + impact * 100))
                })
                
                total_impact += abs(impact)
        
        return {
            'requires_recalibration': total_impact > self.recalibration_config['min_impact_threshold'],
            'total_impact': total_impact,
            'affected_predictions': len(prediction_impacts),
            'prediction_impacts': prediction_impacts,
            'auto_approve': total_impact < self.recalibration_config['auto_approve_threshold']
        }
    
    def _queue_recalibration(self, coupon_id: str, match_id: str, 
                           update_data: Dict, impact_analysis: Dict):
        """Met en file d'attente un recalibrage"""
        
        recalibration_id = f"{coupon_id}_{match_id}_{int(time.time())}"
        
        recalibration_task = {
            'id': recalibration_id,
            'coupon_id': coupon_id,
            'match_id': match_id,
            'update_data': update_data,
            'impact_analysis': impact_analysis,
            'timestamp': pd.Timestamp.now(),
            'status': 'pending',
            'auto_approve': impact_analysis.get('auto_approve', False)
        }
        
        self.pending_recalibrations[recalibration_id] = recalibration_task
        
        # Auto-approbation si impact faible
        if recalibration_task['auto_approve']:
            self._execute_recalibration(recalibration_id)
        else:
            print(f"⚠️ Recalibrage {recalibration_id} nécessite approbation (impact: {impact_analysis['total_impact']:.3f})")
    
    def approve_recalibration(self, recalibration_id: str) -> Dict:
        """Approuve et exécute un recalibrage"""
        
        if recalibration_id not in self.pending_recalibrations:
            return {'error': 'Recalibrage non trouvé'}
        
        return self._execute_recalibration(recalibration_id)
    
    def _execute_recalibration(self, recalibration_id: str) -> Dict:
        """Exécute un recalibrage approuvé"""
        
        task = self.pending_recalibrations.get(recalibration_id)
        if not task:
            return {'error': 'Tâche non trouvée'}
        
        try:
            # Exécution du recalibrage via le système de coupons
            if self.coupon_system:
                recalibrated_coupon = self.coupon_system.recalibrate_coupon_realtime(
                    task['coupon_id'], 
                    task['update_data']
                )
                
                if recalibrated_coupon:
                    # Sauvegarde dans l'historique
                    self.recalibration_history.append({
                        'recalibration_id': recalibration_id,
                        'original_coupon_id': task['coupon_id'],
                        'new_coupon_id': recalibrated_coupon['coupon_id'],
                        'impact_analysis': task['impact_analysis'],
                        'execution_timestamp': pd.Timestamp.now(),
                        'status': 'completed'
                    })
                    
                    # Suppression de la queue
                    del self.pending_recalibrations[recalibration_id]
                    
                    print(f"✅ Recalibrage {recalibration_id} exécuté avec succès")
                    
                    return {
                        'status': 'success',
                        'new_coupon_id': recalibrated_coupon['coupon_id'],
                        'impact_summary': task['impact_analysis']
                    }
            
            return {'error': 'Système de coupons non initialisé'}
            
        except Exception as e:
            task['status'] = 'failed'
            return {'error': str(e)}
    
    def get_pending_recalibrations(self) -> List[Dict]:
        """Retourne les recalibrages en attente d'approbation"""
        
        pending = []
        for recal_id, task in self.pending_recalibrations.items():
            if task['status'] == 'pending' and not task['auto_approve']:
                pending.append({
                    'recalibration_id': recal_id,
                    'coupon_id': task['coupon_id'],
                    'match_id': task['match_id'],
                    'event_type': task['update_data']['event_type'],
                    'total_impact': task['impact_analysis']['total_impact'],
                    'affected_predictions': task['impact_analysis']['affected_predictions'],
                    'timestamp': task['timestamp'].isoformat()
                })
        
        return pending
    
    def get_recalibration_statistics(self) -> Dict:
        """Statistiques du système de recalibrage"""
        
        return {
            'total_recalibrations': len(self.recalibration_history),
            'pending_recalibrations': len([t for t in self.pending_recalibrations.values() if t['status'] == 'pending']),
            'auto_approved_count': len([h for h in self.recalibration_history if h.get('auto_approved', False)]),
            'manual_approved_count': len([h for h in self.recalibration_history if not h.get('auto_approved', False)]),
            'average_impact': np.mean([h['impact_analysis']['total_impact'] for h in self.recalibration_history]) if self.recalibration_history else 0.0,
            'monitoring_active': self.data_monitor.is_monitoring
        }
    
    def _get_coupon_by_id(self, coupon_id: str) -> Optional[Dict]:
        """Récupère un coupon par son ID"""
        
        if not self.coupon_system:
            return None
        
        for coupon in self.coupon_system.coupon_history:
            if coupon['coupon_id'] == coupon_id:
                return coupon
        
        return None
    
    def _extract_matches_from_coupon(self, coupon: Dict) -> Dict[str, Dict]:
        """Extrait les matchs d'un coupon"""
        
        matches = {}
        
        for pred in coupon.get('predictions', []):
            match_context = pred.get('match_context', {})
            match_id = f"{match_context.get('home_team', 'A')}_vs_{match_context.get('away_team', 'B')}"
            
            if match_id not in matches:
                matches[match_id] = match_context
        
        return matches
    
    def _find_coupons_with_match(self, match_id: str) -> List[str]:
        """Trouve tous les coupons contenant un match donné"""
        
        if not self.coupon_system:
            return []
        
        coupon_ids = []
        
        for coupon in self.coupon_system.coupon_history:
            matches = self._extract_matches_from_coupon(coupon)
            if match_id in matches:
                coupon_ids.append(coupon['coupon_id'])
        
        return coupon_ids
    
    def _prediction_matches_event(self, prediction: Dict, match_id: str, update_data: Dict) -> bool:
        """Vérifie si une prédiction est affectée par un événement"""
        
        # Vérification si la prédiction concerne le bon match
        match_context = prediction.get('match_context', {})
        pred_match_id = f"{match_context.get('home_team', 'A')}_vs_{match_context.get('away_team', 'B')}"
        
        if pred_match_id != match_id:
            return False
        
        # Vérification si le type d'événement affecte le type de prédiction
        event_type = update_data.get('event_type', '')
        prediction_type = prediction.get('prediction_type', '')
        
        impact = self.impact_calculator.impact_matrix.get(event_type, {}).get(prediction_type, 0.0)
        
        return abs(impact) > 0.01  # Impact non-négligeable

def test_realtime_recalibration_system():
    """Test du système de recalibrage temps réel"""
    
    print("=== TEST REALTIME RECALIBRATION ENGINE ===")
    
    # Initialisation
    recalibration_engine = RealtimeRecalibrationEngine()
    recalibration_engine.initialize_components()
    
    # Génération d'un coupon de test
    if recalibration_engine.coupon_system:
        matches_data = [{
            'home_team': 'Arsenal',
            'away_team': 'Chelsea', 
            'league': 'Premier_League',
            'match_importance': 'high',
            'kickoff_time': '2025-01-25 15:00:00'
        }]
        
        test_coupon = recalibration_engine.coupon_system.generate_intelligent_coupon(matches_data)
        
        if test_coupon['status'] == 'success':
            print(f"✅ Coupon test généré: {test_coupon['coupon_id']}")
            
            # Test surveillance temps réel
            print(f"\\n--- Test Surveillance Temps Réel ---")
            
            recalibration_engine.start_realtime_monitoring([test_coupon['coupon_id']])
            
            # Simulation d'événements
            print(f"\\n--- Simulation Événements ---")
            
            # Événement 1: Blessure joueur clé
            injury_event = {
                'event_type': 'key_player_injured',
                'player_name': 'Star_Player',
                'team': 'home',
                'severity': 'high',
                'minutes_before_kickoff': 45
            }
            
            match_id = 'Arsenal_vs_Chelsea'
            impact_analysis = recalibration_engine._analyze_update_impact(
                test_coupon['coupon_id'], match_id, injury_event
            )
            
            print(f"Impact blessure joueur clé:")
            print(f"   Recalibrage requis: {'Oui' if impact_analysis['requires_recalibration'] else 'Non'}")
            print(f"   Impact total: {impact_analysis['total_impact']:.3f}")
            print(f"   Prédictions affectées: {impact_analysis['affected_predictions']}")
            
            if impact_analysis['requires_recalibration']:
                recalibration_engine._queue_recalibration(
                    test_coupon['coupon_id'], match_id, injury_event, impact_analysis
                )
            
            # Événement 2: Changement météo
            weather_event = {
                'event_type': 'heavy_rain',
                'severity': 'medium',
                'minutes_before_kickoff': 60,
                'temperature': 8,
                'precipitation': 85
            }
            
            weather_impact = recalibration_engine._analyze_update_impact(
                test_coupon['coupon_id'], match_id, weather_event
            )
            
            print(f"\\nImpact changement météo:")
            print(f"   Recalibrage requis: {'Oui' if weather_impact['requires_recalibration'] else 'Non'}")
            print(f"   Impact total: {weather_impact['total_impact']:.3f}")
            
            if weather_impact['requires_recalibration']:
                recalibration_engine._queue_recalibration(
                    test_coupon['coupon_id'], match_id, weather_event, weather_impact
                )
            
            # Gestion des recalibrages en attente
            print(f"\\n--- Gestion Recalibrages ---")
            
            pending = recalibration_engine.get_pending_recalibrations()
            print(f"Recalibrages en attente: {len(pending)}")
            
            for recal in pending:
                print(f"   • {recal['recalibration_id']}")
                print(f"     Événement: {recal['event_type']}")
                print(f"     Impact: {recal['total_impact']:.3f}")
                
                # Approbation du recalibrage
                approval_result = recalibration_engine.approve_recalibration(recal['recalibration_id'])
                
                if approval_result.get('status') == 'success':
                    print(f"     ✅ Recalibrage approuvé → {approval_result['new_coupon_id']}")
                else:
                    print(f"     ❌ Erreur: {approval_result.get('error', 'Inconnue')}")
            
            # Statistiques système
            print(f"\\n--- Statistiques Système ---")
            stats = recalibration_engine.get_recalibration_statistics()
            
            print(f"   Total recalibrages: {stats['total_recalibrations']}")
            print(f"   En attente: {stats['pending_recalibrations']}")
            print(f"   Auto-approuvés: {stats['auto_approved_count']}")
            print(f"   Impact moyen: {stats['average_impact']:.3f}")
            print(f"   Surveillance active: {'Oui' if stats['monitoring_active'] else 'Non'}")
            
            # Arrêt de la surveillance
            recalibration_engine.data_monitor.stop_monitoring()
        
        else:
            print(f"❌ Échec génération coupon test")
    
    else:
        print("❌ Système de coupons non initialisé")
    
    print("\\n=== TEST TERMINÉ ===")

if __name__ == "__main__":
    test_realtime_recalibration_system()