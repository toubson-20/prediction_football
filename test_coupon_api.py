"""
🧪 TESTS COMPLETS POUR L'API COUPONS
Test tous les endpoints et fonctionnalités
"""

import requests
import json
from datetime import datetime, timedelta
import time

class CouponAPITester:
    """Testeur complet pour l'API de coupons"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
        
    def test_endpoint(self, name, method, endpoint, data=None, expected_status=200):
        """Tester un endpoint spécifique"""
        print(f"\n🧪 Test: {name}")
        print(f"   {method} {endpoint}")
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                response = requests.get(url)
            elif method.upper() == "POST":
                response = requests.post(url, json=data)
            else:
                raise ValueError(f"Méthode {method} non supportée")
            
            success = response.status_code == expected_status
            
            print(f"   Status: {response.status_code} {'OK' if success else 'ERREUR'}")
            
            if success and response.content:
                result = response.json()
                if isinstance(result, dict):
                    if 'predictions' in result:
                        print(f"   Predictions: {len(result['predictions'])}")
                    if 'total_odds' in result:
                        print(f"   Cote totale: {result['total_odds']}")
                    if 'confidence_average' in result:
                        print(f"   Confiance moyenne: {result['confidence_average']}%")
                
                print(f"   Reponse JSON valide")
            
            self.test_results.append({
                'name': name,
                'success': success,
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds()
            })
            
            return response.json() if response.content else None
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            self.test_results.append({
                'name': name,
                'success': False,
                'error': str(e)
            })
            return None
    
    def run_all_tests(self):
        """Exécuter tous les tests"""
        print("🚀 DÉBUT DES TESTS API COUPONS")
        print("="*50)
        
        # Test 1: Health check
        self.test_endpoint(
            "Health Check",
            "GET",
            "/health"
        )
        
        # Test 2: Root endpoint
        self.test_endpoint(
            "Root Endpoint",
            "GET",
            "/"
        )
        
        # Test 3: Ligues disponibles
        self.test_endpoint(
            "Ligues Disponibles",
            "GET",
            "/leagues"
        )
        
        # Test 4: Types de prédictions
        self.test_endpoint(
            "Types de Prédictions",
            "GET",
            "/predictions/types"
        )
        
        # Test 5: Coupon Premier League
        pl_request = {
            "leagues": ["premier_league"],
            "risk_profile": "balanced",
            "min_confidence": 75.0,
            "max_predictions": 5,
            "min_odds": 1.3,
            "max_odds": 3.0
        }
        
        self.test_endpoint(
            "Coupon Premier League",
            "POST",
            "/coupons/league/premier_league",
            pl_request
        )
        
        # Test 6: Coupon Bundesliga
        bl_request = {
            "leagues": ["bundesliga"],
            "risk_profile": "conservative",
            "min_confidence": 85.0,
            "max_predictions": 3,
            "min_odds": 1.2,
            "max_odds": 1.8
        }
        
        self.test_endpoint(
            "Coupon Bundesliga",
            "POST",
            "/coupons/league/bundesliga",
            bl_request
        )
        
        # Test 7: Coupon Multi-Ligues
        multi_request = {
            "leagues": ["premier_league", "la_liga", "serie_a"],
            "risk_profile": "balanced",
            "min_confidence": 70.0,
            "max_predictions": 9,
            "min_odds": 1.4,
            "max_odds": 2.5,
            "prediction_types": ["match_result", "both_teams_score"]
        }
        
        self.test_endpoint(
            "Coupon Multi-Ligues",
            "POST",
            "/coupons/multi-league",
            multi_request
        )
        
        # Test 8: Coupon Toutes Ligues
        self.test_endpoint(
            "Coupon Toutes Ligues",
            "POST",
            "/coupons/all-leagues?risk_profile=aggressive&min_confidence=60&max_predictions=12"
        )
        
        # Test 9: Coupon Agressif
        aggressive_request = {
            "leagues": ["premier_league", "bundesliga"],
            "risk_profile": "aggressive",
            "min_confidence": 60.0,
            "max_predictions": 6,
            "min_odds": 2.0,
            "max_odds": 5.0
        }
        
        self.test_endpoint(
            "Coupon Agressif",
            "POST",
            "/coupons/multi-league",
            aggressive_request
        )
        
        # Test 10: Validation erreurs
        invalid_request = {
            "leagues": [],  # Vide - doit échouer
            "min_confidence": -10  # Invalide
        }
        
        self.test_endpoint(
            "Validation Erreurs",
            "POST",
            "/coupons/multi-league",
            invalid_request,
            expected_status=422  # Validation error
        )
        
        self.print_summary()
    
    def print_summary(self):
        """Afficher résumé des tests"""
        print("\n" + "="*50)
        print("📊 RÉSUMÉ DES TESTS")
        print("="*50)
        
        total_tests = len(self.test_results)
        successful_tests = len([t for t in self.test_results if t['success']])
        failed_tests = total_tests - successful_tests
        
        print(f"Total tests: {total_tests}")
        print(f"✅ Réussis: {successful_tests}")
        print(f"❌ Échoués: {failed_tests}")
        print(f"Taux de réussite: {(successful_tests/total_tests*100):.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Tests échoués:")
            for test in self.test_results:
                if not test['success']:
                    print(f"   - {test['name']}: {test.get('error', 'Échec')}")
        
        # Performance moyenne
        response_times = [t.get('response_time', 0) for t in self.test_results if 'response_time' in t]
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            print(f"\n⚡ Temps de réponse moyen: {avg_time:.3f}s")

def test_specific_scenarios():
    """Tests de scénarios spécifiques"""
    print("\n🎯 TESTS DE SCÉNARIOS SPÉCIFIQUES")
    print("="*50)
    
    base_url = "http://localhost:8000"
    
    # Scénario 1: Frontend demande coupon Conservative pour investisseur prudent
    print("\n💼 Scénario Investisseur Prudent")
    conservative_request = {
        "leagues": ["premier_league", "la_liga"],
        "risk_profile": "conservative", 
        "min_confidence": 90.0,
        "max_predictions": 4,
        "min_odds": 1.2,
        "max_odds": 1.6
    }
    
    response = requests.post(f"{base_url}/coupons/multi-league", json=conservative_request)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Coupon prudent généré:")
        print(f"   - {len(result['predictions'])} prédictions")
        print(f"   - Cote totale: {result['total_odds']}")
        print(f"   - Confiance: {result['confidence_average']}%")
        print(f"   - Risque: {result['risk_score']}/100")
    
    # Scénario 2: Parieur expérimenté cherche gros gains
    print("\n🚀 Scénario Parieur Expérimenté")
    expert_request = {
        "leagues": ["bundesliga", "serie_a", "ligue_1"],
        "risk_profile": "aggressive",
        "min_confidence": 65.0,
        "max_predictions": 8,
        "min_odds": 2.5,
        "max_odds": 6.0,
        "prediction_types": ["match_result", "over_2_5_goals"]
    }
    
    response = requests.post(f"{base_url}/coupons/multi-league", json=expert_request)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Coupon expert généré:")
        print(f"   - {len(result['predictions'])} prédictions")
        print(f"   - Cote totale: {result['total_odds']}")
        print(f"   - Gain potentiel: {result['expected_return']:.2f}")
        print(f"   - Ligues: {len(result['leagues_included'])}")

if __name__ == "__main__":
    # Attendre que l'API soit prête
    print("Attente du demarrage de l'API...")
    time.sleep(2)
    
    # Lancer les tests
    tester = CouponAPITester()
    tester.run_all_tests()
    
    # Tests spécifiques
    test_specific_scenarios()
    
    print("\n🏁 TESTS TERMINÉS")