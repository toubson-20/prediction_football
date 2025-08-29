"""
TEST DE L'API AVEC OPTIMISATION INTELLIGENTE
Teste les nouvelles fonctionnalités d'optimisation stratégique
"""

from advanced_coupon_api import app, coupon_service, CouponRequest, LeagueEnum, RiskProfileEnum, OptimizationStrategyEnum
from fastapi.testclient import TestClient
import json

def test_intelligent_optimization():
    """Test complet de l'optimisation intelligente"""
    print("=== TEST OPTIMISATION INTELLIGENTE ===")
    
    client = TestClient(app)
    
    # Test 1: Stratégies disponibles
    response = client.get("/optimization/strategies")
    print(f"1. Stratégies: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        strategies = data['optimization_strategies']
        print(f"   Stratégies disponibles: {len(strategies)}")
        for strategy in strategies[:3]:
            print(f"   - {strategy['name']}: {strategy['description']}")
    
    # Test 2: Coupon Optimisé Équilibré
    print(f"\n2. Test Coupon Équilibré")
    payload = {
        "leagues": ["premier_league", "la_liga"],
        "risk_profile": "balanced",
        "optimization_strategy": "balanced",
        "min_confidence": 75.0,
        "max_predictions": 6,
        "intelligent_selection": True
    }
    
    response = client.post("/coupons/optimized", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   ID: {data['coupon_id']}")
        print(f"   Prédictions: {len(data['predictions'])}")
        print(f"   Optimisation: {data.get('optimization_used', 'N/A')}")
        print(f"   Score qualité: {data.get('quality_score', 0)}/100")
        print(f"   Kelly weight: {data.get('kelly_weight', 0)}")
        print(f"   Diversification: {data.get('diversification_score', 0)}%")
    
    # Test 3: Stratégie High Confidence
    print(f"\n3. Test Stratégie Sécurisée")
    payload = {
        "leagues": ["premier_league", "bundesliga", "serie_a"],
        "risk_profile": "conservative",
        "optimization_strategy": "high_confidence",
        "min_confidence": 85.0,
        "max_predictions": 4,
        "intelligent_selection": True
    }
    
    response = client.post("/coupons/optimized", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Prédictions: {len(data['predictions'])}")
        print(f"   Confiance moyenne: {data['confidence_average']}%")
        print(f"   Probabilité gain: {data['estimated_win_probability']}%")
        print(f"   Mise recommandée: {data.get('recommended_stake', 0)}")
    
    # Test 4: Value Hunting Agressif
    print(f"\n4. Test Chasse aux Valeurs")
    payload = {
        "leagues": ["la_liga", "ligue_1", "bundesliga"],
        "risk_profile": "aggressive", 
        "optimization_strategy": "value_hunting",
        "min_confidence": 65.0,
        "max_predictions": 8,
        "min_odds": 2.0,
        "max_odds": 5.0
    }
    
    response = client.post("/coupons/optimized", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Prédictions: {len(data['predictions'])}")
        print(f"   Cote totale: {data['total_odds']}")
        print(f"   Gain potentiel: {data['expected_return']}")
        print(f"   Niveau risque: {data.get('risk_level', 'N/A')}")
    
    # Test 5: Auto-Optimal (IA choisit)
    print(f"\n5. Test IA Auto-Optimal")
    payload = {
        "leagues": ["premier_league", "la_liga", "bundesliga", "serie_a"],
        "risk_profile": "balanced",
        "optimization_strategy": "auto_optimal",
        "max_predictions": 10
    }
    
    response = client.post("/coupons/optimized", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        strategy_used = data.get('optimization_used', 'unknown').replace('_intelligent', '')
        print(f"   IA a choisi: {strategy_used}")
        print(f"   Prédictions: {len(data['predictions'])}")
        print(f"   Ligues utilisées: {len(data['leagues_included'])}")
        print(f"   Score qualité: {data.get('quality_score', 0)}/100")
    
    # Test 6: Coupon toutes ligues optimisé
    print(f"\n6. Test Toutes Ligues Optimisé")
    response = client.post("/coupons/all-leagues?optimization_strategy=anti_correlation&max_predictions=12&intelligent_selection=true")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Prédictions: {len(data['predictions'])}")
        print(f"   Diversification: {data.get('diversification_score', 0)}%")
        unique_leagues = len(set(pred['league'] for pred in data.get('predictions', [])))
        print(f"   Ligues diversifiées: {unique_leagues}")
    
    print(f"\n=== DÉTAIL EXEMPLE PRÉDICTION ===")
    # Montrer détail d'une prédiction optimisée
    if response.status_code == 200 and data.get('predictions'):
        pred = data['predictions'][0]
        print(f"Match: {pred['home_team']} vs {pred['away_team']}")
        print(f"Ligue: {pred['league']}")
        print(f"Type: {pred['prediction_type']}")
        print(f"Valeur: {pred['prediction_value']}")
        print(f"Confiance: {pred['confidence']}%")
        print(f"Cote: {pred['odds']}")
        print(f"Valeur attendue: {pred['expected_value']}")
        print(f"Niveau risque: {pred['risk_level']}")

def test_optimization_comparison():
    """Comparer différentes stratégies sur les mêmes données"""
    print(f"\n=== COMPARAISON STRATÉGIES ===")
    
    base_request = {
        "leagues": ["premier_league", "la_liga", "bundesliga"],
        "risk_profile": "balanced",
        "max_predictions": 6,
        "intelligent_selection": True
    }
    
    strategies = ["balanced", "high_confidence", "value_hunting", "anti_correlation"]
    client = TestClient(app)
    
    results = {}
    
    for strategy in strategies:
        payload = base_request.copy()
        payload["optimization_strategy"] = strategy
        
        response = client.post("/coupons/optimized", json=payload)
        if response.status_code == 200:
            data = response.json()
            results[strategy] = {
                'predictions': len(data['predictions']),
                'total_odds': data['total_odds'],
                'confidence': data['confidence_average'],
                'win_probability': data['estimated_win_probability'],
                'quality_score': data.get('quality_score', 0),
                'diversification': data.get('diversification_score', 0)
            }
    
    # Afficher comparaison
    print(f"{'Stratégie':<15} {'Prédictions':<12} {'Cote':<8} {'Confiance':<10} {'Qualité':<8} {'Diversif':<8}")
    print("-" * 70)
    
    for strategy, metrics in results.items():
        quality = metrics['quality_score'] or 0
        diversification = metrics['diversification'] or 0
        print(f"{strategy:<15} {metrics['predictions']:<12} {metrics['total_odds']:<8.1f} {metrics['confidence']:<10.1f}% {quality:<8.1f} {diversification:<8.1f}%")
    
    # Identifier meilleure stratégie
    if results:
        best_strategy = max(results.keys(), key=lambda k: results[k]['quality_score'] or 0)
        best_score = results[best_strategy]['quality_score'] or 0
        print(f"\n[TROPHEE] Meilleure stratégie: {best_strategy} (score {best_score:.1f})")

if __name__ == "__main__":
    test_intelligent_optimization()
    test_optimization_comparison()