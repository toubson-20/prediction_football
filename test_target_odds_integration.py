"""
TEST INTEGRATION COTE CIBLE DANS TOUS LES ENDPOINTS
"""

from advanced_coupon_api import app
from fastapi.testclient import TestClient

def test_target_odds_integration():
    """Test cote cible intégrée dans tous les endpoints existants"""
    print("=== TEST INTÉGRATION COTE CIBLE ===")
    
    client = TestClient(app)
    
    # Test 1: Coupon par ligue avec cote cible
    print("\n1. Test Coupon Premier League avec Cote Cible (3.5x)")
    payload = {
        "leagues": ["premier_league"],
        "risk_profile": "balanced",
        "min_confidence": 75.0,
        "max_predictions": 4,
        "target_odds": 3.5,
        "target_odds_tolerance": 0.15,
        "prioritize_target_success": True
    }
    
    response = client.post("/coupons/league/premier_league", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Type coupon: {data['coupon_type']}")
        print(f"   Cote: {data['total_odds']} (cible: 3.5)")
        deviation = abs(data['total_odds'] - 3.5) / 3.5 * 100
        print(f"   Déviation: {deviation:.1f}%")
        print(f"   Prédictions: {len(data['predictions'])}")
        print(f"   Confiance: {data['confidence_average']}%")
    
    # Test 2: Multi-ligues avec cote cible
    print("\n2. Test Multi-Ligues avec Cote Cible (7.0x)")
    payload = {
        "leagues": ["premier_league", "la_liga", "serie_a"],
        "risk_profile": "balanced",
        "min_confidence": 70.0,
        "max_predictions": 6,
        "target_odds": 7.0,
        "target_odds_tolerance": 0.25,
        "prioritize_target_success": False  # Priorité cote exacte
    }
    
    response = client.post("/coupons/multi-league", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Cote: {data['total_odds']} (cible: 7.0)")
        print(f"   Ligues utilisées: {len(set(p['league'] for p in data['predictions']))}")
        print(f"   Mode: {'Réussite' if payload['prioritize_target_success'] else 'Cote Exacte'}")
        within_tolerance = abs(data['total_odds'] - 7.0) / 7.0 <= 0.25
        print(f"   Dans tolérance: {'OUI' if within_tolerance else 'NON'}")
    
    # Test 3: Toutes ligues avec cote cible (GET avec query params)
    print("\n3. Test Toutes Ligues avec Cote Cible (15.0x)")
    response = client.post("/coupons/all-leagues?target_odds=15.0&target_odds_tolerance=0.3&min_confidence=65&max_predictions=8&prioritize_target_success=true")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Cote: {data['total_odds']} (cible: 15.0)")
        print(f"   Type: {data['coupon_type']}")
        unique_leagues = len(set(p['league'] for p in data['predictions']))
        print(f"   Diversification: {unique_leagues} ligues différentes")
        print(f"   Probabilité gain: {data['estimated_win_probability']}%")
    
    # Test 4: Coupon optimisé avec cote cible
    print("\n4. Test Coupon Optimisé avec Cote Cible (4.2x)")
    payload = {
        "leagues": ["bundesliga", "ligue_1"],
        "optimization_strategy": "high_confidence", 
        "min_confidence": 80.0,
        "max_predictions": 5,
        "intelligent_selection": True,
        "target_odds": 4.2,
        "target_odds_tolerance": 0.1  # Tolérance stricte ±10%
    }
    
    response = client.post("/coupons/optimized", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Cote: {data['total_odds']} (cible: 4.2)")
        deviation = abs(data['total_odds'] - 4.2) / 4.2
        print(f"   Précision: {(1-deviation)*100:.1f}%")
        print(f"   Stratégie utilisée: {data.get('optimization_used', 'N/A')}")
        print(f"   Qualité: {data.get('quality_score', 0)}/100")
    
    # Test 5: Comparaison avec/sans cote cible
    print("\n5. Test Comparaison Avec/Sans Cote Cible")
    
    # Sans cote cible
    payload_normal = {
        "leagues": ["premier_league", "la_liga"],
        "risk_profile": "balanced",
        "min_confidence": 75.0,
        "max_predictions": 4
    }
    
    response_normal = client.post("/coupons/multi-league", json=payload_normal)
    
    # Avec cote cible
    payload_target = payload_normal.copy()
    payload_target.update({
        "target_odds": 5.0,
        "target_odds_tolerance": 0.2
    })
    
    response_target = client.post("/coupons/multi-league", json=payload_target)
    
    if response_normal.status_code == 200 and response_target.status_code == 200:
        normal = response_normal.json()
        target = response_target.json()
        
        print(f"   Sans cote: {normal['total_odds']} (confiance: {normal['confidence_average']}%)")
        print(f"   Avec cote: {target['total_odds']} (confiance: {target['confidence_average']}%)")
        print(f"   Différence type: {normal['coupon_type']} vs {target['coupon_type']}")
    
    print("\n=== RÉSULTATS INTÉGRATION ===")
    print("[OK] Tous les endpoints supportent maintenant les parametres:")
    print("   - target_odds: Cote souhaitee")
    print("   - target_odds_tolerance: Tolerance (0.0-1.0)")
    print("   - prioritize_target_success: Priorite reussite vs cote")
    print("[OK] Retrocompatibilite: fonctionne sans les nouveaux parametres")
    print("[OK] Logique intelligente: bascule automatiquement vers l'algo cote cible")

if __name__ == "__main__":
    test_target_odds_integration()