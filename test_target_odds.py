"""
TEST ENDPOINT COUPON COTE CIBLE
"""

from advanced_coupon_api import app, TargetOddsRequest
from fastapi.testclient import TestClient

def test_target_odds_coupon():
    """Test génération coupons avec cote cible"""
    print("=== TEST COUPON COTE CIBLE ===")
    
    client = TestClient(app)
    
    # Test 1: Cote faible sécurisée
    print("\n1. Test Cote Faible (2.5x)")
    payload = {
        "target_odds": 2.5,
        "leagues": ["premier_league", "la_liga"],
        "tolerance": 0.15,  # ±15%
        "prioritize_success": True,
        "min_confidence": 80.0,
        "max_predictions": 4
    }
    
    response = client.post("/coupons/target-odds", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Coupon ID: {data['coupon_id']}")
        print(f"   Prédictions: {len(data['predictions'])}")
        print(f"   Cote obtenue: {data['total_odds']}")
        print(f"   Cote cible: {data['target_odds']}")
        print(f"   Réussite cible: {data['target_achievement']}%")
        print(f"   Déviation: {data['odds_deviation']}%")
        print(f"   Confiance moyenne: {data['confidence_average']}%")
        print(f"   Probabilité gain: {data['estimated_win_probability']}%")
    
    # Test 2: Cote moyenne équilibrée
    print("\n2. Test Cote Moyenne (5.0x)")
    payload = {
        "target_odds": 5.0,
        "leagues": ["premier_league", "bundesliga", "serie_a"],
        "tolerance": 0.25,  # ±25%
        "prioritize_success": True,
        "min_confidence": 70.0,
        "max_predictions": 6
    }
    
    response = client.post("/coupons/target-odds", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Cote: {data['total_odds']} (cible: {data['target_odds']})")
        print(f"   Réussite: {data['target_achievement']}%")
        print(f"   Prédictions: {len(data['predictions'])}")
        print(f"   Ligues: {len(set(p['league'] for p in data['predictions']))}")
    
    # Test 3: Cote élevée risquée - priorité cote exacte
    print("\n3. Test Cote Élevée (10.0x) - Priorité Cote")
    payload = {
        "target_odds": 10.0,
        "tolerance": 0.3,  # ±30%
        "prioritize_success": False,  # Priorité cote vs réussite
        "min_confidence": 60.0,
        "max_predictions": 8
    }
    
    response = client.post("/coupons/target-odds", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Cote: {data['total_odds']} (cible: {data['target_odds']})")
        print(f"   Réussite: {data['target_achievement']}%")
        print(f"   Mode: {'Priorité Réussite' if data['prioritized_success'] else 'Priorité Cote'}")
        print(f"   Confiance: {data['confidence_average']}%")
        print(f"   Risque: {data['risk_score']}/100")
    
    # Test 4: Cote très élevée (challenge)
    print("\n4. Test Cote Très Élevée (25.0x)")
    payload = {
        "target_odds": 25.0,
        "tolerance": 0.4,  # ±40%
        "prioritize_success": True,
        "min_confidence": 55.0,
        "max_predictions": 10
    }
    
    response = client.post("/coupons/target-odds", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Cote: {data['total_odds']} (cible: {data['target_odds']})")
        print(f"   Déviation: {data['odds_deviation']}%")
        print(f"   Tolérance: ±{data['tolerance_used']}%")
        print(f"   Prédictions: {len(data['predictions'])}")
        
        # Détail des prédictions
        print("   Détail prédictions:")
        for i, pred in enumerate(data['predictions'][:3]):  # Montrer 3 premières
            print(f"     {i+1}. {pred['home_team']} vs {pred['away_team']} ({pred['league']})")
            print(f"        {pred['prediction_type']}: confiance {pred['confidence']}%, cote {pred['odds']}")
    
    # Test 5: Tolérance très stricte
    print("\n5. Test Tolérance Stricte (3.0x ±5%)")
    payload = {
        "target_odds": 3.0,
        "tolerance": 0.05,  # ±5% seulement
        "prioritize_success": True,
        "min_confidence": 75.0,
        "max_predictions": 5
    }
    
    response = client.post("/coupons/target-odds", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        achieved = data['total_odds']
        target = data['target_odds']
        within_tolerance = abs(achieved - target) / target <= 0.05
        print(f"   Cote: {achieved} (cible: {target})")
        print(f"   Dans tolérance: {'OUI' if within_tolerance else 'NON'}")
        print(f"   Réussite: {data['target_achievement']}%")
    
    print("\n=== ANALYSE PERFORMANCES ===")
    print("Algorithme optimise selon 2 modes:")
    print("- prioritize_success=True: 70% confiance + 30% proximité cote")  
    print("- prioritize_success=False: 30% confiance + 70% proximité cote")
    print("L'algorithme teste toutes combinaisons possibles dans la tolérance")

if __name__ == "__main__":
    test_target_odds_coupon()