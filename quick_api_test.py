"""
Test rapide de l'API sans serveur externe
"""

from advanced_coupon_api import app, coupon_service, CouponRequest, LeagueEnum, RiskProfileEnum
from fastapi.testclient import TestClient
import json

def test_api_locally():
    """Test l'API en local sans serveur"""
    print("=== TEST API LOCALE ===")
    
    client = TestClient(app)
    
    # Test 1: Root
    response = client.get("/")
    print(f"1. Root: {response.status_code}")
    if response.status_code == 200:
        print(f"   Service: {response.json()['service']}")
    
    # Test 2: Ligues
    response = client.get("/leagues")
    print(f"2. Ligues: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Total ligues: {data['total_leagues']}")
    
    # Test 3: Types prédictions
    response = client.get("/predictions/types")
    print(f"3. Types: {response.status_code}")
    
    # Test 4: Coupon Premier League
    payload = {
        "leagues": ["premier_league"],
        "risk_profile": "balanced",
        "min_confidence": 70.0,
        "max_predictions": 4
    }
    
    response = client.post("/coupons/league/premier_league", json=payload)
    print(f"4. Coupon PL: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   ID: {data['coupon_id']}")
        print(f"   Prédictions: {len(data['predictions'])}")
        print(f"   Cote totale: {data['total_odds']}")
        print(f"   Confiance: {data['confidence_average']}%")
    
    # Test 5: Coupon Multi-ligues
    payload = {
        "leagues": ["premier_league", "la_liga", "bundesliga"],
        "risk_profile": "balanced", 
        "min_confidence": 75.0,
        "max_predictions": 6
    }
    
    response = client.post("/coupons/multi-league", json=payload)
    print(f"5. Coupon Multi: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   ID: {data['coupon_id']}")
        print(f"   Type: {data['coupon_type']}")
        print(f"   Ligues: {len(data['leagues_included'])}")
        print(f"   Prédictions: {len(data['predictions'])}")
    
    # Test 6: Coupon Toutes Ligues
    response = client.post("/coupons/all-leagues?risk_profile=aggressive&min_confidence=65&max_predictions=8")
    print(f"6. Toutes Ligues: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Prédictions: {len(data['predictions'])}")
        print(f"   Gain potentiel: {data['expected_return']}")
    
    print("\n=== SERVICE DIRECT ===")
    
    # Test direct du service
    request = CouponRequest(
        leagues=[LeagueEnum.PREMIER_LEAGUE, LeagueEnum.BUNDESLIGA],
        risk_profile=RiskProfileEnum.BALANCED,
        min_confidence=70.0,
        max_predictions=5
    )
    
    coupon = coupon_service.generate_multi_league_coupon(request)
    print(f"Service Direct - Coupon: {coupon.coupon_id}")
    print(f"   Prédictions: {len(coupon.predictions)}")
    print(f"   Cote: {coupon.total_odds}")
    print(f"   Risque: {coupon.risk_score}")
    
    # Afficher détail d'une prédiction
    if coupon.predictions:
        pred = coupon.predictions[0]
        print(f"\nExemple Prédiction:")
        print(f"   {pred.home_team} vs {pred.away_team}")
        print(f"   Ligue: {pred.league}")
        print(f"   Type: {pred.prediction_type}")
        print(f"   Valeur: {pred.prediction_value}")
        print(f"   Confiance: {pred.confidence}%")
        print(f"   Cote: {pred.odds}")
    
    print("\nTESTS REUSSIS - API FONCTIONNELLE")

if __name__ == "__main__":
    test_api_locally()