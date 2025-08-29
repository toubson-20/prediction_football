"""
TEST ENDPOINT PRÉDICTION MATCH SPÉCIFIQUE
"""

from advanced_coupon_api import app, MatchPredictionRequest
from fastapi.testclient import TestClient

def test_match_prediction():
    """Test prédiction pour un match spécifique"""
    print("=== TEST PRÉDICTION MATCH SPÉCIFIQUE ===")
    
    client = TestClient(app)
    
    # Test 1: Match Premier League
    print("\n1. Test Match Premier League")
    payload = {
        "home_team": "Manchester United",
        "away_team": "Liverpool", 
        "league": "premier_league",
        "prediction_types": ["match_result", "both_teams_score"]
    }
    
    response = client.post("/predictions/match", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Match: {data['match']}")
        print(f"   Ligue: {data['league']}")
        print(f"   Prédictions: {data['total_predictions']}")
        print(f"   Confiance moyenne: {data['average_confidence']}%")
        print(f"   Cote combinée: {data['combined_odds']}")
        
        # Détail prédictions
        for pred in data['predictions']:
            print(f"     - {pred['prediction_type']}: {pred['prediction_value']} (confiance: {pred['confidence']}%)")
    
    # Test 2: Match Bundesliga avec tous les types
    print("\n2. Test Match Bundesliga complet")
    payload = {
        "home_team": "Bayern Munich",
        "away_team": "Borussia Dortmund",
        "league": "bundesliga"
        # prediction_types non spécifié = tous les types
    }
    
    response = client.post("/predictions/match", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Match: {data['match']}")
        print(f"   Prédictions totales: {data['total_predictions']}")
        print(f"   Confiance: {data['average_confidence']}%")
        print(f"   Cote: {data['combined_odds']}")
    
    # Test 3: Match La Liga spécifique
    print("\n3. Test Match La Liga")
    payload = {
        "home_team": "Barcelona", 
        "away_team": "Real Madrid",
        "league": "la_liga",
        "prediction_types": ["match_result", "over_2_5_goals"]
    }
    
    response = client.post("/predictions/match", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Clasico: {data['match']}")
        print(f"   Prédictions: {data['total_predictions']}")
        
        for pred in data['predictions']:
            print(f"     {pred['prediction_type']}: confiance {pred['confidence']}%, cote {pred['odds']}")
    
    # Test 4: Équipe introuvable
    print("\n4. Test Équipe Inexistante")
    payload = {
        "home_team": "Equipe Inexistante",
        "away_team": "Autre Equipe",
        "league": "premier_league"
    }
    
    response = client.post("/predictions/match", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 404:
        print("   Erreur 404 attendue - équipe inexistante")
    
    print("\n=== TESTS TERMINÉS ===")

if __name__ == "__main__":
    test_match_prediction()