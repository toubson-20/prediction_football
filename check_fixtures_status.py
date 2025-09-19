#!/usr/bin/env python3
"""
Vérifier le statut des fixtures API et chercher sur une période plus large
"""

import requests
import json
from datetime import datetime, timedelta
from config import Config

def check_fixtures_status():
    api_key = Config.FOOTBALL_API_KEY
    base_url = "https://v3.football.api-sports.io"
    headers = {'x-apisports-key': api_key}

    print("VERIFICATION STATUT FIXTURES API")
    print("=" * 50)

    # Test Premier League sur une période plus large
    today = datetime.now()
    start_date = today - timedelta(days=3)
    end_date = today + timedelta(days=14)

    print(f"Période de test: {start_date.strftime('%Y-%m-%d')} à {end_date.strftime('%Y-%m-%d')}")
    print()

    # Test Premier League
    print("TEST Premier League (ID: 39):")
    try:
        params = {
            'league': 39,
            'season': 2025,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d')
        }

        response = requests.get(f"{base_url}/fixtures", headers=headers, params=params)
        print(f"  Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"  API Calls restants: {response.headers.get('x-ratelimit-requests-remaining', 'N/A')}")
            print(f"  Résultats: {data.get('results', 0)}")

            fixtures = data.get('response', [])
            if fixtures:
                print(f"  {len(fixtures)} matchs trouvés:")
                for fixture in fixtures[:5]:
                    date = fixture['fixture']['date']
                    home = fixture['teams']['home']['name']
                    away = fixture['teams']['away']['name']
                    status = fixture['fixture']['status']['short']
                    print(f"    {date[:10]} - {home} vs {away} ({status})")
            else:
                print("  Aucun match dans cette période")
        else:
            print(f"  Erreur: {response.text}")

    except Exception as e:
        print(f"  Exception: {e}")

    print()

    # Test avec saison 2024 pour comparaison
    print("TEST Premier League saison 2024:")
    try:
        params = {
            'league': 39,
            'season': 2024,
            'from': '2025-01-01',
            'to': '2025-09-17'
        }

        response = requests.get(f"{base_url}/fixtures", headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            fixtures = data.get('response', [])
            print(f"  Saison 2024: {len(fixtures)} matchs trouvés")
        else:
            print(f"  Erreur saison 2024: {response.status_code}")

    except Exception as e:
        print(f"  Exception saison 2024: {e}")

    print()

    # Test status API général
    print("TEST STATUS API:")
    try:
        response = requests.get(f"{base_url}/status", headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"  API Status: {data}")
        else:
            print(f"  Erreur status: {response.status_code}")

    except Exception as e:
        print(f"  Exception status: {e}")

if __name__ == "__main__":
    check_fixtures_status()