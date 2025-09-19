#!/usr/bin/env python3
"""
Recherche les prochains matchs dans les 7 prochains jours
"""

import requests
import json
from datetime import datetime, timedelta
from config import Config

def find_upcoming_matches():
    api_key = Config.FOOTBALL_API_KEY
    base_url = "https://v3.football.api-sports.io"
    headers = {'x-apisports-key': api_key}

    competitions = {
        'Premier League': 39,
        'La Liga': 140,
        'Ligue 1': 61,
        'Bundesliga': 78,
        'Serie A': 135,
        'Champions League': 2,
        'Europa League': 3
    }

    today = datetime.now()
    end_date = today + timedelta(days=7)

    print("RECHERCHE PROCHAINS MATCHS (7 prochains jours)")
    print("=" * 60)
    print(f"Période: {today.strftime('%Y-%m-%d')} à {end_date.strftime('%Y-%m-%d')}")
    print()

    all_matches = []

    for comp_name, league_id in competitions.items():
        print(f"--- {comp_name} ---")

        try:
            params = {
                'league': league_id,
                'season': 2025,
                'from': today.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d')
            }

            response = requests.get(f"{base_url}/fixtures", headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()
                matches = data.get('response', [])

                if matches:
                    print(f"  {len(matches)} match(s) trouvé(s)")
                    for match in matches[:5]:  # Afficher les 5 premiers
                        date = match['fixture']['date']
                        home = match['teams']['home']['name']
                        away = match['teams']['away']['name']
                        print(f"    {date[:10]} {date[11:16]} - {home} vs {away}")

                    all_matches.extend(matches)
                else:
                    print("  Aucun match trouvé")
            else:
                print(f"  Erreur API: {response.status_code}")

        except Exception as e:
            print(f"  Erreur: {e}")

        print()

    print("=" * 60)
    print(f"TOTAL: {len(all_matches)} matchs trouvés dans les 7 prochains jours")

    if all_matches:
        print("\nPROCHAINS MATCHS PAR DATE:")
        matches_by_date = {}
        for match in all_matches:
            date = match['fixture']['date'][:10]
            if date not in matches_by_date:
                matches_by_date[date] = []
            matches_by_date[date].append(match)

        for date in sorted(matches_by_date.keys()):
            print(f"\n{date} ({len(matches_by_date[date])} matchs):")
            for match in matches_by_date[date][:3]:  # Top 3 par jour
                home = match['teams']['home']['name']
                away = match['teams']['away']['name']
                time = match['fixture']['date'][11:16]
                comp = match['league']['name']
                print(f"  {time} - {home} vs {away} ({comp})")

if __name__ == "__main__":
    find_upcoming_matches()