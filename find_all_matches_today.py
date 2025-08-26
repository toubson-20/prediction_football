"""
RECHERCHE TOUS MATCHS AUJOURD'HUI
Trouve tous les matchs du jour dans nos competitions suivies
"""

import requests
from datetime import datetime, timedelta
import sys

sys.path.append('src')
from config import API_FOOTBALL_CONFIG

def find_all_matches_today():
    """Chercher tous les matchs aujourd'hui dans nos ligues"""
    
    print("RECHERCHE MATCHS AUJOURD'HUI")
    print("="*60)
    
    # Configuration
    api_config = API_FOOTBALL_CONFIG
    base_url = "https://v3.football.api-sports.io"
    headers = {"X-RapidAPI-Key": api_config["api_key"]}
    
    # Nos ligues suivies
    leagues = {
        39: "Premier League",
        140: "La Liga", 
        61: "Ligue 1",
        78: "Bundesliga",
        135: "Serie A",
        2: "Champions League",
        3: "Europa League"
    }
    
    # Periode de recherche
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    
    print(f"Date recherchee: {today_str}")
    print()
    
    all_matches = []
    
    for league_id, league_name in leagues.items():
        print(f"--- {league_name} ---")
        
        try:
            response = requests.get(
                f"{base_url}/fixtures",
                headers=headers,
                params={
                    "league": league_id,
                    "season": 2025,
                    "date": today_str
                }
            )
            
            if response.status_code == 200:
                fixtures_data = response.json()
                
                if fixtures_data.get('response'):
                    matches = fixtures_data['response']
                    
                    if matches:
                        print(f"  {len(matches)} match(s) trouve(s)")
                        
                        for match in matches:
                            fixture = match['fixture']
                            home_team = match['teams']['home']
                            away_team = match['teams']['away']
                            
                            match_time = datetime.fromisoformat(fixture['date'].replace('Z', '+00:00'))
                            local_time = match_time.strftime('%H:%M')
                            
                            match_info = {
                                'league': league_name,
                                'league_id': league_id,
                                'fixture_id': fixture['id'],
                                'time': local_time,
                                'home_team': home_team['name'],
                                'away_team': away_team['name'],
                                'home_id': home_team['id'],
                                'away_id': away_team['id'],
                                'status': fixture['status']['long'],
                                'venue': fixture['venue']['name'] if fixture['venue'] else 'Unknown'
                            }
                            
                            all_matches.append(match_info)
                            
                            print(f"  {local_time} - {home_team['name']} vs {away_team['name']}")
                            print(f"           Stade: {match_info['venue']}")
                            print(f"           Statut: {match_info['status']}")
                            print()
                    else:
                        print("  Aucun match")
                else:
                    print("  Aucune donnee retournee")
            else:
                print(f"  ERREUR API: {response.status_code}")
                
        except Exception as e:
            print(f"  ERREUR: {e}")
        
        print()
    
    # Resume general
    print("="*60)
    print("RESUME MATCHS AUJOURD'HUI")
    print("="*60)
    
    if all_matches:
        print(f"TOTAL: {len(all_matches)} match(s) trouve(s)")
        print()
        
        # Grouper par ligue
        by_league = {}
        for match in all_matches:
            league = match['league']
            if league not in by_league:
                by_league[league] = []
            by_league[league].append(match)
        
        # Affichage par ligue
        for league, matches in by_league.items():
            print(f"{league} ({len(matches)} match(s)):")
            for match in sorted(matches, key=lambda x: x['time']):
                print(f"  {match['time']} - {match['home_team']} vs {match['away_team']}")
            print()
        
        print("MATCHS INTERESSANTS POUR PREDICTIONS:")
        print("-" * 40)
        
        # Mettre en evidence certains matchs
        big_teams = [
            'Liverpool', 'Manchester City', 'Arsenal', 'Chelsea', 'Tottenham', 'Manchester United',
            'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Valencia',
            'Paris Saint Germain', 'Monaco', 'Lyon', 'Marseille',
            'Bayern Munich', 'Borussia Dortmund', 'RB Leipzig',
            'Juventus', 'AC Milan', 'Inter', 'Napoli', 'AS Roma'
        ]
        
        interesting_matches = []
        for match in all_matches:
            if (any(team in match['home_team'] for team in big_teams) or 
                any(team in match['away_team'] for team in big_teams)):
                interesting_matches.append(match)
        
        if interesting_matches:
            for match in sorted(interesting_matches, key=lambda x: x['time']):
                print(f"  {match['time']} - {match['home_team']} vs {match['away_team']} ({match['league']})")
        else:
            print("  Aucun match avec grandes equipes aujourd'hui")
        
    else:
        print("AUCUN MATCH TROUVE AUJOURD'HUI")
        print()
        print("Verification matchs des prochains jours...")
        
        # Chercher matchs des 3 prochains jours
        for day_offset in range(1, 4):
            future_date = today + timedelta(days=day_offset)
            future_date_str = future_date.strftime("%Y-%m-%d")
            
            print(f"\n{future_date_str}:")
            
            found_future = False
            for league_id, league_name in list(leagues.items())[:3]:  # Juste 3 ligues pour test
                try:
                    response = requests.get(
                        f"{base_url}/fixtures",
                        headers=headers,
                        params={
                            "league": league_id,
                            "season": 2025,
                            "date": future_date_str
                        }
                    )
                    
                    if response.status_code == 200:
                        fixtures_data = response.json()
                        if fixtures_data.get('response') and fixtures_data['response']:
                            found_future = True
                            print(f"  {league_name}: {len(fixtures_data['response'])} match(s)")
                            break
                except:
                    continue
            
            if not found_future:
                print("  Aucun match trouve")
    
    return all_matches

def main():
    try:
        matches = find_all_matches_today()
        print(f"\nRECHERCHE TERMINEE!")
        return matches
        
    except Exception as e:
        print(f"ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    main()