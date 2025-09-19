#!/usr/bin/env python3
"""
Test simple du système de prédictions ML
Utilise les nouveaux modèles entraînés
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

def test_prediction_system():
    print("=" * 60)
    print("TEST SYSTEME DE PREDICTIONS ML")
    print("=" * 60)

    models_dir = Path("models/complete_models")

    # Test Premier League
    model_file = models_dir / "complete_39_next_match_result.joblib"
    scaler_file = models_dir / "complete_scaler_39_next_match_result.joblib"

    if not model_file.exists():
        print("ERREUR: Modèle Premier League non trouvé")
        return False

    if not scaler_file.exists():
        print("ERREUR: Scaler Premier League non trouvé")
        return False

    # Charger modèle et scaler
    print("Chargement modèle Premier League...")
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    print(f"Modèle: {type(model).__name__}")
    print(f"Scaler: {type(scaler).__name__}")

    # Créer données factices pour test (53 features)
    print("Création données de test...")

    # Données d'équipe factices (moyennes typiques Premier League)
    test_data = {
        'matches_played': 20,
        'win_rate': 0.45,
        'avg_goals_for': 1.2,
        'avg_goals_against': 1.1,
        'wins': 9,
        'draws': 6,
        'losses': 5,
        'points': 33,
        'goal_difference': 2,
        'total_goals_scored': 24,
        'avg_shots_per_match': 12.5,
        'shot_accuracy': 0.35,
        'avg_corners_per_match': 5.2,
        'total_goals_conceded': 22,
        'clean_sheets': 4,
        'avg_possession': 52.3,
        'pass_accuracy': 0.82,
        'avg_fouls_per_match': 10.8,
        'avg_cards_per_match': 2.1
    }

    # Créer array de 53 features (compléter avec des valeurs moyennes)
    feature_array = np.zeros(53)

    # Remplir avec nos données de test
    values = list(test_data.values())
    feature_array[:len(values)] = values

    # Compléter le reste avec des valeurs moyennes
    for i in range(len(values), 53):
        feature_array[i] = np.random.normal(0.5, 0.2)  # Valeurs centrées

    # Assurer que les valeurs sont dans des ranges raisonnables
    feature_array = np.clip(feature_array, 0, 10)

    print(f"Features préparées: {feature_array.shape}")

    # Prediction
    print("Exécution prédiction...")

    # Scaler les données
    feature_scaled = scaler.transform([feature_array])

    # Prédiction
    prediction = model.predict(feature_scaled)[0]

    print(f"RESULTAT PREDICTION:")
    print(f"Win probability: {prediction:.3f} ({prediction*100:.1f}%)")

    # Test autres modèles
    print(f"\nTest autres modèles disponibles:")

    available_models = [
        ('goals_scored', 'Buts marqués'),
        ('both_teams_score', 'Both teams score'),
        ('over_2_5_goals', 'Plus de 2.5 buts')
    ]

    for model_type, description in available_models:
        model_path = models_dir / f"complete_39_{model_type}.joblib"
        scaler_path = models_dir / f"complete_scaler_39_{model_type}.joblib"

        if model_path.exists() and scaler_path.exists():
            test_model = joblib.load(model_path)
            test_scaler = joblib.load(scaler_path)

            # Même données, prédiction différente
            scaled_data = test_scaler.transform([feature_array])
            pred = test_model.predict(scaled_data)[0]

            print(f"  {description}: {pred:.3f}")
        else:
            print(f"  {description}: Modèle non disponible")

    print(f"\nTEST REUSSI!")
    print(f"Le système de prédictions est opérationnel.")

    return True

if __name__ == "__main__":
    test_prediction_system()