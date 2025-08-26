# 🏈 Football ML Prediction System

Système de Machine Learning pour prédictions football avec collecte automatique de données et re-entraînement intelligent.

## ✨ Fonctionnalités

- **🔄 Collecte automatique** des données via API-Football
- **🤖 Re-entraînement intelligent** des modèles ML
- **📊 Prédictions haute précision** (R² > 0.88)
- **⏰ Maintenance automatique** quotidienne
- **🎯 Support multi-ligues** (Premier League, La Liga, Serie A, etc.)

## 🚀 Installation

1. **Cloner le repository**
```bash
git clone <repository-url>
cd sport
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

3. **Configuration API**
   - Configurer `config.py` avec votre clé API-Football
   - Voir `config.py` pour les détails

## 🎯 Utilisation

### Système Automatique
```bash
# Configuration du planificateur (une fois)
python setup_auto_retrain_scheduler.py

# Vérification du statut
python check_scheduler_status.py
```

### Exécution Manuelle
```bash
# Mise à jour complète
python final_auto_retrain_system.py

# Prédictions
python predict_liverpool_today.py
python find_all_matches_today.py

# Analyse des données
python analyze_liverpool_training_data.py
```

## 📊 Architecture

### Scripts Principaux
- `final_auto_retrain_system.py` - Système principal de re-entraînement
- `complete_data_collector.py` - Collecteur de données API
- `integrate_complete_datasets.py` - Intégrateur datasets ML
- `adapt_ml_models_complete.py` - Entraîneur de modèles

### Structure des Données
```
data/
├── complete_collection/     # Données brutes API
├── ultra_processed/        # Datasets ML optimisés
└── raw/                   # Données historiques
```

### Modèles ML
```
models/complete_models/     # 53 modèles entraînés
├── complete_39_*.joblib   # Premier League
├── complete_140_*.joblib  # La Liga  
└── complete_135_*.joblib  # Serie A
```

## 🔧 Configuration

### Planificateur Windows
- **Fréquence**: Quotidienne à 08:00
- **Tâche**: `AutoRetrainML`
- **Action**: Vérification et mise à jour automatique

### Ligues Supportées
- ⚽ Premier League (39)
- 🇪🇸 La Liga (140) 
- 🇮🇹 Serie A (135)
- 🇫🇷 Ligue 1 (61)
- 🇩🇪 Bundesliga (78)

## 📈 Performance

- **R² Score**: 0.884 (Premier League)
- **Features**: 81 colonnes optimisées
- **Équipes**: 202 équipes analysées
- **Joueurs**: 3,544 profils complets

## 🛠️ Développement

### Tests
```bash
python check_scheduler_status.py  # Vérifier système
```

### Logs
```bash
logs/                          # Logs d'activité
├── critical_data_collection.log
└── immediate_retraining.log
```

## 📝 Notes

- Système optimisé après nettoyage (56 fichiers supprimés)
- Architecture finale avec 12 fichiers essentiels
- Backup automatique avant modifications
- Compatible Windows avec tâches planifiées

## 🔒 Sécurité

- Clés API protégées via `.gitignore`
- Backup automatique avant modifications
- Gestion d'erreurs robuste
- Logs complets pour debugging

---

**🎯 Système ready for production - Maintenance zéro intervention**