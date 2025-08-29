# 🎯 API AVANCÉE DE COUPONS FOOTBALL

API complète pour la génération de coupons de paris football, compatible frontend avec support multi-ligues et filtrage avancé.

## 🚀 Démarrage Rapide

### Installation
```bash
pip install fastapi uvicorn pydantic
```

### Lancement
```bash
cd C:\dev\ia\sport
python advanced_coupon_api.py
```

L'API sera disponible sur : **http://localhost:8000**

### Documentation Interactive
- **Swagger UI** : http://localhost:8000/docs  
- **ReDoc** : http://localhost:8000/redoc

## 📋 Endpoints Principaux

### 🏆 1. Coupons par Ligue Spécifique

```http
POST /coupons/league/{league}
```

**Génère un coupon pour une ligue donnée.**

**Paramètres :**
- `league` : `premier_league`, `la_liga`, `bundesliga`, `serie_a`, `ligue_1`, `champions_league`, `europa_league`

**Exemple :**
```json
POST /coupons/league/premier_league
{
  "risk_profile": "balanced",
  "min_confidence": 75.0,
  "max_predictions": 5,
  "min_odds": 1.3,
  "max_odds": 3.0,
  "prediction_types": ["match_result", "both_teams_score"]
}
```

**Réponse :**
```json
{
  "coupon_id": "LC_20250829_210000_premier_league",
  "coupon_type": "single_league",
  "predictions": [
    {
      "match_id": 12345,
      "home_team": "Manchester City",
      "away_team": "Liverpool",
      "league": "premier_league",
      "prediction_type": "match_result",
      "prediction_value": "1",
      "confidence": 78.5,
      "odds": 2.1,
      "expected_value": 0.649,
      "risk_level": "Moyen",
      "match_date": "2025-08-30T15:00:00"
    }
  ],
  "total_odds": 12.45,
  "expected_return": 0.234,
  "risk_score": 45.2,
  "confidence_average": 76.8,
  "leagues_included": ["premier_league"],
  "estimated_win_probability": 65.4
}
```

### 🌍 2. Coupons Multi-Ligues

```http
POST /coupons/multi-league
```

**Génère un coupon combinant plusieurs ligues avec répartition équilibrée.**

**Exemple :**
```json
{
  "leagues": ["premier_league", "la_liga", "bundesliga"],
  "risk_profile": "balanced", 
  "min_confidence": 70.0,
  "max_predictions": 9,
  "min_odds": 1.4,
  "max_odds": 2.8
}
```

### 🌟 3. Coupon Toutes Ligues

```http
POST /coupons/all-leagues?risk_profile=balanced&min_confidence=70&max_predictions=10
```

**Génère automatiquement un coupon optimisé avec toutes les ligues disponibles.**

### ⚙️ 4. Configuration

```http
GET /leagues
```
**Retourne toutes les ligues supportées**

```http
GET /predictions/types  
```
**Retourne tous les types de prédictions disponibles**

```http
GET /health
```
**État de l'API et statistiques**

## 🎛️ Profils de Risque

### Conservative 🛡️
- **Confiance** : ≥ 85%
- **Cotes** : 1.2 - 1.8
- **Usage** : Investisseurs prudents, débutants

### Balanced ⚖️
- **Confiance** : ≥ 70%  
- **Cotes** : 1.4 - 2.5
- **Usage** : Parieur moyen, équilibré

### Aggressive 🚀
- **Confiance** : ≥ 60%
- **Cotes** : 1.8 - 4.0
- **Usage** : Experts, gros gains potentiels

## 📊 Types de Prédictions

| Type | Description | Exemple |
|------|-------------|---------|
| `match_result` | Résultat 1X2 | "1" (Victoire domicile) |
| `both_teams_score` | Les deux marquent | "Oui" / "Non" |  
| `over_2_5_goals` | Plus de 2.5 buts | "Plus de 2.5" |
| `under_2_5_goals` | Moins de 2.5 buts | "Moins de 2.5" |
| `clean_sheet` | Cage inviolée | "Domicile" / "Extérieur" |
| `win_probability` | Probabilité victoire | 0.75 |

## 🔧 Intégration Frontend

### React/Vue.js Exemple

```javascript
// Générer coupon multi-ligues
const generateCoupon = async () => {
  const request = {
    leagues: ['premier_league', 'la_liga'],
    risk_profile: 'balanced',
    min_confidence: 75,
    max_predictions: 6,
    min_odds: 1.5,
    max_odds: 3.0
  };
  
  try {
    const response = await fetch('http://localhost:8000/coupons/multi-league', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request)
    });
    
    const coupon = await response.json();
    console.log('Coupon généré:', coupon);
    
    // Afficher les prédictions
    coupon.predictions.forEach(pred => {
      console.log(`${pred.home_team} vs ${pred.away_team}: ${pred.prediction_value} (${pred.confidence}%)`);
    });
    
  } catch (error) {
    console.error('Erreur génération coupon:', error);
  }
};

// Récupérer ligues disponibles
const getLeagues = async () => {
  const response = await fetch('http://localhost:8000/leagues');
  const data = await response.json();
  return data.available_leagues;
};
```

### Angular Exemple

```typescript
import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class CouponService {
  private baseUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  generateLeagueCoupon(league: string, options: any) {
    return this.http.post(`${this.baseUrl}/coupons/league/${league}`, options);
  }

  generateMultiLeagueCoupon(options: any) {
    return this.http.post(`${this.baseUrl}/coupons/multi-league`, options);
  }

  getAllLeaguesCoupon(riskProfile = 'balanced', minConfidence = 70) {
    const params = `?risk_profile=${riskProfile}&min_confidence=${minConfidence}`;
    return this.http.post(`${this.baseUrl}/coupons/all-leagues${params}`, {});
  }
}
```

## 🧪 Tests et Validation

### Test Automatisé
```bash
python test_coupon_api.py
```

### Test Manuel avec cURL
```bash
# Test coupon Premier League
curl -X POST "http://localhost:8000/coupons/league/premier_league" \
     -H "Content-Type: application/json" \
     -d '{
       "risk_profile": "balanced",
       "min_confidence": 75,
       "max_predictions": 5
     }'

# Test coupon multi-ligues
curl -X POST "http://localhost:8000/coupons/multi-league" \
     -H "Content-Type: application/json" \
     -d '{
       "leagues": ["premier_league", "la_liga"],
       "risk_profile": "aggressive",
       "min_confidence": 65,
       "max_predictions": 8
     }'
```

## 📈 Métriques et Optimisation

### Métriques Retournées
- **total_odds** : Cote totale du coupon
- **expected_return** : Retour attendu (ROI)
- **risk_score** : Score de risque (0-100)
- **confidence_average** : Confiance moyenne
- **estimated_win_probability** : Probabilité de gain

### Optimisation Automatique
L'API optimise automatiquement :
- ✅ **Répartition équilibrée** entre ligues
- ✅ **Diversification des types** de prédictions  
- ✅ **Filtrage par confiance** et cotes
- ✅ **Exclusion d'équipes** spécifiques
- ✅ **Profils de risque** adaptatifs

## 🔒 Configuration Production

### CORS et Sécurité
```python
# Dans advanced_coupon_api.py - Modifier pour production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://votre-domain.com"],  # Spécifier domaines
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Variables d'Environnement
```bash
# .env
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
MODELS_PATH=models/complete_models
DATA_PATH=data/ultra_processed
```

### Déploiement
```bash
# Production avec Gunicorn
pip install gunicorn
gunicorn advanced_coupon_api:app -w 4 -b 0.0.0.0:8000

# Ou avec Uvicorn
uvicorn advanced_coupon_api:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📞 Support Frontend

### État de l'API
- ✅ **CORS activé** pour tous domaines (dev) 
- ✅ **JSON responses** standardisées
- ✅ **Validation Pydantic** automatique
- ✅ **Documentation OpenAPI** intégrée
- ✅ **Codes d'erreur HTTP** appropriés
- ✅ **Logging structuré** 

### Gestion d'Erreurs
```javascript
// Gestion d'erreurs frontend
const handleCouponError = (error) => {
  if (error.status === 422) {
    console.error('Validation error:', error.detail);
    // Afficher erreurs de validation à l'utilisateur
  } else if (error.status === 500) {
    console.error('Server error');
    // Afficher message d'erreur générique
  }
};
```

---

## 🎯 Cas d'Usage Typiques

### 1. Application Mobile de Paris
```javascript
// Coupon rapide équilibré
const quickCoupon = await generateMultiLeagueCoupon({
  leagues: ['premier_league', 'la_liga', 'bundesliga'],
  risk_profile: 'balanced',
  max_predictions: 6
});
```

### 2. Plateforme d'Investissement Sportif  
```javascript
// Coupon conservateur pour investisseurs
const safeCoupon = await generateAllLeaguesCoupon('conservative', 90, 4);
```

### 3. Site de Paris Experts
```javascript
// Coupon agressif haute variance
const expertCoupon = await generateMultiLeagueCoupon({
  risk_profile: 'aggressive',
  min_odds: 2.5,
  max_odds: 6.0
});
```

---

**🏆 API Prête pour Production - Frontend Ready !**