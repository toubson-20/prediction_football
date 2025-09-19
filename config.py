import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Chemins du projet
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = DATA_DIR / "models"
    
    # API Football Configuration (API-FOOTBALL.COM)
    FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY", "692266271feedb5e410f7771a3e39d87")
    FOOTBALL_API_BASE_URL = "https://v3.football.api-sports.io"
    
    # Rate limiting
    API_RATE_LIMIT_DELAY = 1  # secondes entre les requêtes
    
    # Principales compétitions pour l'entraînement
    TARGET_LEAGUES = {
        'Premier League': 39,      # England
        'La Liga': 140,           # Spain  
        'Ligue 1': 61,            # France
        'Bundesliga': 78,         # Germany
        'Champions League': 2,     # UEFA
        'Europa League': 3         # UEFA
    }
    
    # Saisons d'entraînement (historique + actuelle)
    TRAINING_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    CURRENT_SEASON = 2025  # Saison active pour prédictions temps réel (2025-2026)
    
    # Timezone Paris pour tous les matchs
    TIMEZONE_PARIS = "Europe/Paris"
    
    # Base de données
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/football_ml.db")
    
    # Configuration ML
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2
    
    # Modèles
    MODEL_NAMES = ["random_forest", "xgboost", "neural_network"]
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "localhost")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_WORKERS = int(os.getenv("API_WORKERS", 1))
    
    # Configuration OpenAI pour l'IA
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", 800))
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.3))
    
    # Configuration IA avancée
    AI_ENABLED = os.getenv("AI_ENABLED", "true").lower() == "true"
    AI_FALLBACK_ENABLED = os.getenv("AI_FALLBACK_ENABLED", "true").lower() == "true"
    AI_CACHE_ENABLED = os.getenv("AI_CACHE_ENABLED", "true").lower() == "true"
    AI_CACHE_TTL_MINUTES = int(os.getenv("AI_CACHE_TTL_MINUTES", 60))
    AI_MAX_CALLS_PER_HOUR = int(os.getenv("AI_MAX_CALLS_PER_HOUR", 500))
    AI_BUDGET_LIMIT_MONTHLY = float(os.getenv("AI_BUDGET_LIMIT_MONTHLY", 50.0))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def create_directories(cls):
        """Créer les dossiers nécessaires s'ils n'existent pas"""
        for dir_path in [cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, cls.MODELS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_ai_config(cls) -> bool:
        """Valider la configuration IA"""
        if not cls.AI_ENABLED:
            return True  # Si IA désactivée, pas besoin de validation
            
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == "your-openai-api-key-here":
            print("WARNING: Cle API OpenAI manquante dans .env")
            print("   Definir OPENAI_API_KEY=sk-votre-cle-api")
            return False
            
        if not cls.OPENAI_API_KEY.startswith("sk-"):
            print("ERROR: Format cle API OpenAI invalide")
            return False
            
        return True
    
    @classmethod
    def get_ai_config(cls) -> dict:
        """Obtenir la configuration IA formatée"""
        return {
            "api_key": cls.OPENAI_API_KEY,
            "model": cls.OPENAI_MODEL,
            "max_tokens": cls.OPENAI_MAX_TOKENS,
            "temperature": cls.OPENAI_TEMPERATURE,
            "enabled": cls.AI_ENABLED,
            "fallback_enabled": cls.AI_FALLBACK_ENABLED,
            "cache_enabled": cls.AI_CACHE_ENABLED,
            "cache_ttl": cls.AI_CACHE_TTL_MINUTES,
            "rate_limit": cls.AI_MAX_CALLS_PER_HOUR,
            "budget_limit": cls.AI_BUDGET_LIMIT_MONTHLY
        }

config = Config()

# Configuration API Football pour compatibilite
API_FOOTBALL_CONFIG = {
    "api_key": Config.FOOTBALL_API_KEY,
    "base_url": Config.FOOTBALL_API_BASE_URL,
    "rate_limit_delay": Config.API_RATE_LIMIT_DELAY
}