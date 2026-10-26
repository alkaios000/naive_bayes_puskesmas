import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """
    Konfigurasi global aplikasi diambil dari berkas .env
    Mendukung environment SQLite, MySQL, dan PostgreSQL.
    """
    APP_NAME: str = "Prediksi Tren Penyakit Puskesmas Sarang 1"
    DEBUG: bool = False
    
    # Keamanan & JWT
    SECRET_KEY: str = "super-secret-key-puskesmas-sarang-1-2026"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_HOURS: int = 8
    
    # Koneksi Basis Data
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/naive_bayes_puskesmas.db"
    
    # Pengguna Default (Seeder)
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "admin_puskesmas_sarang1"
    VIEWER_USERNAME: str = "viewer"
    VIEWER_PASSWORD: str = "viewer_puskesmas_sarang1"
    
    # Penyimpanan & Unggahan
    UPLOAD_DIR: str = str(BASE_DIR / "uploads")
    MODEL_STORAGE_DIR: str = str(BASE_DIR / "models_storage")
    MAX_UPLOAD_SIZE_MB: int = 5

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Pastikan direktori penting tersedia
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.MODEL_STORAGE_DIR, exist_ok=True)
