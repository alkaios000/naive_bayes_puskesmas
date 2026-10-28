from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

# Inisialisasi engine database (mendukung SQLite, MySQL, Postgres)
is_sqlite = settings.DATABASE_URL.startswith("sqlite")
connect_args = {"check_same_thread": False} if is_sqlite else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """
    Dependency generator untuk mendapatkan sesi database per-request
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db_schema():
    """
    Membuat tabel jika belum ada dan memperbarui skema kolom baru pada SQLite
    """
    Base.metadata.create_all(bind=engine)
    if is_sqlite:
        with engine.connect() as conn:
            # Periksa tabel model_runs
            existing_cols = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(model_runs)").fetchall()]
            new_model_run_cols = [
                ("top_n_diseases", "INTEGER"),
                ("cv_scores_json", "TEXT"),
                ("group_ablation_json", "TEXT"),
                ("disease_metrics_json", "TEXT"),
                ("preprocessing_report_json", "TEXT"),
                ("threshold_sensitivity_json", "TEXT"),
                ("walk_forward_json", "TEXT"),
            ]
            for col_name, col_type in new_model_run_cols:
                if col_name not in existing_cols:
                    conn.exec_driver_sql(f"ALTER TABLE model_runs ADD COLUMN {col_name} {col_type}")

            # Periksa tabel predictions
            pred_cols = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(predictions)").fetchall()]
            new_pred_cols = [
                ("top_factors_json", "TEXT"),
                ("math_breakdown_json", "TEXT"),
            ]
            for col_name, col_type in new_pred_cols:
                if col_name not in pred_cols:
                    conn.exec_driver_sql(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
            conn.commit()

