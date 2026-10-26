import os
import sys
from pathlib import Path

# Tambahkan root path ke sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.database import engine, Base, SessionLocal
from app.models import User, UserRole, Dataset, ModelRun, MonthlyCase
from app.config import settings
from app.security import get_password_hash
from app.services.ingest import validate_and_parse_file, save_dataset_to_db
from app.services.training import train_and_evaluate_model
from app.services.predict import run_predictions_for_next_month

def run_seed():
    """
    Inisialisasi tabel, membuat user admin & viewer awal,
    serta memuat dataset dummy Rekap_Bulanan_DUMMY_2023-2025.csv.
    """
    print("=== [1/4] Inisialisasi Skema Basis Data ===")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        print("=== [2/4] Membuat Akun Pengguna Default ===")
        # 1. Admin
        admin = db.query(User).filter(User.username == settings.ADMIN_USERNAME).first()
        if not admin:
            admin = User(
                username=settings.ADMIN_USERNAME,
                password_hash=get_password_hash(settings.ADMIN_PASSWORD),
                role=UserRole.ADMIN
            )
            db.add(admin)
            print(f"-> Akun Admin '{settings.ADMIN_USERNAME}' berhasil dibuat.")
        else:
            print(f"-> Akun Admin '{settings.ADMIN_USERNAME}' sudah tersedia.")

        # 2. Viewer
        viewer = db.query(User).filter(User.username == settings.VIEWER_USERNAME).first()
        if not viewer:
            viewer = User(
                username=settings.VIEWER_USERNAME,
                password_hash=get_password_hash(settings.VIEWER_PASSWORD),
                role=UserRole.VIEWER
            )
            db.add(viewer)
            print(f"-> Akun Viewer '{settings.VIEWER_USERNAME}' berhasil dibuat.")
        else:
            print(f"-> Akun Viewer '{settings.VIEWER_USERNAME}' sudah tersedia.")

        db.commit()
        db.refresh(admin)

        print("=== [3/4] Memuat Dataset Dummy 2023-2025 ===")
        # Cari file dataset dummy
        dummy_paths = [
            BASE_DIR / "dataset" / "Rekap_Bulanan_DUMMY_2023-2025.csv",
            BASE_DIR / "data" / "Rekap_Bulanan_DUMMY_2023-2025.csv",
        ]
        chosen_path = None
        for p in dummy_paths:
            if p.exists():
                chosen_path = p
                break

        dataset_id = None
        if chosen_path:
            filename = chosen_path.name
            existing_ds = db.query(Dataset).filter(Dataset.original_filename == filename).first()
            if not existing_ds:
                print(f"-> Membaca {chosen_path}...")
                with open(chosen_path, "rb") as f:
                    content = f.read()

                df, errors, is_dummy = validate_and_parse_file(content, filename)
                if errors:
                    print(f"Error validasi saat seeding: {errors}")
                else:
                    dataset = save_dataset_to_db(df, filename, is_dummy, admin.id, db)
                    dataset_id = dataset.id
                    print(f"-> Dataset '{dataset.name}' (ID: {dataset.id}) berhasil disimpan ({dataset.row_count} baris, {dataset.disease_count} penyakit).")
            else:
                dataset_id = existing_ds.id
                print(f"-> Dataset '{existing_ds.name}' sudah ada di database.")

        if dataset_id:
            print("=== [4/4] Menjalankan Pelatihan & Prediksi Awal ===")
            existing_run = db.query(ModelRun).filter(ModelRun.dataset_id == dataset_id).first()
            if not existing_run:
                print("-> Menjalankan pelatihan model Naive Bayes & pembanding...")
                model_run = train_and_evaluate_model(
                    dataset_id=dataset_id,
                    user_id=admin.id,
                    threshold_pct=0.10,
                    min_abs_change=3,
                    alpha=1.0,
                    test_ratio=0.20,
                    selected_features=None,
                    db=db
                )
                print(f"-> Model Run #{model_run.id} selesai dilatih.")

                print("-> Menghasilkan prediksi bulan berikutnya...")
                t_year, t_month, preds = run_predictions_for_next_month(model_run.id, db)
                print(f"-> Prediksi periode {t_year}-{t_month:02d} selesai ({len(preds)} penyakit).")
            else:
                print(f"-> Model Run #{existing_run.id} sudah ada.")

        print("=== SEEDING BERHASIL SELESAI ===")
    finally:
        db.close()

if __name__ == "__main__":
    run_seed()
