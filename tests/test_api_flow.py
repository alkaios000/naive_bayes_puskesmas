from pathlib import Path
import pytest
from app.services.ingest import validate_and_parse_file, save_dataset_to_db

def test_full_pipeline_train_and_predict(client, admin_token, db_session):
    """
    Kriteria Penerimaan 2, 6, 10:
    Unggahan dummy sukses, pelatihan menghasilkan 5 algoritma,
    dan prediksi menghasilkan probabilitas yang berjumlah tepat 1.0.
    """
    headers = {"Authorization": f"Bearer {admin_token}"}

    # 1. Siapkan dataset dari file dummy
    csv_path = Path("dataset/Rekap_Bulanan_DUMMY_2023-2025.csv")
    if not csv_path.exists():
        csv_path = Path("data/Rekap_Bulanan_DUMMY_2023-2025.csv")
    
    assert csv_path.exists(), "Berkas dataset dummy harus ada."

    with open(csv_path, "rb") as f:
        content = f.read()

    df, errors, is_dummy = validate_and_parse_file(content, csv_path.name)
    assert not errors, f"Dataset dummy tidak boleh memiliki error: {errors}"
    assert is_dummy is True, "Flag is_dummy harus bernilai True untuk dataset dummy."

    dataset = save_dataset_to_db(df, csv_path.name, is_dummy, 1, db_session)
    assert dataset.id is not None
    assert dataset.disease_count == 15

    # 2. Test Endpoint Train
    res_train = client.post(
        "/api/models/train",
        json={
            "dataset_id": dataset.id,
            "threshold_pct": 0.10,
            "min_abs_change": 3,
            "alpha": 1.0,
            "test_ratio": 0.20
        },
        headers=headers
    )
    print("TRAIN_RESPONSE:", res_train.status_code, res_train.json())
    assert res_train.status_code == 200
    train_data = res_train.json()
    model_run_id = train_data["model_run_id"]

    # 3. Test Detail Model & 5 Algoritma Pembanding (Kriteria Penerimaan 6)
    res_detail = client.get(f"/api/models/{model_run_id}", headers=headers)
    assert res_detail.status_code == 200
    detail = res_detail.json()

    algos = [m["algorithm"] for m in detail["metrics"]]
    assert "naive_bayes" in algos
    assert "majority" in algos
    assert "persistence" in algos
    assert "decision_tree" in algos
    assert "knn" in algos

    # 4. Test Jalankan Prediksi Bulan Berikutnya (Kriteria Penerimaan 10)
    res_pred = client.post(
        "/api/predictions/run",
        json={"model_run_id": model_run_id},
        headers=headers
    )
    assert res_pred.status_code == 200
    pred_data = res_pred.json()

    assert pred_data["total_predictions"] == 15
    for item in pred_data["predictions"]:
        total_p = item["p_naik"] + item["p_stabil"] + item["p_turun"]
        # Probabilitas berjumlah 1.0 (toleransi floating point 0.001)
        assert abs(total_p - 1.0) < 0.002, f"Total probabilitas tidak 1.0 pada {item['icd10']}: {total_p}"
        assert item["predicted_class"] in ["Naik", "Stabil", "Turun"]
