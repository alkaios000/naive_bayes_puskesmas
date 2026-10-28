from pathlib import Path
import pytest
from app.services.ingest import validate_and_parse_file, save_dataset_to_db

def test_proposal_features_pipeline(client, admin_token, db_session):
    """
    Menguji fitur pengembangan baru yang dijanjikan proposal:
    1. min_abs_change = 0 (rumus proposal murni)
    2. top_n_diseases = 10 (filter 10 penyakit terbanyak tanpa kebocoran)
    3. Validasi silang deret waktu expanding window
    4. Probabilitas bersyarat P(Kategori|Kelas)
    5. Detail perhitungan matematis Naive Bayes per prediksi
    6. Ekspor tabel AIMRaD dan P(X|Y) CSV
    """
    headers = {"Authorization": f"Bearer {admin_token}"}

    # 1. Dataset
    csv_path = Path("dataset/Rekap_Bulanan_DUMMY_2023-2025.csv")
    if not csv_path.exists():
        csv_path = Path("data/Rekap_Bulanan_DUMMY_2023-2025.csv")
    assert csv_path.exists()

    with open(csv_path, "rb") as f:
        content = f.read()

    df, errors, is_dummy = validate_and_parse_file(content, csv_path.name)
    assert not errors
    dataset = save_dataset_to_db(df, csv_path.name, is_dummy, 1, db_session)

    # 2. Latih Model dengan Preset Proposal (min_abs_change=0, top_n_diseases=10)
    res_train = client.post(
        "/api/models/train",
        json={
            "dataset_id": dataset.id,
            "threshold_pct": 0.10,
            "min_abs_change": 0, # Rumus murni proposal
            "alpha": 1.0,
            "test_ratio": 0.20,
            "top_n_diseases": 10
        },
        headers=headers
    )
    assert res_train.status_code == 200
    model_run_id = res_train.json()["model_run_id"]

    # 3. Verifikasi Detail Model Run
    res_detail = client.get(f"/api/models/{model_run_id}", headers=headers)
    assert res_detail.status_code == 200
    detail = res_detail.json()
    assert detail["min_abs_change"] == 0
    assert detail["top_n_diseases"] == 10
    assert detail["cv_scores"] is not None
    assert "mean_f1" in detail["cv_scores"]
    assert detail["group_ablation"] is not None
    assert len(detail["group_ablation"]) > 0

    # 4. Verifikasi Probabilitas Bersyarat
    res_cp = client.get(f"/api/models/{model_run_id}/conditional-probabilities", headers=headers)
    assert res_cp.status_code == 200
    cond_probs = res_cp.json()
    assert len(cond_probs) > 0
    for cp in cond_probs[:5]:
        assert "feature" in cp
        assert "category" in cp
        assert "p_naik" in cp
        assert "p_stabil" in cp
        assert "p_turun" in cp

    # 5. Jalankan Prediksi
    res_pred = client.post("/api/predictions/run", json={"model_run_id": model_run_id}, headers=headers)
    assert res_pred.status_code == 200
    pred_data = res_pred.json()
    assert pred_data["total_predictions"] == 10 # Karena difilter ke 10 penyakit teratas

    first_pred = pred_data["predictions"][0]
    assert len(first_pred["top_factors"]) > 0

    # 6. Verifikasi Detail Matematis Naive Bayes
    res_math = client.get(f"/api/predictions/{first_pred['id']}/math-breakdown", headers=headers)
    assert res_math.status_code == 200
    math_detail = res_math.json()
    assert "breakdown" in math_detail
    assert "priors" in math_detail["breakdown"]
    assert "feature_likelihoods" in math_detail["breakdown"]

    # 7. Verifikasi Ekspor AIMRaD CSV & P(X|Y) CSV
    res_exp_aimrad = client.get(f"/api/export/aimrad_tables.csv?model_run_id={model_run_id}", headers=headers)
    assert res_exp_aimrad.status_code == 200
    assert "TABEL 1. PERBANDINGAN KINERJA ALGORITMA" in res_exp_aimrad.text

    res_exp_cp = client.get(f"/api/export/conditional_probabilities.csv?model_run_id={model_run_id}", headers=headers)
    assert res_exp_cp.status_code == 200
    assert "P(Kategori|Naik)" in res_exp_cp.text
