import os
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from sqlalchemy.orm import Session

from app.models import (
    ModelRun, Prediction, Disease, MonthlyCase, TrendClass
)
from app.services.features import (
    determine_season, FeatureDiscretizer, NUMERIC_LAG_FEATURES
)
from app.services.labeling import generate_labels_for_dataframe
from app.services.evaluation import CLASSES

def run_predictions_for_next_month(model_run_id: int, db: Session) -> Tuple[int, int, List[Prediction]]:
    """
    Memprediksi tren jumlah kasus bulan berikutnya untuk seluruh penyakit pada dataset.
    Memenuhi Kriteria Penerimaan Bagian 13:
    Probabilitas 3 kelas (Naik, Stabil, Turun) berjumlah tepat 1.0 per baris.
    """
    model_run = db.query(ModelRun).filter(ModelRun.id == model_run_id).first()
    if not model_run:
        raise ValueError(f"ModelRun ID {model_run_id} tidak ditemukan.")

    if not model_run.model_path or not os.path.exists(model_run.model_path):
        raise ValueError("Berkas model terlatih tidak ditemukan di storage.")

    # 1. Muat artefak model
    artifact = joblib.load(model_run.model_path)
    nb_model = artifact["nb_model"]
    encoder = artifact["encoder"]
    discretizer: FeatureDiscretizer = artifact["discretizer"]
    selected_features = artifact["selected_features"]
    model_classes = list(artifact["classes"])

    # 2. Ambil seluruh data kasus riwayat pada dataset
    cases_query = (
        db.query(
            MonthlyCase.year.label("tahun"),
            MonthlyCase.month.label("bulan"),
            Disease.id.label("disease_id"),
            Disease.icd10.label("kode_icd10"),
            Disease.name.label("nama_penyakit"),
            MonthlyCase.cases.label("jumlah_kasus"),
            MonthlyCase.male.label("kasus_laki"),
            MonthlyCase.female.label("kasus_perempuan"),
            MonthlyCase.age_lt1.label("kasus_lt1th"),
            MonthlyCase.age_1_5.label("kasus_1_5th"),
            MonthlyCase.age_6_14.label("kasus_6_14th"),
            MonthlyCase.age_15_59.label("kasus_15_59th"),
            MonthlyCase.age_gte60.label("kasus_gte60th"),
            MonthlyCase.total_visits.label("total_kunjungan"),
            MonthlyCase.rainfall_mm.label("curah_hujan_mm"),
            MonthlyCase.temp_avg.label("suhu_rata2"),
        )
        .join(Disease, MonthlyCase.disease_id == Disease.id)
        .filter(MonthlyCase.dataset_id == model_run.dataset_id)
        .order_by(MonthlyCase.year, MonthlyCase.month, Disease.icd10)
    )
    df_raw = pd.DataFrame([dict(r._mapping) for r in cases_query.all()])
    if df_raw.empty:
        raise ValueError("Data historis tidak ditemukan untuk prediksi.")

    # 3. Bentuk label historis
    df_labeled = generate_labels_for_dataframe(
        df_raw, model_run.threshold_pct, model_run.min_abs_change
    )

    # 4. Tentukan periode target (bulan setelah bulan terakhir data)
    periods = df_raw[["tahun", "bulan"]].drop_duplicates().sort_values(by=["tahun", "bulan"]).to_dict("records")
    latest_p = periods[-1]
    latest_y = int(latest_p["tahun"])
    latest_m = int(latest_p["bulan"])

    target_year = latest_y if latest_m < 12 else latest_y + 1
    target_month = latest_m + 1 if latest_m < 12 else 1
    target_season = determine_season(target_month)

    # 5. Bangun baris input fitur untuk setiap penyakit
    predict_rows = []
    diseases = df_raw[["disease_id", "kode_icd10", "nama_penyakit"]].drop_duplicates().to_dict("records")

    for d in diseases:
        d_id = d["disease_id"]
        icd = d["kode_icd10"]
        d_name = d["nama_penyakit"]

        # Ambil data historis terurut untuk diagnosis ini
        d_history = df_labeled[df_labeled["kode_icd10"] == icd].sort_values(by=["tahun", "bulan"]).to_dict("records")
        if not d_history:
            continue

        lag1 = d_history[-1]
        lag2 = d_history[-2] if len(d_history) >= 2 else None

        label_lag1 = lag1["label"] if lag1["label"] is not None else "Stabil"
        label_lag2 = lag2["label"] if (lag2 and lag2["label"] is not None) else "tidak_ada"
        cases_lag1 = max(int(lag1.get("jumlah_kasus", 0) or 0), 1)

        prop_laki = float(lag1["kasus_laki"]) / cases_lag1 if lag1.get("kasus_laki") is not None and pd.notna(lag1.get("kasus_laki")) else None

        def get_prop(col):
            val = lag1.get(col)
            return float(val) / cases_lag1 if val is not None and pd.notna(val) else None

        prop_lt1 = get_prop("kasus_lt1th")
        prop_1_5 = get_prop("kasus_1_5th")
        prop_6_14 = get_prop("kasus_6_14th")
        prop_15_59 = get_prop("kasus_15_59th")
        prop_gte60 = get_prop("kasus_gte60th")

        tk = lag1.get("total_kunjungan")
        rasio_kunj = float(lag1.get("jumlah_kasus", 0)) / float(tk) if tk is not None and pd.notna(tk) and int(tk) > 0 else None

        ch = float(lag1["curah_hujan_mm"]) if lag1.get("curah_hujan_mm") is not None and pd.notna(lag1.get("curah_hujan_mm")) else None
        suhu = float(lag1["suhu_rata2"]) if lag1.get("suhu_rata2") is not None and pd.notna(lag1.get("suhu_rata2")) else None

        predict_rows.append({
            "disease_id": d_id,
            "kode_icd10": icd,
            "nama_penyakit": d_name,
            "tahun": target_year,
            "bulan": target_month,
            "bulan_target": str(target_month),
            "musim": target_season,
            "label_lag1": label_lag1,
            "label_lag2": label_lag2,
            "proporsi_laki_lag1": prop_laki,
            "proporsi_umur_lt1_lag1": prop_lt1,
            "proporsi_umur_1_5_lag1": prop_1_5,
            "proporsi_umur_6_14_lag1": prop_6_14,
            "proporsi_umur_15_59_lag1": prop_15_59,
            "proporsi_umur_gte60_lag1": prop_gte60,
            "rasio_kasus_kunjungan_lag1": rasio_kunj,
            "curah_hujan_lag1": ch,
            "suhu_lag1": suhu,
        })

    df_predict_raw = pd.DataFrame(predict_rows)

    # 6. Diskretisasi kuantil menggunakan model artifact
    numeric_in_features = [f for f in selected_features if f in NUMERIC_LAG_FEATURES]
    df_predict_disc = discretizer.transform(df_predict_raw, numeric_in_features)

    # 7. Encoding Ordinal
    X_pred_raw = df_predict_disc[selected_features].astype(str)
    X_pred_enc = encoder.transform(X_pred_raw)
    X_pred_enc[X_pred_enc == 999] = 0

    # 8. Inferensi Prediksi & Probabilitas
    pred_classes = nb_model.predict(X_pred_enc)
    pred_probas = nb_model.predict_proba(X_pred_enc)

    # 9. Format hasil dan pastikan probabilitas berjumlah 1.0 per baris
    # Hapus prediksi lama untuk model_run_id ini jika ada
    db.query(Prediction).filter(Prediction.model_run_id == model_run_id).delete()

    created_predictions: List[Prediction] = []
    for i, row in df_predict_raw.iterrows():
        d_id = int(row["disease_id"])
        predicted_class_str = str(pred_classes[i])

        prob_map = {cls: 0.0 for cls in CLASSES}
        for j, cls_name in enumerate(model_classes):
            if cls_name in prob_map:
                prob_map[cls_name] = float(pred_probas[i][j])

        # Normalisasi tepat berjumlah 1.0
        total_p = sum(prob_map.values())
        if total_p > 0:
            p_naik = round(prob_map["Naik"] / total_p, 4)
            p_stabil = round(prob_map["Stabil"] / total_p, 4)
            # Pastikan pembulatan tidak membuat selisih
            p_turun = round(1.0 - (p_naik + p_stabil), 4)
        else:
            p_naik, p_stabil, p_turun = 0.3333, 0.3334, 0.3333

        pred_obj = Prediction(
            model_run_id=model_run_id,
            disease_id=d_id,
            target_year=target_year,
            target_month=target_month,
            predicted_class=TrendClass(predicted_class_str),
            p_naik=p_naik,
            p_stabil=p_stabil,
            p_turun=p_turun
        )
        db.add(pred_obj)
        created_predictions.append(pred_obj)

    db.commit()
    return target_year, target_month, created_predictions
