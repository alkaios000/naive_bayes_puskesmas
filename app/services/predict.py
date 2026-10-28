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
from app.services.training import encode_categorical_matrix

def compute_row_math_breakdown(
    nb_model,
    encoder,
    selected_features: List[str],
    row_disc: pd.Series,
    row_enc: np.ndarray,
    model_classes: List[str],
    p_final: Dict[str, float]
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Menghitung rincian matematis Naive Bayes per baris:
    prior, likelihood tiap atribut, penjumlahan log, dan penentu probabilitas terbesar.
    Memenuhi Lampiran B proposal dan transparansi model (Explainable AI).
    """
    class_indices = {cls_name: idx for idx, cls_name in enumerate(model_classes)}
    priors = {cls_name: round(float(np.exp(nb_model.class_log_prior_[idx])), 4) for cls_name, idx in class_indices.items()}

    feature_breakdowns = []
    log_sums = {cls_name: float(nb_model.class_log_prior_[idx]) for cls_name, idx in class_indices.items()}

    # Analisis kontribusi tiap atribut
    feature_advantages = []

    for feat_idx, feat_name in enumerate(selected_features):
        cat_idx = int(row_enc[feat_idx])
        raw_cat = str(row_disc.get(feat_name, "tidak_ada"))

        lh_per_class = {}
        log_lh_per_class = {}

        for cls_name, c_idx in class_indices.items():
            # Jika cat_idx melebihi kategori train, gunakan min_categories slot
            if cat_idx < nb_model.feature_log_prob_[feat_idx].shape[1]:
                log_p = float(nb_model.feature_log_prob_[feat_idx][c_idx, cat_idx])
            else:
                log_p = -10.0
            p_val = float(np.exp(log_p))

            lh_per_class[cls_name] = round(p_val, 4)
            log_lh_per_class[cls_name] = round(log_p, 4)
            log_sums[cls_name] += log_p

        feature_breakdowns.append({
            "feature": feat_name,
            "category": raw_cat,
            "likelihoods": lh_per_class,
            "log_likelihoods": log_lh_per_class
        })

        # Hitung keunggulan atribut terhadap kelas pemenang
        winning_class = max(p_final.items(), key=lambda x: x[1])[0]
        win_idx = class_indices[winning_class]
        win_log_p = log_lh_per_class[winning_class]
        other_logs = [log_lh_per_class[c] for c in model_classes if c != winning_class]
        avg_other = float(np.mean(other_logs)) if other_logs else win_log_p
        advantage = win_log_p - avg_other

        feature_advantages.append({
            "feature": feat_name,
            "category": raw_cat,
            "advantage": advantage
        })

    # Ambil 3 atribut paling mendukung kelas pemenang
    feature_advantages.sort(key=lambda x: x["advantage"], reverse=True)
    top_factors = [
        f"{fa['feature']} = {fa['category']} (skor bobot: +{round(fa['advantage'], 2)})"
        for fa in feature_advantages[:3]
    ]

    math_breakdown = {
        "priors": priors,
        "feature_likelihoods": feature_breakdowns,
        "log_sums": {k: round(v, 4) for k, v in log_sums.items()},
        "posterior_probabilities": p_final
    }

    return top_factors, math_breakdown

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
    top_diseases_list = artifact.get("top_diseases_list")

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

        # Jika model dilatih khusus top-N, filter hanya top penyakit tersebut
        if top_diseases_list is not None and icd not in top_diseases_list:
            continue

        # Ambil data historis terurut untuk diagnosis ini
        d_history = df_labeled[df_labeled["kode_icd10"] == icd].sort_values(by=["tahun", "bulan"]).to_dict("records")
        if not d_history:
            continue

        lag1 = d_history[-1]
        lag2 = d_history[-2] if len(d_history) >= 2 else None

        # Fallback konsisten: gunakan 'tidak_ada' jika riwayat belum memiliki label
        lbl1 = lag1.get("label")
        label_lag1 = str(lbl1) if (lbl1 is not None and pd.notna(lbl1)) else "tidak_ada"
        lbl2 = lag2.get("label") if lag2 else None
        label_lag2 = str(lbl2) if (lbl2 is not None and pd.notna(lbl2)) else "tidak_ada"
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
        if tk is not None and pd.notna(tk) and int(tk) > 0:
            total_kunj_lag1 = float(tk)
            rasio_kunj = float(lag1.get("jumlah_kasus", 0)) / total_kunj_lag1
        else:
            total_kunj_lag1 = None
            rasio_kunj = None

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
            "total_kunjungan_lag1": total_kunj_lag1,
            "curah_hujan_lag1": ch,
            "suhu_lag1": suhu,
        })

    df_predict_raw = pd.DataFrame(predict_rows)

    # 6. Diskretisasi kuantil menggunakan model artifact
    numeric_in_features = [f for f in selected_features if f in NUMERIC_LAG_FEATURES]
    df_predict_disc = discretizer.transform(df_predict_raw, numeric_in_features)

    # 7. Encoding Ordinal (dengan penanganan unseen categories yang aman)
    X_pred_enc = encode_categorical_matrix(encoder, df_predict_disc, selected_features)

    # 8. Inferensi Prediksi & Probabilitas
    pred_classes = nb_model.predict(X_pred_enc)
    pred_probas = nb_model.predict_proba(X_pred_enc)

    # 9. Format hasil, penjelasan top factors, dan perhitungan matematis
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
            p_turun = round(1.0 - (p_naik + p_stabil), 4)
        else:
            p_naik, p_stabil, p_turun = 0.3333, 0.3334, 0.3333

        p_final = {"Naik": p_naik, "Stabil": p_stabil, "Turun": p_turun}

        # Hitung faktor penentu dan rincian rumus matematis
        top_factors, math_breakdown = compute_row_math_breakdown(
            nb_model=nb_model,
            encoder=encoder,
            selected_features=selected_features,
            row_disc=df_predict_disc.iloc[i],
            row_enc=X_pred_enc[i],
            model_classes=model_classes,
            p_final=p_final
        )

        pred_obj = Prediction(
            model_run_id=model_run_id,
            disease_id=d_id,
            target_year=target_year,
            target_month=target_month,
            predicted_class=TrendClass(predicted_class_str),
            p_naik=p_naik,
            p_stabil=p_stabil,
            p_turun=p_turun,
            top_factors_json=json.dumps(top_factors),
            math_breakdown_json=json.dumps(math_breakdown)
        )
        db.add(pred_obj)
        created_predictions.append(pred_obj)

    db.commit()
    return target_year, target_month, created_predictions
