import os
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from sqlalchemy.orm import Session

from app.models import (
    ModelRun, ModelMetric, FeatureInfluence, Dataset, MonthlyCase, Disease,
    ModelRunStatus, AlgorithmType, TrendClass
)
from app.config import settings
from app.services.labeling import generate_labels_for_dataframe
from app.services.features import (
    build_feature_matrix, FeatureDiscretizer, ALL_AVAILABLE_FEATURES, NUMERIC_LAG_FEATURES
)
from app.services.evaluation import calculate_metrics, CLASSES

class BaselineMajority:
    """Baseline 1: Selalu menebak kelas mayoritas di data latih"""
    def __init__(self):
        self.majority_class = "Stabil"

    def fit(self, X, y):
        counts = pd.Series(y).value_counts()
        if len(counts) > 0:
            self.majority_class = counts.index[0]
        return self

    def predict(self, X):
        return [self.majority_class] * len(X)

class BaselinePersistence:
    """Baseline 2: Menebak label bulan t-1 untuk bulan t"""
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, df_features: pd.DataFrame):
        preds = []
        for val in df_features["label_lag1"]:
            if val in CLASSES:
                preds.append(val)
            else:
                preds.append("Stabil")
        return preds

def split_train_test_by_time(df: pd.DataFrame, test_ratio: float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    Split berdasarkan urutan waktu tanpa pengacakan (PRD Bagian 5.3):
    test_ratio bulan terakhir sebagai data uji.
    """
    periods = df[["tahun", "bulan"]].drop_duplicates().sort_values(by=["tahun", "bulan"]).to_dict("records")
    total_periods = len(periods)
    if total_periods < 2:
        raise ValueError("Data memiliki terlalu sedikit periode untuk dipecah latih/uji.")

    n_test_periods = max(1, int(round(total_periods * test_ratio)))
    n_train_periods = total_periods - n_test_periods

    train_period_tuples = set((p["tahun"], p["bulan"]) for p in periods[:n_train_periods])
    test_period_tuples = set((p["tahun"], p["bulan"]) for p in periods[n_train_periods:])

    df_train = df[df.apply(lambda r: (r["tahun"], r["bulan"]) in train_period_tuples, axis=1)].copy()
    df_test = df[df.apply(lambda r: (r["tahun"], r["bulan"]) in test_period_tuples, axis=1)].copy()

    first_tr = periods[0]
    last_tr = periods[n_train_periods - 1]
    first_te = periods[n_train_periods]
    last_te = periods[-1]

    train_period_str = f"{first_tr['tahun']}-{first_tr['bulan']:02d} s.d. {last_tr['tahun']}-{last_tr['bulan']:02d}"
    test_period_str = f"{first_te['tahun']}-{first_te['bulan']:02d} s.d. {last_te['tahun']}-{last_te['bulan']:02d}"

    return df_train, df_test, train_period_str, test_period_str

def train_and_evaluate_model(
    dataset_id: int,
    user_id: Optional[int],
    threshold_pct: float,
    min_abs_change: int,
    alpha: float,
    test_ratio: float,
    selected_features: Optional[List[str]],
    db: Session
) -> ModelRun:
    """
    Menjalankan proses pelatihan dan evaluasi model secara komprehensif sesuai PRD.
    """
    # 1. Ambil data kasus dari DB
    cases_query = (
        db.query(
            MonthlyCase.year.label("tahun"),
            MonthlyCase.month.label("bulan"),
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
        .filter(MonthlyCase.dataset_id == dataset_id)
        .order_by(MonthlyCase.year, MonthlyCase.month, Disease.icd10)
    )
    df_raw = pd.DataFrame([dict(r._mapping) for r in cases_query.all()])
    if df_raw.empty:
        raise ValueError(f"Dataset ID {dataset_id} tidak memiliki data kasus.")

    # 2. Tentukan fitur yang dipakai
    if not selected_features:
        selected_features = [f for f in ALL_AVAILABLE_FEATURES if f != "musim"] # default subset representatif

    # 3. Bentuk label tren
    df_labeled = generate_labels_for_dataframe(df_raw, threshold_pct, min_abs_change)

    # 4. Bangun matriks fitur lag (pencegahan kebocoran)
    df_feat_matrix = build_feature_matrix(df_labeled)

    # Buang baris yang target_label-nya None (bulan pertama tiap penyakit)
    df_valid = df_feat_matrix.dropna(subset=["target_label"]).copy()

    # 5. Split berdasarkan waktu
    df_train, df_test, train_period_str, test_period_str = split_train_test_by_time(df_valid, test_ratio)

    # 6. Diskretisasi kuantil hanya di data latih
    discretizer = FeatureDiscretizer()
    numeric_in_features = [f for f in selected_features if f in NUMERIC_LAG_FEATURES]
    discretizer.fit(df_train, numeric_in_features)

    df_train_disc = discretizer.transform(df_train, numeric_in_features)
    df_test_disc = discretizer.transform(df_test, numeric_in_features)

    # 7. Encoding Ordinal untuk model ML
    X_train_raw = df_train_disc[selected_features].astype(str)
    X_test_raw = df_test_disc[selected_features].astype(str)

    y_train = df_train["target_label"].tolist()
    y_test = df_test["target_label"].tolist()

    # Bangun kategori mapping dari data latih
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=999)
    encoder.fit(X_train_raw)

    X_train_enc = encoder.transform(X_train_raw)
    X_test_enc = encoder.transform(X_test_raw)

    # Tangani unknown di X_test_enc (ganti dengan index kategori pertama 0)
    X_test_enc[X_test_enc == 999] = 0

    # 8. Tuning hyperparameter alpha untuk Naive Bayes (TimeSeriesSplit pada data latih)
    alphas_to_try = [0.1, 0.5, 1.0, 2.0]
    best_alpha = alpha
    best_cv_f1 = -1.0

    unique_train_p = df_train[["tahun", "bulan"]].drop_duplicates().sort_values(by=["tahun", "bulan"])
    n_splits = min(5, max(2, len(unique_train_p) - 1))

    if n_splits >= 2:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for cand_alpha in alphas_to_try:
            fold_f1s = []
            for tr_idx, val_idx in tscv.split(X_train_enc):
                X_tr_f, X_val_f = X_train_enc[tr_idx], X_train_enc[val_idx]
                y_tr_f, y_val_f = [y_train[i] for i in tr_idx], [y_train[i] for i in val_idx]
                
                # Cek jika minimal ada 2 kelas di fold latih
                if len(set(y_tr_f)) < 2:
                    continue
                try:
                    # Tentukan min_categories untuk mengatasi nilai tak terlihat antar-fold
                    min_cats = [len(cats) + 1 for cats in encoder.categories_]
                    nb_fold = CategoricalNB(alpha=cand_alpha, min_categories=min_cats)
                    nb_fold.fit(X_tr_f, y_tr_f)
                    preds_val = nb_fold.predict(X_val_f)
                    fold_f1 = f1_score(y_val_f, preds_val, average="macro", zero_division=0)
                    fold_f1s.append(fold_f1)
                except Exception:
                    pass
            if fold_f1s:
                mean_f1 = float(np.mean(fold_f1s))
                if mean_f1 > best_cv_f1:
                    best_cv_f1 = mean_f1
                    best_alpha = cand_alpha

    # 9. Latih Naive Bayes Utama
    min_cats = [len(cats) + 1 for cats in encoder.categories_]
    nb_model = CategoricalNB(alpha=best_alpha, min_categories=min_cats)
    nb_model.fit(X_train_enc, y_train)
    y_pred_nb = nb_model.predict(X_test_enc)

    # 10. Latih Model Pembanding (PRD Bagian 5.4)
    # Mayoritas
    maj_model = BaselineMajority().fit(X_train_enc, y_train)
    y_pred_maj = maj_model.predict(X_test_enc)

    # Persistence
    pers_model = BaselinePersistence().fit(X_train_enc, y_train)
    y_pred_pers = pers_model.predict(df_test)

    # Decision Tree (max_depth=4)
    dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_model.fit(X_train_enc, y_train)
    y_pred_dt = dt_model.predict(X_test_enc)

    # KNN (k=5)
    k_val = min(5, len(X_train_enc))
    knn_model = KNeighborsClassifier(n_neighbors=k_val)
    knn_model.fit(X_train_enc, y_train)
    y_pred_knn = knn_model.predict(X_test_enc)

    # 11. Hitung Metrik Semua Model
    metrics_data = {
        AlgorithmType.NAIVE_BAYES: calculate_metrics(y_test, y_pred_nb),
        AlgorithmType.MAJORITY: calculate_metrics(y_test, y_pred_maj),
        AlgorithmType.PERSISTENCE: calculate_metrics(y_test, y_pred_pers),
        AlgorithmType.DECISION_TREE: calculate_metrics(y_test, y_pred_dt),
        AlgorithmType.KNN: calculate_metrics(y_test, y_pred_knn),
    }

    # 12. Drop-one-feature (Pengaruh Atribut)
    full_f1_macro = metrics_data[AlgorithmType.NAIVE_BAYES]["f1_macro"]
    feature_influences: List[Dict[str, Any]] = []

    if len(selected_features) > 1:
        for feat in selected_features:
            sub_feats = [f for f in selected_features if f != feat]
            X_tr_sub = df_train_disc[sub_feats].astype(str)
            X_te_sub = df_test_disc[sub_feats].astype(str)

            enc_sub = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=999)
            enc_sub.fit(X_tr_sub)

            X_tr_sub_enc = enc_sub.transform(X_tr_sub)
            X_te_sub_enc = enc_sub.transform(X_te_sub)
            X_te_sub_enc[X_te_sub_enc == 999] = 0

            sub_min_cats = [len(cats) + 1 for cats in enc_sub.categories_]
            nb_sub = CategoricalNB(alpha=best_alpha, min_categories=sub_min_cats)
            nb_sub.fit(X_tr_sub_enc, y_train)
            sub_preds = nb_sub.predict(X_te_sub_enc)

            f1_sub = float(f1_score(y_test, sub_preds, average="macro", zero_division=0))
            delta_f1 = full_f1_macro - f1_sub

            feature_influences.append({
                "feature": feat,
                "f1_macro_without": round(f1_sub, 4),
                "delta_f1": round(delta_f1, 4)
            })

        feature_influences.sort(key=lambda x: x["delta_f1"], reverse=True)

    # 13. Simpan Run ke Database
    dist_counts = pd.Series(y_train + y_test).value_counts().to_dict()

    model_run = ModelRun(
        dataset_id=dataset_id,
        created_by=user_id,
        threshold_pct=threshold_pct,
        min_abs_change=min_abs_change,
        alpha=best_alpha,
        test_ratio=test_ratio,
        features_json=json.dumps(selected_features),
        train_rows=len(df_train),
        test_rows=len(df_test),
        train_period=train_period_str,
        test_period=test_period_str,
        class_distribution_json=json.dumps(dist_counts),
        status=ModelRunStatus.DONE
    )
    db.add(model_run)
    db.flush()

    # Simpan metrik
    for algo, m in metrics_data.items():
        mm = ModelMetric(
            model_run_id=model_run.id,
            algorithm=algo,
            accuracy=m["accuracy"],
            f1_macro=m["f1_macro"],
            precision_json=json.dumps(m["precision"]),
            recall_json=json.dumps(m["recall"]),
            f1_json=json.dumps(m["f1"]),
            confusion_matrix_json=json.dumps(m["confusion_matrix"]),
        )
        db.add(mm)

    # Simpan feature influence
    for fi in feature_influences:
        db.add(FeatureInfluence(
            model_run_id=model_run.id,
            feature=fi["feature"],
            f1_macro_without=fi["f1_macro_without"],
            delta_f1=fi["delta_f1"]
        ))

    # 14. Simpan Artifak Model ke disk
    model_filename = f"model_run_{model_run.id}.joblib"
    model_filepath = os.path.join(settings.MODEL_STORAGE_DIR, model_filename)
    joblib.dump({
        "nb_model": nb_model,
        "encoder": encoder,
        "discretizer": discretizer,
        "selected_features": selected_features,
        "numeric_features": numeric_in_features,
        "classes": list(nb_model.classes_),
        "alpha": best_alpha,
        "threshold_pct": threshold_pct,
        "min_abs_change": min_abs_change
    }, model_filepath)

    model_run.model_path = model_filepath
    db.commit()
    db.refresh(model_run)

    return model_run
