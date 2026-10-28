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
from sklearn.metrics import f1_score
from sqlalchemy.orm import Session

from app.models import (
    ModelRun, ModelMetric, FeatureInfluence, Dataset, MonthlyCase, Disease,
    ModelRunStatus, AlgorithmType, TrendClass
)
from app.config import settings
from app.services.labeling import generate_labels_for_dataframe
from app.services.features import (
    build_feature_matrix, FeatureDiscretizer, ALL_AVAILABLE_FEATURES, 
    NUMERIC_LAG_FEATURES, FEATURE_GROUPS
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

def encode_categorical_matrix(encoder: OrdinalEncoder, df: pd.DataFrame, features: List[str]) -> np.ndarray:
    """
    Meng-encode matriks fitur kategorikal.
    Nilai tak dikenal (unseen) di-mapping ke index khusus cadangan (len(cats))
    sehingga mendapat penanganan smoothing Laplace yang adil tanpa merusak kategori index 0.
    """
    raw = df[features].astype(str)
    encoded = encoder.transform(raw)
    for col_idx, cats in enumerate(encoder.categories_):
        unseen_slot = len(cats)
        encoded[encoded[:, col_idx] == -1, col_idx] = unseen_slot
    return encoded

def run_period_expanding_window_cv(
    df_train_raw: pd.DataFrame,
    selected_features: List[str],
    numeric_in_features: List[str],
    alphas_to_try: List[float] = [0.1, 0.5, 1.0, 2.0],
    n_splits: int = 4
) -> Tuple[float, Dict[str, Any]]:
    """
    Validasi Silang Deret Waktu (Expanding Window) berbasis Periode Kalender.
    Membagi fold secara temporal per-blok bulan untuk seluruh diagnosis (tanpa kebocoran antar penyakit).
    """
    periods = df_train_raw[["tahun", "bulan"]].drop_duplicates().sort_values(by=["tahun", "bulan"]).to_dict("records")
    n_periods = len(periods)
    if n_periods < 4:
        return 1.0, {"folds": [], "mean_f1": 0.0, "std_f1": 0.0, "best_alpha": 1.0}

    min_train_periods = max(4, n_periods // 2)
    step = max(1, (n_periods - min_train_periods) // n_splits)

    alpha_scores: Dict[float, List[float]] = {a: [] for a in alphas_to_try}
    fold_reports = []

    split_count = 0
    for split_point in range(min_train_periods, n_periods, step):
        if split_count >= n_splits:
            break
        val_end = min(split_point + step, n_periods)
        train_p = periods[:split_point]
        val_p = periods[split_point:val_end]
        if not val_p:
            break

        train_keys = set((p["tahun"], p["bulan"]) for p in train_p)
        val_keys = set((p["tahun"], p["bulan"]) for p in val_p)

        df_fold_train = df_train_raw[df_train_raw.apply(lambda r: (r["tahun"], r["bulan"]) in train_keys, axis=1)].copy()
        df_fold_val = df_train_raw[df_train_raw.apply(lambda r: (r["tahun"], r["bulan"]) in val_keys, axis=1)].copy()

        if df_fold_train.empty or df_fold_val.empty:
            continue

        y_fold_tr = df_fold_train["target_label"].tolist()
        y_fold_val = df_fold_val["target_label"].tolist()

        if len(set(y_fold_tr)) < 2:
            continue

        # Diskretisasi kuantil HANYA dari fold train
        disc = FeatureDiscretizer()
        disc.fit(df_fold_train, numeric_in_features)
        df_ftr_disc = disc.transform(df_fold_train, numeric_in_features)
        df_fval_disc = disc.transform(df_fold_val, numeric_in_features)

        # Encoder kategori
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(df_ftr_disc[selected_features].astype(str))
        X_ftr_enc = encode_categorical_matrix(enc, df_ftr_disc, selected_features)
        X_fval_enc = encode_categorical_matrix(enc, df_fval_disc, selected_features)
        min_cats = [len(cats) + 1 for cats in enc.categories_]

        fold_alpha_res = {}
        for cand_alpha in alphas_to_try:
            nb_fold = CategoricalNB(alpha=cand_alpha, min_categories=min_cats)
            nb_fold.fit(X_ftr_enc, y_fold_tr)
            val_preds = nb_fold.predict(X_fval_enc)
            f1_macro = float(f1_score(y_fold_val, val_preds, average="macro", zero_division=0))
            alpha_scores[cand_alpha].append(f1_macro)
            fold_alpha_res[str(cand_alpha)] = round(f1_macro, 4)

        fold_reports.append({
            "fold": split_count + 1,
            "train_period": f"{train_p[0]['tahun']}-{train_p[0]['bulan']:02d} s.d. {train_p[-1]['tahun']}-{train_p[-1]['bulan']:02d}",
            "val_period": f"{val_p[0]['tahun']}-{val_p[0]['bulan']:02d} s.d. {val_p[-1]['tahun']}-{val_p[-1]['bulan']:02d}",
            "train_samples": len(df_fold_train),
            "val_samples": len(df_fold_val),
            "f1_scores_per_alpha": fold_alpha_res
        })
        split_count += 1

    # Pilih alpha terbaik dengan rata-rata F1 makro tertinggi
    best_alpha = alphas_to_try[0]
    best_mean_f1 = -1.0
    for a in alphas_to_try:
        scores = alpha_scores[a]
        mean_s = float(np.mean(scores)) if scores else 0.0
        if mean_s > best_mean_f1:
            best_mean_f1 = mean_s
            best_alpha = a

    best_scores = alpha_scores.get(best_alpha, [0.0])
    summary = {
        "best_alpha": best_alpha,
        "mean_f1": round(float(np.mean(best_scores)), 4) if best_scores else 0.0,
        "std_f1": round(float(np.std(best_scores)), 4) if best_scores else 0.0,
        "folds": fold_reports
    }
    return best_alpha, summary

def extract_conditional_probabilities(
    nb_model: CategoricalNB,
    encoder: OrdinalEncoder,
    selected_features: List[str]
) -> List[Dict[str, Any]]:
    """
    Mengekstrak tabel probabilitas bersyarat P(Kategori | Kelas) untuk setiap atribut.
    Menjawab Rumusan Masalah 3 dan Bagian 7.6 Proposal.
    """
    class_indices = {cls_name: idx for idx, cls_name in enumerate(nb_model.classes_)}
    idx_naik = class_indices.get("Naik")
    idx_stabil = class_indices.get("Stabil")
    idx_turun = class_indices.get("Turun")

    records = []
    for feat_idx, feat_name in enumerate(selected_features):
        categories = list(encoder.categories_[feat_idx])
        log_probs = nb_model.feature_log_prob_[feat_idx]
        for cat_idx, cat_val in enumerate(categories):
            p_n = float(np.exp(log_probs[idx_naik, cat_idx])) if idx_naik is not None else 0.0
            p_s = float(np.exp(log_probs[idx_stabil, cat_idx])) if idx_stabil is not None else 0.0
            p_t = float(np.exp(log_probs[idx_turun, cat_idx])) if idx_turun is not None else 0.0

            records.append({
                "feature": feat_name,
                "category": str(cat_val),
                "p_naik": round(p_n, 4),
                "p_stabil": round(p_s, 4),
                "p_turun": round(p_t, 4)
            })
    return records

def compute_group_ablation(
    df_train_disc: pd.DataFrame,
    df_test_disc: pd.DataFrame,
    selected_features: List[str],
    y_train: List[str],
    y_test: List[str],
    alpha: float,
    full_f1_macro: float
) -> List[Dict[str, Any]]:
    """
    Ablasi kinerja per-kelompok atribut tematik (Kalender, Riwayat Lag, Demografi, Kunjungan, Iklim).
    """
    results = []
    for group_name, group_cols in FEATURE_GROUPS.items():
        present_cols = [c for c in group_cols if c in selected_features]
        if not present_cols:
            continue
        remaining_feats = [c for c in selected_features if c not in present_cols]
        if not remaining_feats:
            continue

        enc_sub = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc_sub.fit(df_train_disc[remaining_feats].astype(str))
        X_tr = encode_categorical_matrix(enc_sub, df_train_disc, remaining_feats)
        X_te = encode_categorical_matrix(enc_sub, df_test_disc, remaining_feats)
        min_cats = [len(cats) + 1 for cats in enc_sub.categories_]

        nb_sub = CategoricalNB(alpha=alpha, min_categories=min_cats)
        nb_sub.fit(X_tr, y_train)
        preds = nb_sub.predict(X_te)
        f1_sub = float(f1_score(y_test, preds, average="macro", zero_division=0))
        delta = full_f1_macro - f1_sub

        results.append({
            "group": group_name,
            "features": present_cols,
            "f1_macro_without": round(f1_sub, 4),
            "delta_f1": round(delta, 4)
        })
    results.sort(key=lambda x: x["delta_f1"], reverse=True)
    return results

def compute_disease_metrics(
    df_test: pd.DataFrame,
    y_test: List[str],
    y_pred: List[str]
) -> List[Dict[str, Any]]:
    """
    Kinerja model Naive Bayes per diagnosis penyakit ICD-10 pada data uji.
    """
    df_eval = df_test.copy()
    df_eval["y_true"] = y_test
    df_eval["y_pred"] = y_pred

    results = []
    for icd, grp in df_eval.groupby("kode_icd10"):
        y_t = grp["y_true"].tolist()
        y_p = grp["y_pred"].tolist()
        acc = float(np.mean([t == p for t, p in zip(y_t, y_p)]))
        f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
        results.append({
            "icd10": icd,
            "test_samples": len(grp),
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4),
            "class_distribution": pd.Series(y_t).value_counts().to_dict()
        })
    results.sort(key=lambda x: x["f1_macro"], reverse=True)
    return results

def compute_threshold_sensitivity(
    df_raw: pd.DataFrame,
    selected_features: List[str],
    test_ratio: float = 0.20
) -> List[Dict[str, Any]]:
    """
    Tabel analisis sensitivitas ambang label (5%, 10%, 15%, 20%).
    Membuktikan secara empiris dasar penetapan parameter ambang.
    """
    threshold_configs = [
        (0.05, 0),
        (0.05, 3),
        (0.10, 0),  # Proposal Murni
        (0.10, 3),  # Default Sistem
        (0.15, 3),
        (0.20, 5),
    ]
    results = []
    for pct, min_c in threshold_configs:
        try:
            df_lbl = generate_labels_for_dataframe(df_raw, pct, min_c)
            df_m = build_feature_matrix(df_lbl).dropna(subset=["target_label"]).copy()
            counts = df_m["target_label"].value_counts().to_dict()
            total = len(df_m)

            df_tr, df_te, _, _ = split_train_test_by_time(df_m, test_ratio)
            num_feats = [f for f in selected_features if f in NUMERIC_LAG_FEATURES]
            disc = FeatureDiscretizer()
            disc.fit(df_tr, num_feats)
            df_tr_d = disc.transform(df_tr, num_feats)
            df_te_d = disc.transform(df_te, num_feats)

            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            enc.fit(df_tr_d[selected_features].astype(str))
            X_tr = encode_categorical_matrix(enc, df_tr_d, selected_features)
            X_te = encode_categorical_matrix(enc, df_te_d, selected_features)
            min_cats = [len(cats) + 1 for cats in enc.categories_]

            nb = CategoricalNB(alpha=1.0, min_categories=min_cats)
            nb.fit(X_tr, df_tr["target_label"].tolist())
            preds = nb.predict(X_te)
            f1 = float(f1_score(df_te["target_label"].tolist(), preds, average="macro", zero_division=0))
            acc = float(np.mean([p == t for p, t in zip(preds, df_te["target_label"].tolist())]))

            results.append({
                "threshold_pct": pct,
                "min_abs_change": min_c,
                "label_formula": "Proposal murni (persen saja)" if min_c == 0 else f"Ambang {int(pct*100)}% & min {min_c} kasus",
                "total_samples": total,
                "naik": counts.get("Naik", 0),
                "stabil": counts.get("Stabil", 0),
                "turun": counts.get("Turun", 0),
                "pct_naik": round((counts.get("Naik", 0) / total) * 100, 1) if total else 0,
                "pct_stabil": round((counts.get("Stabil", 0) / total) * 100, 1) if total else 0,
                "pct_turun": round((counts.get("Turun", 0) / total) * 100, 1) if total else 0,
                "f1_macro_nb": round(f1, 4),
                "accuracy_nb": round(acc, 4)
            })
        except Exception:
            pass
    return results

def run_walk_forward_evaluation(
    df_valid: pd.DataFrame,
    selected_features: List[str],
    numeric_in_features: List[str],
    alpha: float,
    window_test_months: int = 6
) -> Dict[str, Any]:
    """
    Evaluasi Walk-Forward Berulang pada beberapa titik waktu uji
    (mencakup periode musim hujan Jan-Mar dan kemarau Mei-Okt).
    """
    periods = df_valid[["tahun", "bulan"]].drop_duplicates().sort_values(by=["tahun", "bulan"]).to_dict("records")
    total_p = len(periods)
    if total_p < window_test_months + 6:
        return {"splits": [], "mean_f1": 0.0, "std_f1": 0.0}

    cutoffs = [
        total_p - (window_test_months * 3),
        total_p - (window_test_months * 2),
        total_p - window_test_months
    ]
    splits = []
    f1s = []

    for i, cut in enumerate(cutoffs):
        if cut < 6:
            continue
        tr_p = periods[:cut]
        te_p = periods[cut:cut + window_test_months]
        if not te_p:
            continue

        tr_keys = set((p["tahun"], p["bulan"]) for p in tr_p)
        te_keys = set((p["tahun"], p["bulan"]) for p in te_p)

        df_tr = df_valid[df_valid.apply(lambda r: (r["tahun"], r["bulan"]) in tr_keys, axis=1)].copy()
        df_te = df_valid[df_valid.apply(lambda r: (r["tahun"], r["bulan"]) in te_keys, axis=1)].copy()

        disc = FeatureDiscretizer()
        disc.fit(df_tr, numeric_in_features)
        df_tr_d = disc.transform(df_tr, numeric_in_features)
        df_te_d = disc.transform(df_te, numeric_in_features)

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(df_tr_d[selected_features].astype(str))
        X_tr = encode_categorical_matrix(enc, df_tr_d, selected_features)
        X_te = encode_categorical_matrix(enc, df_te_d, selected_features)
        min_cats = [len(cats) + 1 for cats in enc.categories_]

        nb = CategoricalNB(alpha=alpha, min_categories=min_cats)
        nb.fit(X_tr, df_tr["target_label"].tolist())
        preds = nb.predict(X_te)
        f1 = float(f1_score(df_te["target_label"].tolist(), preds, average="macro", zero_division=0))
        acc = float(np.mean([p == t for p, t in zip(preds, df_te["target_label"].tolist())]))
        f1s.append(f1)

        splits.append({
            "split_index": i + 1,
            "train_period": f"{tr_p[0]['tahun']}-{tr_p[0]['bulan']:02d} s.d. {tr_p[-1]['tahun']}-{tr_p[-1]['bulan']:02d}",
            "test_period": f"{te_p[0]['tahun']}-{te_p[0]['bulan']:02d} s.d. {te_p[-1]['tahun']}-{te_p[-1]['bulan']:02d}",
            "test_samples": len(df_te),
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4)
        })

    return {
        "splits": splits,
        "mean_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "std_f1": round(float(np.std(f1s)), 4) if f1s else 0.0
    }

def train_and_evaluate_model(
    dataset_id: int,
    user_id: Optional[int],
    threshold_pct: float,
    min_abs_change: int,
    alpha: float,
    test_ratio: float,
    selected_features: Optional[List[str]],
    top_n_diseases: Optional[int] = None,
    db: Session = None
) -> ModelRun:
    """
    Menjalankan proses pelatihan dan evaluasi model komprehensif sesuai PRD dan Proposal.
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

    # 2. Tentukan fitur yang dipakai (default subset representatif)
    if not selected_features:
        selected_features = [f for f in ALL_AVAILABLE_FEATURES if f != "musim"]

    # 3. Bentuk label tren
    df_labeled = generate_labels_for_dataframe(df_raw, threshold_pct, min_abs_change)

    # 4. Bangun matriks fitur lag (pencegahan kebocoran)
    df_feat_matrix = build_feature_matrix(df_labeled)

    # Baris valid (buang bulan pertama tiap penyakit yang belum punya label target)
    df_valid = df_feat_matrix.dropna(subset=["target_label"]).copy()

    # 5. Split berdasarkan waktu
    df_train, df_test, train_period_str, test_period_str = split_train_test_by_time(df_valid, test_ratio)

    # 5.1 Opsi Top-N Penyakit (dihitung HANYA dari data latih agar tidak ada kebocoran data)
    top_diseases_list = None
    if top_n_diseases is not None and top_n_diseases > 0:
        disease_totals = df_train.groupby("kode_icd10")["jumlah_kasus"].sum().sort_values(ascending=False)
        top_diseases_list = disease_totals.head(top_n_diseases).index.tolist()
        df_train = df_train[df_train["kode_icd10"].isin(top_diseases_list)].copy()
        df_test = df_test[df_test["kode_icd10"].isin(top_diseases_list)].copy()

    # 6. Diskretisasi kuantil hanya di data latih
    discretizer = FeatureDiscretizer()
    numeric_in_features = [f for f in selected_features if f in NUMERIC_LAG_FEATURES]
    discretizer.fit(df_train, numeric_in_features)

    df_train_disc = discretizer.transform(df_train, numeric_in_features)
    df_test_disc = discretizer.transform(df_test, numeric_in_features)

    y_train = df_train["target_label"].tolist()
    y_test = df_test["target_label"].tolist()

    # 7. Encoding Ordinal untuk model ML (dengan unseen category mapping)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(df_train_disc[selected_features].astype(str))

    X_train_enc = encode_categorical_matrix(encoder, df_train_disc, selected_features)
    X_test_enc = encode_categorical_matrix(encoder, df_test_disc, selected_features)

    # 8. Tuning hyperparameter alpha untuk Naive Bayes (Expanding Window Deret Waktu)
    best_alpha, cv_summary = run_period_expanding_window_cv(
        df_train_raw=df_train,
        selected_features=selected_features,
        numeric_in_features=numeric_in_features,
        alphas_to_try=[0.1, 0.5, 1.0, 2.0],
        n_splits=4
    )
    # Jika user eksplisit menentukan alpha custom dan tidak meminta tuning, gunakan itu jika sesuai
    if alpha not in [0.1, 0.5, 1.0, 2.0]:
        best_alpha = alpha

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

    # 12. Drop-one-feature (Pengaruh Atribut Individu)
    full_f1_macro = metrics_data[AlgorithmType.NAIVE_BAYES]["f1_macro"]
    feature_influences: List[Dict[str, Any]] = []

    if len(selected_features) > 1:
        for feat in selected_features:
            sub_feats = [f for f in selected_features if f != feat]
            enc_sub = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            enc_sub.fit(df_train_disc[sub_feats].astype(str))

            X_tr_sub = encode_categorical_matrix(enc_sub, df_train_disc, sub_feats)
            X_te_sub = encode_categorical_matrix(enc_sub, df_test_disc, sub_feats)
            sub_min_cats = [len(cats) + 1 for cats in enc_sub.categories_]

            nb_sub = CategoricalNB(alpha=best_alpha, min_categories=sub_min_cats)
            nb_sub.fit(X_tr_sub, y_train)
            sub_preds = nb_sub.predict(X_te_sub)

            f1_sub = float(f1_score(y_test, sub_preds, average="macro", zero_division=0))
            delta_f1 = full_f1_macro - f1_sub

            feature_influences.append({
                "feature": feat,
                "f1_macro_without": round(f1_sub, 4),
                "delta_f1": round(delta_f1, 4)
            })

        feature_influences.sort(key=lambda x: x["delta_f1"], reverse=True)

    # 13. Analisis Lanjutan Sesuai Proposal
    # A. Probabilitas Bersyarat
    conditional_probs = extract_conditional_probabilities(nb_model, encoder, selected_features)

    # B. Ablasi Kelompok Atribut
    group_ablation = compute_group_ablation(
        df_train_disc, df_test_disc, selected_features, y_train, y_test, best_alpha, full_f1_macro
    )

    # C. Kinerja per Penyakit
    disease_perf = compute_disease_metrics(df_test, y_test, y_pred_nb)

    # D. Sensitivitas Ambang
    threshold_sens = compute_threshold_sensitivity(df_raw, selected_features, test_ratio)

    # E. Evaluasi Walk-Forward Berulang
    walk_forward_res = run_walk_forward_evaluation(df_valid, selected_features, numeric_in_features, best_alpha)

    # F. Laporan Preprocessing Terdokumentasi
    prep_report = {
        "raw_rows": len(df_raw),
        "labeled_rows": len(df_labeled),
        "valid_rows_after_lag": len(df_valid),
        "dropped_lag_rows": len(df_raw) - len(df_valid),
        "train_rows": len(df_train),
        "test_rows": len(df_test),
        "top_n_applied": top_n_diseases,
        "top_diseases": top_diseases_list,
        "quantile_cutoffs": discretizer.get_cutoffs()
    }

    # 14. Simpan Run ke Database
    dist_counts = pd.Series(y_train + y_test).value_counts().to_dict()

    model_run = ModelRun(
        dataset_id=dataset_id,
        created_by=user_id,
        threshold_pct=threshold_pct,
        min_abs_change=min_abs_change,
        alpha=best_alpha,
        test_ratio=test_ratio,
        top_n_diseases=top_n_diseases,
        features_json=json.dumps(selected_features),
        train_rows=len(df_train),
        test_rows=len(df_test),
        train_period=train_period_str,
        test_period=test_period_str,
        class_distribution_json=json.dumps(dist_counts),
        cv_scores_json=json.dumps(cv_summary),
        group_ablation_json=json.dumps(group_ablation),
        disease_metrics_json=json.dumps(disease_perf),
        preprocessing_report_json=json.dumps(prep_report),
        threshold_sensitivity_json=json.dumps(threshold_sens),
        walk_forward_json=json.dumps(walk_forward_res),
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

    # 15. Simpan Artifak Model ke disk
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
        "min_abs_change": min_abs_change,
        "top_n_diseases": top_n_diseases,
        "top_diseases_list": top_diseases_list,
        "conditional_probabilities": conditional_probs,
        "cv_summary": cv_summary,
        "group_ablation": group_ablation,
        "preprocessing_report": prep_report,
        "threshold_sensitivity": threshold_sens,
        "walk_forward": walk_forward_res
    }, model_filepath)

    model_run.model_path = model_filepath
    db.commit()
    db.refresh(model_run)

    return model_run
