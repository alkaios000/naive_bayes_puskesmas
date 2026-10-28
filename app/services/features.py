from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

ALL_AVAILABLE_FEATURES = [
    "kode_icd10",
    "bulan_target",
    "musim",
    "label_lag1",
    "label_lag2",
    "proporsi_laki_lag1",
    "proporsi_umur_lt1_lag1",
    "proporsi_umur_1_5_lag1",
    "proporsi_umur_6_14_lag1",
    "proporsi_umur_15_59_lag1",
    "proporsi_umur_gte60_lag1",
    "rasio_kasus_kunjungan_lag1",
    "total_kunjungan_lag1",
    "curah_hujan_lag1",
    "suhu_lag1"
]

NUMERIC_LAG_FEATURES = [
    "proporsi_laki_lag1",
    "proporsi_umur_lt1_lag1",
    "proporsi_umur_1_5_lag1",
    "proporsi_umur_6_14_lag1",
    "proporsi_umur_15_59_lag1",
    "proporsi_umur_gte60_lag1",
    "rasio_kasus_kunjungan_lag1",
    "total_kunjungan_lag1",
    "curah_hujan_lag1",
    "suhu_lag1"
]

FEATURE_GROUPS = {
    "Kalender": ["bulan_target", "musim"],
    "Riwayat Tren (Lag)": ["label_lag1", "label_lag2"],
    "Demografi (Gender & Umur)": [
        "proporsi_laki_lag1",
        "proporsi_umur_lt1_lag1",
        "proporsi_umur_1_5_lag1",
        "proporsi_umur_6_14_lag1",
        "proporsi_umur_15_59_lag1",
        "proporsi_umur_gte60_lag1",
    ],
    "Kunjungan": ["rasio_kasus_kunjungan_lag1", "total_kunjungan_lag1"],
    "Iklim & Cuaca": ["curah_hujan_lag1", "suhu_lag1"],
    "Diagnosis": ["kode_icd10"]
}

def determine_season(month: int) -> str:
    """
    Menentukan musim berdasarkan PRD:
    Hujan: Nov-Apr (11, 12, 1, 2, 3, 4)
    Kemarau: Mei-Okt (5, 6, 7, 8, 9, 10)
    """
    if month in [11, 12, 1, 2, 3, 4]:
        return "hujan"
    return "kemarau"

def build_feature_matrix(df_labeled: pd.DataFrame) -> pd.DataFrame:
    """
    Membangun dataset fitur dengan atribut lag dan variabel kalender (bulan target & musim).
    MENJAMIN TIDAK ADA KEBOCORAN DATA:
    Seluruh variabel prediktor diambil dari bulan t-1 dan t-2, bukan bulan t.
    """
    df_sorted = df_labeled.sort_values(by=["kode_icd10", "tahun", "bulan"]).copy()

    feature_rows = []
    grouped = df_sorted.groupby("kode_icd10")

    for icd, group in grouped:
        rows = group.to_dict("records")
        for i in range(len(rows)):
            curr = rows[i]

            # Fitur bulan target
            target_month = int(curr["bulan"])
            target_year = int(curr["tahun"])
            musim = determine_season(target_month)

            # Lag 1
            if i >= 1:
                lag1 = rows[i - 1]
                lbl1 = lag1.get("label")
                label_lag1 = str(lbl1) if (lbl1 is not None and pd.notna(lbl1)) else "tidak_ada"
                cases_lag1 = max(int(lag1.get("jumlah_kasus", 0) or 0), 1)

                # Proporsi laki-laki
                if lag1.get("kasus_laki") is not None and pd.notna(lag1.get("kasus_laki")):
                    prop_laki = float(lag1["kasus_laki"]) / cases_lag1
                else:
                    prop_laki = None

                # Proporsi umur
                def get_age_prop(col_name: str) -> Optional[float]:
                    val = lag1.get(col_name)
                    if val is not None and pd.notna(val):
                        return float(val) / cases_lag1
                    return None

                prop_lt1 = get_age_prop("kasus_lt1th")
                prop_1_5 = get_age_prop("kasus_1_5th")
                prop_6_14 = get_age_prop("kasus_6_14th")
                prop_15_59 = get_age_prop("kasus_15_59th")
                prop_gte60 = get_age_prop("kasus_gte60th")

                # Kunjungan
                if lag1.get("total_kunjungan") is not None and pd.notna(lag1.get("total_kunjungan")) and int(lag1["total_kunjungan"]) > 0:
                    total_kunjungan_lag1 = float(lag1["total_kunjungan"])
                    rasio_kunjungan = float(lag1.get("jumlah_kasus", 0)) / total_kunjungan_lag1
                else:
                    total_kunjungan_lag1 = None
                    rasio_kunjungan = None

                curah_hujan = float(lag1["curah_hujan_mm"]) if lag1.get("curah_hujan_mm") is not None and pd.notna(lag1.get("curah_hujan_mm")) else None
                suhu = float(lag1["suhu_rata2"]) if lag1.get("suhu_rata2") is not None and pd.notna(lag1.get("suhu_rata2")) else None
            else:
                label_lag1 = "tidak_ada"
                prop_laki = None
                prop_lt1 = None
                prop_1_5 = None
                prop_6_14 = None
                prop_15_59 = None
                prop_gte60 = None
                rasio_kunjungan = None
                total_kunjungan_lag1 = None
                curah_hujan = None
                suhu = None

            # Lag 2
            if i >= 2:
                lag2 = rows[i - 2]
                lbl2 = lag2.get("label")
                label_lag2 = str(lbl2) if (lbl2 is not None and pd.notna(lbl2)) else "tidak_ada"
            else:
                label_lag2 = "tidak_ada"

            feature_rows.append({
                "tahun": target_year,
                "bulan": target_month,
                "kode_icd10": icd,
                "jumlah_kasus": int(curr.get("jumlah_kasus", 0) or 0),
                "bulan_target": str(target_month),
                "musim": musim,
                "label_lag1": label_lag1,
                "label_lag2": label_lag2,
                "proporsi_laki_lag1": prop_laki,
                "proporsi_umur_lt1_lag1": prop_lt1,
                "proporsi_umur_1_5_lag1": prop_1_5,
                "proporsi_umur_6_14_lag1": prop_6_14,
                "proporsi_umur_15_59_lag1": prop_15_59,
                "proporsi_umur_gte60_lag1": prop_gte60,
                "rasio_kasus_kunjungan_lag1": rasio_kunjungan,
                "total_kunjungan_lag1": total_kunjungan_lag1,
                "curah_hujan_lag1": curah_hujan,
                "suhu_lag1": suhu,
                "target_label": curr["label"]
            })

    return pd.DataFrame(feature_rows)

class FeatureDiscretizer:
    """
    Melakukan diskretisasi atribut numerik menjadi 3 kategori: 'rendah', 'sedang', 'tinggi'
    dan 'tidak_ada' untuk nilai kosong.
    KUANTIL DIHITUNG HANYA DARI DATA LATIH untuk mencegah kebocoran data (PRD Bagian 5.2).
    """
    def __init__(self):
        self.bins: Dict[str, Tuple[float, float]] = {}

    def fit(self, df_train: pd.DataFrame, numeric_features: List[str]):
        self.bins = {}
        for feat in numeric_features:
            if feat in df_train.columns:
                valid_vals = df_train[feat].dropna().values
                if len(valid_vals) > 0:
                    q33 = np.percentile(valid_vals, 33.33)
                    q66 = np.percentile(valid_vals, 66.67)
                    self.bins[feat] = (float(q33), float(q66))

    def transform(self, df: pd.DataFrame, numeric_features: List[str]) -> pd.DataFrame:
        df_out = df.copy()
        for feat in numeric_features:
            if feat in df.columns:
                if feat not in self.bins:
                    df_out[feat] = "tidak_ada"
                    continue
                q33, q66 = self.bins[feat]
                cats = []
                for val in df[feat]:
                    if val is None or pd.isna(val):
                        cats.append("tidak_ada")
                    elif val <= q33:
                        cats.append("rendah")
                    elif val <= q66:
                        cats.append("sedang")
                    else:
                        cats.append("tinggi")
                df_out[feat] = cats
        return df_out

    def get_cutoffs(self) -> Dict[str, Dict[str, float]]:
        return {
            feat: {"q33": round(val[0], 4), "q66": round(val[1], 4)}
            for feat, val in self.bins.items()
        }

