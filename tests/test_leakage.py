import pytest
import pandas as pd
from app.services.features import build_feature_matrix
from app.services.training import split_train_test_by_time
from app.services.labeling import generate_labels_for_dataframe

def test_leakage_prevention_on_target_month():
    """
    Kriteria Penerimaan 7:
    Membuktikan tidak ada kebocoran data.
    Mengubah nilai jumlah_kasus pada bulan target t TIDAK boleh mengubah nilai atribut apa pun
    yang digunakan untuk memprediksi bulan t tersebut.
    """
    # Data dasar 3 bulan berturut-turut untuk diagnosis J00
    base_data = [
        {"tahun": 2024, "bulan": 1, "kode_icd10": "J00", "nama_penyakit": "Cold", "jumlah_kasus": 100, "kasus_laki": 50, "total_kunjungan": 1000},
        {"tahun": 2024, "bulan": 2, "kode_icd10": "J00", "nama_penyakit": "Cold", "jumlah_kasus": 120, "kasus_laki": 60, "total_kunjungan": 1000},
        {"tahun": 2024, "bulan": 3, "kode_icd10": "J00", "nama_penyakit": "Cold", "jumlah_kasus": 130, "kasus_laki": 65, "total_kunjungan": 1000},
    ]

    # Data modifikasi: ubah jumlah_kasus bulan Maret 2024 dari 130 menjadi 999999
    modified_data = [
        {"tahun": 2024, "bulan": 1, "kode_icd10": "J00", "nama_penyakit": "Cold", "jumlah_kasus": 100, "kasus_laki": 50, "total_kunjungan": 1000},
        {"tahun": 2024, "bulan": 2, "kode_icd10": "J00", "nama_penyakit": "Cold", "jumlah_kasus": 120, "kasus_laki": 60, "total_kunjungan": 1000},
        {"tahun": 2024, "bulan": 3, "kode_icd10": "J00", "nama_penyakit": "Cold", "jumlah_kasus": 999999, "kasus_laki": 65, "total_kunjungan": 1000},
    ]

    df_base_labeled = generate_labels_for_dataframe(pd.DataFrame(base_data))
    df_mod_labeled = generate_labels_for_dataframe(pd.DataFrame(modified_data))

    feat_base = build_feature_matrix(df_base_labeled)
    feat_mod = build_feature_matrix(df_mod_labeled)

    # Ambil baris fitur bulan Maret 2024 (bulan target t = 3)
    row_base_target = feat_base[(feat_base["tahun"] == 2024) & (feat_base["bulan"] == 3)].iloc[0]
    row_mod_target = feat_mod[(feat_mod["tahun"] == 2024) & (feat_mod["bulan"] == 3)].iloc[0]

    # Daftar atribut prediktor
    feature_columns = [
        "bulan_target", "musim", "label_lag1", "label_lag2",
        "proporsi_laki_lag1", "rasio_kasus_kunjungan_lag1"
    ]

    # Seluruh atribut prediktor untuk bulan ke-3 HARUS IDENTIK 100%
    for col in feature_columns:
        assert row_base_target[col] == row_mod_target[col], f"Kebocoran terdeteksi pada atribut {col}!"

def test_time_based_split_chronological_order():
    """
    Kriteria Penerimaan 8:
    Split data berurutan waktu: semua bulan uji lebih baru daripada semua bulan latih.
    """
    records = []
    # 24 bulan dari Jan 2024 sampai Des 2025
    for year in [2024, 2025]:
        for month in range(1, 13):
            records.append({
                "tahun": year,
                "bulan": month,
                "kode_icd10": "J00",
                "nama_penyakit": "Cold",
                "jumlah_kasus": 100,
                "target_label": "Stabil"
            })

    df = pd.DataFrame(records)
    df_train, df_test, tr_str, te_str = split_train_test_by_time(df, test_ratio=0.20)

    # Ambil periode latih maksimum
    train_max_year = df_train["tahun"].max()
    train_max_month = df_train[df_train["tahun"] == train_max_year]["bulan"].max()

    # Ambil periode uji minimum
    test_min_year = df_test["tahun"].min()
    test_min_month = df_test[df_test["tahun"] == test_min_year]["bulan"].min()

    # Pastikan periode uji paling awal lebih baru dari periode latih paling akhir
    assert (test_min_year > train_max_year) or (test_min_year == train_max_year and test_min_month > train_max_month)
