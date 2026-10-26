import pytest
from fastapi import HTTPException
from app.services.ingest import validate_and_parse_file

def test_pii_detection_rejects_upload():
    """
    Kriteria Penerimaan 3:
    Unggahan file berkolom 'nama' atau 'nik' ditolak dengan 422 dan pesan jelas.
    """
    csv_pii = """tahun,bulan,kode_icd10,nama_penyakit,jumlah_kasus,nik,nama_pasien
2024,1,J00,Cold,100,3317012345678901,Budi Santoso
"""
    with pytest.raises(HTTPException) as excinfo:
        validate_and_parse_file(csv_pii.encode("utf-8"), "dataset_pii.csv")
    assert excinfo.value.status_code == 422
    assert "identitas pasien" in str(excinfo.value.detail)

def test_timeline_gap_reporting():
    """
    Kriteria Penerimaan 4:
    Bulan bolong dilaporkan sebagai kesalahan.
    """
    # Januari 2024 lalu langsung Maret 2024 (Februari bolong)
    csv_gap = """tahun,bulan,kode_icd10,nama_penyakit,jumlah_kasus
2024,1,J00,Cold,100
2024,3,J00,Cold,120
"""
    df, errors, _ = validate_and_parse_file(csv_gap.encode("utf-8"), "dataset_gap.csv")
    assert any(err["type"] == "TIMELINE_GAP" for err in errors)

def test_gender_sum_mismatch():
    """
    Kriteria Penerimaan 4:
    Jumlah kasus_laki + kasus_perempuan != jumlah_kasus dilaporkan sebagai error.
    """
    csv_mismatch = """tahun,bulan,kode_icd10,nama_penyakit,jumlah_kasus,kasus_laki,kasus_perempuan
2024,1,J00,Cold,100,50,40
"""
    df, errors, _ = validate_and_parse_file(csv_mismatch.encode("utf-8"), "mismatch.csv")
    assert any(err["type"] == "SUM_MISMATCH" for err in errors)
