import io
import re
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.orm import Session

from app.models import Dataset, Disease, MonthlyCase, User
from app.config import settings

PII_KEYWORDS = ["nama_pasien", "nik", "alamat", "no_rm", "norm", "nomor_rm", "orang_tua", "nama_ibu", "nama_ayah"]

COLUMN_MAPPING = {
    "kasus_<1th": "kasus_lt1th",
    "kasus_1-5th": "kasus_1_5th",
    "kasus_6-14th": "kasus_6_14th",
    "kasus_15-59th": "kasus_15_59th",
    "kasus_60th+": "kasus_gte60th",
}

MANDATORY_COLUMNS = ["tahun", "bulan", "kode_icd10", "nama_penyakit", "jumlah_kasus"]

def check_pii_violation(columns: List[str]) -> Optional[str]:
    """
    Periksa apakah nama kolom memuat indikasi data identitas pasien (PII)
    """
    for col in columns:
        cleaned = re.sub(r"[^a-zA-Z0-9_]", "", str(col).lower().strip())
        if cleaned in ["nama", "nik", "alamat", "no_rm", "norm", "orang_tua"]:
            return col
        for keyword in PII_KEYWORDS:
            if keyword in cleaned:
                return col
    return None

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalisasi nama kolom dataset agar konsisten
    """
    clean_cols = {}
    for col in df.columns:
        c_str = str(col).strip()
        mapped = COLUMN_MAPPING.get(c_str, c_str)
        clean_cols[col] = mapped
    return df.rename(columns=clean_cols)

def validate_and_parse_file(content: bytes, filename: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]], bool]:
    """
    Membaca dan memvalidasi berkas CSV atau XLSX
    Mengembalikan DataFrame yang telah divalidasi, daftar error, dan flag is_dummy
    """
    errors: List[Dict[str, Any]] = []

    # 1. Parsing file
    try:
        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(content), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(content), encoding="latin1")
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            errors.append({"row": 0, "type": "FORMAT_ERROR", "message": "Format berkas harus .csv atau .xlsx"})
            return pd.DataFrame(), errors, False
    except Exception as e:
        errors.append({"row": 0, "type": "PARSE_ERROR", "message": f"Gagal membaca berkas: {str(e)}"})
        return pd.DataFrame(), errors, False

    # 2. Cek kolom PII
    pii_found = check_pii_violation(list(df.columns))
    if pii_found:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": {
                    "code": "PII_VIOLATION",
                    "message": f"Berkas ditolak: Terdeteksi kolom identitas pasien '{pii_found}'. Dilarang mengunggah data identitas.",
                    "details": [{"row": 1, "type": "PII_VIOLATION", "message": f"Kolom identitas '{pii_found}' dilarang."}]
                }
            }
        )

    # 3. Normalisasi kolom
    df = normalize_column_names(df)

    # 4. Validasi kolom wajib
    missing_cols = [c for c in MANDATORY_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append({
            "row": 1,
            "type": "MISSING_COLUMNS",
            "message": f"Kolom wajib tidak lengkap: {', '.join(missing_cols)}"
        })
        return df, errors, False

    # Cek flag dummy
    is_dummy = False
    if "catatan" in df.columns:
        if df["catatan"].astype(str).str.contains("DATA DUMMY", case=False, na=False).any():
            is_dummy = True

    # 5. Validasi tipe data dan batasan per baris
    seen_keys = set()
    total_cases_per_period: Dict[Tuple[int, int], int] = {}
    
    # Hitung total kasus per periode untuk validasi total_kunjungan
    for idx, row in df.iterrows():
        try:
            th = int(row["tahun"])
            bl = int(row["bulan"])
            ks = int(row["jumlah_kasus"])
            key = (th, bl)
            total_cases_per_period[key] = total_cases_per_period.get(key, 0) + ks
        except Exception:
            pass

    for idx, row in df.iterrows():
        row_num = idx + 2  # baris di excel/csv (1-indexed + header)

        # Tahun & Bulan
        try:
            tahun = int(row["tahun"])
            if not (2015 <= tahun <= 2100):
                errors.append({"row": row_num, "type": "INVALID_YEAR", "message": f"Tahun {tahun} di luar rentang 2015-2100"})
        except Exception:
            errors.append({"row": row_num, "type": "INVALID_TYPE", "message": "Kolom tahun harus berupa angka bulat"})
            tahun = 0

        try:
            bulan = int(row["bulan"])
            if not (1 <= bulan <= 12):
                errors.append({"row": row_num, "type": "INVALID_MONTH", "message": f"Bulan {bulan} di luar rentang 1-12"})
        except Exception:
            errors.append({"row": row_num, "type": "INVALID_TYPE", "message": "Kolom bulan harus berupa angka bulat"})
            bulan = 0

        # ICD10
        icd10 = str(row["kode_icd10"]).strip().upper()
        if not icd10:
            errors.append({"row": row_num, "type": "EMPTY_FIELD", "message": "kode_icd10 tidak boleh kosong"})

        # Duplikasi
        key = (tahun, bulan, icd10)
        if key in seen_keys:
            errors.append({"row": row_num, "type": "DUPLICATE_ROW", "message": f"Duplikat data untuk periode {tahun}-{bulan:02d} dan diagnosis {icd10}"})
        else:
            seen_keys.add(key)

        # Jumlah kasus
        try:
            cases = int(row["jumlah_kasus"])
            if cases < 0:
                errors.append({"row": row_num, "type": "NEGATIVE_VALUE", "message": "jumlah_kasus tidak boleh bernilai negatif"})
        except Exception:
            errors.append({"row": row_num, "type": "INVALID_TYPE", "message": "jumlah_kasus harus angka bulat"})
            cases = 0

        # Validasi jenis kelamin jika ada
        has_male = "kasus_laki" in df.columns and pd.notna(row["kasus_laki"])
        has_female = "kasus_perempuan" in df.columns and pd.notna(row["kasus_perempuan"])
        if has_male and has_female:
            try:
                m = int(row["kasus_laki"])
                f = int(row["kasus_perempuan"])
                if m < 0 or f < 0:
                    errors.append({"row": row_num, "type": "NEGATIVE_VALUE", "message": "kasus_laki/kasus_perempuan tidak boleh negatif"})
                elif m + f != cases:
                    errors.append({"row": row_num, "type": "SUM_MISMATCH", "message": f"kasus_laki ({m}) + kasus_perempuan ({f}) != jumlah_kasus ({cases})"})
            except Exception:
                errors.append({"row": row_num, "type": "INVALID_TYPE", "message": "kasus_laki/perempuan harus berupa angka bulat"})

        # Validasi umur jika ada
        age_cols = ["kasus_lt1th", "kasus_1_5th", "kasus_6_14th", "kasus_15_59th", "kasus_gte60th"]
        if all(c in df.columns and pd.notna(row[c]) for c in age_cols):
            try:
                ages = [int(row[c]) for c in age_cols]
                if any(a < 0 for a in ages):
                    errors.append({"row": row_num, "type": "NEGATIVE_VALUE", "message": "Kelompok umur tidak boleh bernilai negatif"})
                elif sum(ages) != cases:
                    errors.append({"row": row_num, "type": "SUM_MISMATCH", "message": f"Total umur ({sum(ages)}) != jumlah_kasus ({cases})"})
            except Exception:
                errors.append({"row": row_num, "type": "INVALID_TYPE", "message": "Kolom umur harus berupa angka bulat"})

        # Validasi total kunjungan
        if "total_kunjungan" in df.columns and pd.notna(row["total_kunjungan"]):
            try:
                tk = int(row["total_kunjungan"])
                period_cases = total_cases_per_period.get((tahun, bulan), 0)
                if tk < period_cases:
                    errors.append({"row": row_num, "type": "TOTAL_VISITS_INVALID", "message": f"total_kunjungan ({tk}) < total kasus semua penyakit pada bulan tersebut ({period_cases})"})
            except Exception:
                errors.append({"row": row_num, "type": "INVALID_TYPE", "message": "total_kunjungan harus angka bulat"})

        # Validasi cuaca jika ada
        if "curah_hujan_mm" in df.columns and pd.notna(row["curah_hujan_mm"]):
            try:
                ch = float(row["curah_hujan_mm"])
                if ch < 0:
                    errors.append({"row": row_num, "type": "NEGATIVE_VALUE", "message": "curah_hujan_mm tidak boleh negatif"})
            except Exception:
                errors.append({"row": row_num, "type": "INVALID_TYPE", "message": "curah_hujan_mm harus berupa angka desimal"})

        if "suhu_rata2" in df.columns and pd.notna(row["suhu_rata2"]):
            try:
                suhu = float(row["suhu_rata2"])
                if not (15.0 <= suhu <= 40.0):
                    errors.append({"row": row_num, "type": "TEMPERATURE_OUT_OF_RANGE", "message": f"suhu_rata2 ({suhu}) di luar rentang wajar 15-40 C"})
            except Exception:
                errors.append({"row": row_num, "type": "INVALID_TYPE", "message": "suhu_rata2 harus berupa angka desimal"})

    # 6. Validasi kontinuitas waktu (tidak boleh ada bulan bolong per penyakit)
    diseases_grouped = df.groupby("kode_icd10")
    for icd10, group in diseases_grouped:
        sorted_group = group.sort_values(by=["tahun", "bulan"])
        prev_year = None
        prev_month = None
        for idx, r in sorted_group.iterrows():
            curr_year = int(r["tahun"])
            curr_month = int(r["bulan"])
            if prev_year is not None:
                expected_year = prev_year + (1 if prev_month == 12 else 0)
                expected_month = 1 if prev_month == 12 else prev_month + 1
                if (curr_year, curr_month) != (expected_year, expected_month):
                    errors.append({
                        "row": idx + 2,
                        "type": "TIMELINE_GAP",
                        "message": f"Terdapat celah bulan pada penyakit {icd10}: setelah {prev_year}-{prev_month:02d} seharusnya {expected_year}-{expected_month:02d}, ditemukan {curr_year}-{curr_month:02d}"
                    })
            prev_year = curr_year
            prev_month = curr_month

    return df, errors, is_dummy

def save_dataset_to_db(df: pd.DataFrame, filename: str, is_dummy: bool, uploader_id: Optional[int], db: Session) -> Dataset:
    """
    Menyimpan dataset yang valid ke dalam database
    """
    # 1. Simpan/Update Master Penyakit (Diseases)
    unique_diseases = df[["kode_icd10", "nama_penyakit"]].drop_duplicates()
    disease_map: Dict[str, int] = {}
    for _, row in unique_diseases.iterrows():
        code = str(row["kode_icd10"]).strip().upper()
        name = str(row["nama_penyakit"]).strip()
        disease_obj = db.query(Disease).filter(Disease.icd10 == code).first()
        if not disease_obj:
            disease_obj = Disease(icd10=code, name=name)
            db.add(disease_obj)
            db.flush()
        else:
            if disease_obj.name != name:
                disease_obj.name = name
                db.flush()
        disease_map[code] = disease_obj.id

    # 2. Hitung statistik rentang periode
    df_sorted = df.sort_values(by=["tahun", "bulan"])
    first_row = df_sorted.iloc[0]
    last_row = df_sorted.iloc[-1]
    start_period = f"{int(first_row['tahun'])}--{int(first_row['bulan']):02d}".replace("--", "-")
    end_period = f"{int(last_row['tahun'])}--{int(last_row['bulan']):02d}".replace("--", "-")
    disease_count = len(unique_diseases)

    # 3. Buat entri Dataset
    dataset = Dataset(
        name=f"Dataset Rekap {start_period} s.d. {end_period}",
        original_filename=filename,
        is_dummy=is_dummy,
        row_count=len(df),
        start_period=start_period,
        end_period=end_period,
        disease_count=disease_count,
        uploaded_by=uploader_id
    )
    db.add(dataset)
    db.flush()

    # 4. Simpan MonthlyCase
    monthly_cases_to_add = []
    for _, row in df.iterrows():
        icd = str(row["kode_icd10"]).strip().upper()
        d_id = disease_map[icd]
        mc = MonthlyCase(
            dataset_id=dataset.id,
            disease_id=d_id,
            year=int(row["tahun"]),
            month=int(row["bulan"]),
            cases=int(row["jumlah_kasus"]),
            male=int(row["kasus_laki"]) if "kasus_laki" in row and pd.notna(row["kasus_laki"]) else None,
            female=int(row["kasus_perempuan"]) if "kasus_perempuan" in row and pd.notna(row["kasus_perempuan"]) else None,
            age_lt1=int(row["kasus_lt1th"]) if "kasus_lt1th" in row and pd.notna(row["kasus_lt1th"]) else None,
            age_1_5=int(row["kasus_1_5th"]) if "kasus_1_5th" in row and pd.notna(row["kasus_1_5th"]) else None,
            age_6_14=int(row["kasus_6_14th"]) if "kasus_6_14th" in row and pd.notna(row["kasus_6_14th"]) else None,
            age_15_59=int(row["kasus_15_59th"]) if "kasus_15_59th" in row and pd.notna(row["kasus_15_59th"]) else None,
            age_gte60=int(row["kasus_gte60th"]) if "kasus_gte60th" in row and pd.notna(row["kasus_gte60th"]) else None,
            total_visits=int(row["total_kunjungan"]) if "total_kunjungan" in row and pd.notna(row["total_kunjungan"]) else None,
            rainfall_mm=float(row["curah_hujan_mm"]) if "curah_hujan_mm" in row and pd.notna(row["curah_hujan_mm"]) else None,
            temp_avg=float(row["suhu_rata2"]) if "suhu_rata2" in row and pd.notna(row["suhu_rata2"]) else None,
        )
        monthly_cases_to_add.append(mc)

    db.bulk_save_objects(monthly_cases_to_add)
    db.commit()
    db.refresh(dataset)
    return dataset
