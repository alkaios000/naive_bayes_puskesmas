import io
import csv
import json
from fastapi import APIRouter, Depends, HTTPException, status, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Prediction, Disease, ModelMetric, ModelRun, User
from app.security import require_viewer_or_admin

router = APIRouter(prefix="/api/export", tags=["export"])

@router.get("/predictions.csv")
def export_predictions_csv(
    model_run_id: int = Query(...),
    current_user: User = Depends(require_viewer_or_admin),
    db: Session = Depends(get_db)
):
    """
    Ekspor hasil prediksi tren bulan berikutnya ke format CSV
    """
    predictions = (
        db.query(Prediction, Disease)
        .join(Disease, Prediction.disease_id == Disease.id)
        .filter(Prediction.model_run_id == model_run_id)
        .order_by(Disease.icd10)
        .all()
    )

    if not predictions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "NOT_FOUND", "message": "Data prediksi tidak ditemukan untuk model ini", "details": []}}
        )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "tahun_target", "bulan_target", "kode_icd10", "nama_penyakit", 
        "prediksi_tren", "probabilitas_naik", "probabilitas_stabil", "probabilitas_turun"
    ])

    for p, d in predictions:
        writer.writerow([
            p.target_year, p.target_month, d.icd10, d.name,
            p.predicted_class.value, p.p_naik, p.p_stabil, p.p_turun
        ])

    output.seek(0)
    filename = f"prediksi_tren_model_{model_run_id}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@router.get("/metrics.csv")
def export_metrics_csv(
    model_run_id: int = Query(...),
    current_user: User = Depends(require_viewer_or_admin),
    db: Session = Depends(get_db)
):
    """
    Ekspor metrik perbandingan algoritma ke CSV
    """
    metrics = db.query(ModelMetric).filter(ModelMetric.model_run_id == model_run_id).all()
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "NOT_FOUND", "message": "Metrik model tidak ditemukan", "details": []}}
        )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "algoritma", "akurasi", "f1_macro", 
        "precision_naik", "precision_stabil", "precision_turun",
        "recall_naik", "recall_stabil", "recall_turun",
        "f1_naik", "f1_stabil", "f1_turun"
    ])

    for m in metrics:
        prec = json.loads(m.precision_json)
        rec = json.loads(m.recall_json)
        f1 = json.loads(m.f1_json)
        writer.writerow([
            m.algorithm.value, m.accuracy, m.f1_macro,
            prec.get("Naik", 0), prec.get("Stabil", 0), prec.get("Turun", 0),
            rec.get("Naik", 0), rec.get("Stabil", 0), rec.get("Turun", 0),
            f1.get("Naik", 0), f1.get("Stabil", 0), f1.get("Turun", 0),
        ])

    output.seek(0)
    filename = f"metrik_evaluasi_model_{model_run_id}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@router.get("/conditional_probabilities.csv")
def export_conditional_probabilities_csv(
    model_run_id: int = Query(...),
    current_user: User = Depends(require_viewer_or_admin),
    db: Session = Depends(get_db)
):
    """
    Ekspor tabel probabilitas bersyarat P(Kategori | Kelas) ke format CSV
    (Rumusan Masalah 3 & Bab 7.6 Proposal).
    """
    import os
    import joblib
    mr = db.query(ModelRun).filter(ModelRun.id == model_run_id).first()
    if not mr or not mr.model_path or not os.path.exists(mr.model_path):
        raise HTTPException(status_code=404, detail="Model artefak tidak ditemukan")

    artifact = joblib.load(mr.model_path)
    cond_probs = artifact.get("conditional_probabilities", [])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["atribut", "kategori_nilai", "P(Kategori|Naik)", "P(Kategori|Stabil)", "P(Kategori|Turun)"])
    for cp in cond_probs:
        writer.writerow([
            cp["feature"], cp["category"], cp["p_naik"], cp["p_stabil"], cp["p_turun"]
        ])

    output.seek(0)
    filename = f"probabilitas_bersyarat_model_{model_run_id}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@router.get("/aimrad_tables.csv")
def export_aimrad_tables_csv(
    model_run_id: int = Query(...),
    current_user: User = Depends(require_viewer_or_admin),
    db: Session = Depends(get_db)
):
    """
    Ekspor paket tabel hasil penelitian terstruktur format AIMRaD (siap dicopy ke Word/LaTeX/Artikel).
    """
    mr = db.query(ModelRun).filter(ModelRun.id == model_run_id).first()
    if not mr:
        raise HTTPException(status_code=404, detail="Model run tidak ditemukan")

    metrics = db.query(ModelMetric).filter(ModelMetric.model_run_id == model_run_id).all()
    influences = json.loads(mr.features_json or "[]")
    group_ablation = json.loads(mr.group_ablation_json or "[]")
    threshold_sens = json.loads(mr.threshold_sensitivity_json or "[]")

    output = io.StringIO()
    writer = csv.writer(output)

    # 1. Judul & Metadata
    writer.writerow(["=== TABEL HASIL EKSPERIMEN PENELITIAN NAIVE BAYES PUSKESMAS SARANG 1 (AIMRaD) ==="])
    writer.writerow(["Model Run ID", mr.id, "Periode Latih", mr.train_period, "Periode Uji", mr.test_period])
    writer.writerow(["Ambang Tren", f"{int(mr.threshold_pct*100)}%", "Perubahan Mutlak Min", mr.min_abs_change, "Smoothing Alpha", mr.alpha])
    writer.writerow([])

    # 2. Tabel Perbandingan Kinerja Algoritma
    writer.writerow(["--- TABEL 1. PERBANDINGAN KINERJA ALGORITMA PREDIKSI TREN ---"])
    writer.writerow(["Algoritma", "Akurasi", "F1-Macro", "Presisi Naik", "Presisi Stabil", "Presisi Turun", "Recall Naik", "Recall Stabil", "Recall Turun"])
    for m in metrics:
        p = json.loads(m.precision_json)
        r = json.loads(m.recall_json)
        writer.writerow([
            m.algorithm.value, m.accuracy, m.f1_macro,
            p.get("Naik", 0), p.get("Stabil", 0), p.get("Turun", 0),
            r.get("Naik", 0), r.get("Stabil", 0), r.get("Turun", 0)
        ])
    writer.writerow([])

    # 3. Matriks Konfusi Naive Bayes
    writer.writerow(["--- TABEL 2. MATRIKS KONFUSI NAIVE BAYES ---"])
    nb_metric = next((m for m in metrics if m.algorithm.value == "naive_bayes"), None)
    if nb_metric:
        cm = json.loads(nb_metric.confusion_matrix_json)
        writer.writerow(["Aktual \\ Prediksi", "Prediksi Naik", "Prediksi Stabil", "Prediksi Turun"])
        labels = ["Aktual Naik", "Aktual Stabil", "Aktual Turun"]
        for idx, row in enumerate(cm):
            writer.writerow([labels[idx]] + row)
    writer.writerow([])

    # 4. Tabel Analisis Sensitivitas Ambang
    writer.writerow(["--- TABEL 3. ANALISIS SENSITIVITAS AMBANG LABEL TREN ---"])
    writer.writerow(["Formula Ambang", "Threshold (%)", "Min Kasus", "Total Sampel", "Kasus Naik (%)", "Kasus Stabil (%)", "Kasus Turun (%)", "F1-Macro NB", "Akurasi NB"])
    for ts in threshold_sens:
        writer.writerow([
            ts.get("label_formula"), f"{int(ts.get('threshold_pct', 0)*100)}%", ts.get("min_abs_change"),
            ts.get("total_samples"), f"{ts.get('pct_naik')}%", f"{ts.get('pct_stabil')}%", f"{ts.get('pct_turun')}%",
            ts.get("f1_macro_nb"), ts.get("accuracy_nb")
        ])
    writer.writerow([])

    # 5. Tabel Ablasi Kelompok Atribut
    writer.writerow(["--- TABEL 4. ABLASI KELOMPOK ATRIBUT TEMATIK ---"])
    writer.writerow(["Kelompok Atribut", "Daftar Atribut Terkait", "F1-Macro Tanpa Kelompok", "Delta Penurunan F1"])
    for ga in group_ablation:
        writer.writerow([
            ga.get("group"), ", ".join(ga.get("features", [])), ga.get("f1_macro_without"), ga.get("delta_f1")
        ])

    output.seek(0)
    filename = f"tabel_aimrad_model_{model_run_id}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/template.csv")
def download_template_csv():
    """
    Template berkas CSV untuk pengunggahan data rekap bulanan Puskesmas Sarang 1
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "tahun", "bulan", "kode_icd10", "nama_penyakit", "jumlah_kasus",
        "kasus_laki", "kasus_perempuan",
        "kasus_<1th", "kasus_1-5th", "kasus_6-14th", "kasus_15-59th", "kasus_60th+",
        "total_kunjungan", "curah_hujan_mm", "suhu_rata2", "catatan"
    ])
    # Contoh baris
    writer.writerow([
        2025, 1, "J00", "Acute nasopharyngitis [common cold]", 150,
        70, 80, 10, 30, 25, 60, 25,
        2500, 220.5, 27.2, "DATA DUMMY"
    ])
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=template_rekap_bulanan_puskesmas.csv"}
    )
