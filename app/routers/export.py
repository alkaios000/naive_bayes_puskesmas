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
