import json
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Prediction, Disease, ModelRun, User
from app.schemas import PredictionRunRequest, PredictionRunResponse, PredictionItemResponse
from app.security import require_admin, require_viewer_or_admin
from app.services.predict import run_predictions_for_next_month

router = APIRouter(prefix="/api/predictions", tags=["predictions"])

@router.post("/run", response_model=PredictionRunResponse)
def execute_predictions(
    payload: PredictionRunRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Menghasilkan prediksi tren kasus penyakit untuk bulan berikutnya (Khusus Admin)
    """
    try:
        t_year, t_month, preds = run_predictions_for_next_month(payload.model_run_id, db)
        items = []
        for p in preds:
            disease = db.query(Disease).filter(Disease.id == p.disease_id).first()
            top_factors = json.loads(p.top_factors_json or "[]")
            items.append(PredictionItemResponse(
                id=p.id,
                disease_id=p.disease_id,
                icd10=disease.icd10 if disease else "-",
                disease_name=disease.name if disease else "-",
                target_year=p.target_year,
                target_month=p.target_month,
                predicted_class=p.predicted_class.value,
                p_naik=p.p_naik,
                p_stabil=p.p_stabil,
                p_turun=p.p_turun,
                top_factors=top_factors
            ))
        return PredictionRunResponse(
            model_run_id=payload.model_run_id,
            target_year=t_year,
            target_month=t_month,
            total_predictions=len(items),
            predictions=items
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "PREDICTION_FAILED", "message": f"Prediksi gagal dijalankan: {str(e)}", "details": []}}
        )

@router.get("", response_model=List[PredictionItemResponse])
def get_predictions(
    model_run_id: int = Query(...),
    current_user: User = Depends(require_viewer_or_admin),
    db: Session = Depends(get_db)
):
    """
    Mengambil hasil prediksi bulan berikutnya berdasarkan model_run_id
    """
    predictions = (
        db.query(Prediction, Disease)
        .join(Disease, Prediction.disease_id == Disease.id)
        .filter(Prediction.model_run_id == model_run_id)
        .order_by(Disease.icd10)
        .all()
    )

    return [
        PredictionItemResponse(
            id=p.id,
            disease_id=p.disease_id,
            icd10=d.icd10,
            disease_name=d.name,
            target_year=p.target_year,
            target_month=p.target_month,
            predicted_class=p.predicted_class.value,
            p_naik=p.p_naik,
            p_stabil=p.p_stabil,
            p_turun=p.p_turun,
            top_factors=json.loads(p.top_factors_json or "[]")
        ) for p, d in predictions
    ]

@router.get("/{prediction_id}/math-breakdown")
def get_prediction_math_breakdown(
    prediction_id: int,
    current_user: User = Depends(require_viewer_or_admin),
    db: Session = Depends(get_db)
):
    """
    Detail langkah perhitungan matematis Naive Bayes untuk satu baris diagnosis
    (Prior, Likelihood, Log Sum, Posterior ter-normalisasi - Sesuai Lampiran B Proposal).
    """
    pred = db.query(Prediction, Disease).join(Disease, Prediction.disease_id == Disease.id).filter(Prediction.id == prediction_id).first()
    if not pred:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data prediksi tidak ditemukan")
    
    p_obj, disease = pred
    breakdown = json.loads(p_obj.math_breakdown_json or "{}")
    return {
        "prediction_id": p_obj.id,
        "disease_id": p_obj.disease_id,
        "icd10": disease.icd10,
        "disease_name": disease.name,
        "target_period": f"{p_obj.target_year}-{p_obj.target_month:02d}",
        "predicted_class": p_obj.predicted_class.value,
        "p_naik": p_obj.p_naik,
        "p_stabil": p_obj.p_stabil,
        "p_turun": p_obj.p_turun,
        "breakdown": breakdown
    }
