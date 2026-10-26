from typing import List
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
            items.append(PredictionItemResponse(
                disease_id=p.disease_id,
                icd10=disease.icd10 if disease else "-",
                disease_name=disease.name if disease else "-",
                target_year=p.target_year,
                target_month=p.target_month,
                predicted_class=p.predicted_class.value,
                p_naik=p.p_naik,
                p_stabil=p.p_stabil,
                p_turun=p.p_turun
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
            disease_id=p.disease_id,
            icd10=d.icd10,
            disease_name=d.name,
            target_year=p.target_year,
            target_month=p.target_month,
            predicted_class=p.predicted_class.value,
            p_naik=p.p_naik,
            p_stabil=p.p_stabil,
            p_turun=p.p_turun
        ) for p, d in predictions
    ]
