import json
from typing import List, Dict, Any, Optional
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import (
    ModelRun, ModelMetric, FeatureInfluence, Dataset, MonthlyCase, Disease, User, AlgorithmType
)
from app.schemas import (
    LabelPreviewRequest, LabelPreviewResponse, DiseaseLabelSummary,
    TrainModelRequest, ModelRunDetailResponse, ModelMetricResponse, FeatureInfluenceResponse
)
from app.security import require_admin, require_viewer_or_admin
from app.services.labeling import generate_labels_for_dataframe
from app.services.training import train_and_evaluate_model

router = APIRouter(prefix="/api", tags=["models"])

@router.post("/labels/preview", response_model=LabelPreviewResponse)
def preview_labels(
    payload: LabelPreviewRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Pratinjau pembentukan label tren (Naik, Stabil, Turun) sebelum melatih model (Khusus Admin)
    """
    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "NOT_FOUND", "message": "Dataset tidak ditemukan", "details": []}}
        )

    # Ambil data kasus
    cases_query = (
        db.query(
            MonthlyCase.year.label("tahun"),
            MonthlyCase.month.label("bulan"),
            Disease.id.label("disease_id"),
            Disease.icd10.label("kode_icd10"),
            Disease.name.label("nama_penyakit"),
            MonthlyCase.cases.label("jumlah_kasus"),
        )
        .join(Disease, MonthlyCase.disease_id == Disease.id)
        .filter(MonthlyCase.dataset_id == payload.dataset_id)
        .order_by(MonthlyCase.year, MonthlyCase.month, Disease.icd10)
    )
    df = pd.DataFrame([dict(r._mapping) for r in cases_query.all()])
    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"code": "EMPTY_DATASET", "message": "Dataset tidak memiliki baris data kasus.", "details": []}}
        )

    df_labeled = generate_labels_for_dataframe(df, payload.threshold_pct, payload.min_abs_change)

    # Hitung distribusi total
    valid_labels = df_labeled["label"].dropna()
    total_dist = valid_labels.value_counts().to_dict()
    total_samples = len(valid_labels)

    # Distribusi per penyakit
    disease_summaries: List[DiseaseLabelSummary] = []
    grouped = df_labeled.groupby(["disease_id", "kode_icd10", "nama_penyakit"])
    for (d_id, icd, name), grp in grouped:
        lbl_counts = grp["label"].dropna().value_counts().to_dict()
        disease_summaries.append(DiseaseLabelSummary(
            disease_id=d_id,
            icd10=icd,
            disease_name=name,
            total_months=len(grp),
            naik=lbl_counts.get("Naik", 0),
            stabil=lbl_counts.get("Stabil", 0),
            turun=lbl_counts.get("Turun", 0)
        ))

    return LabelPreviewResponse(
        dataset_id=payload.dataset_id,
        threshold_pct=payload.threshold_pct,
        min_abs_change=payload.min_abs_change,
        total_samples=total_samples,
        overall_distribution={
            "Naik": total_dist.get("Naik", 0),
            "Stabil": total_dist.get("Stabil", 0),
            "Turun": total_dist.get("Turun", 0),
        },
        disease_summaries=disease_summaries
    )

@router.post("/models/train")
def train_model(
    payload: TrainModelRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Melatih model Naive Bayes dan 4 algoritma pembanding secara sinkron (Khusus Admin)
    """
    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "NOT_FOUND", "message": "Dataset tidak ditemukan", "details": []}}
        )

    try:
        model_run = train_and_evaluate_model(
            dataset_id=payload.dataset_id,
            user_id=current_user.id,
            threshold_pct=payload.threshold_pct,
            min_abs_change=payload.min_abs_change,
            alpha=payload.alpha,
            test_ratio=payload.test_ratio,
            selected_features=payload.features,
            db=db
        )
        return {
            "status": "success",
            "message": "Pelatihan dan evaluasi model berhasil diselesaikan.",
            "model_run_id": model_run.id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"code": "TRAINING_FAILED", "message": f"Pelatihan model gagal: {str(e)}", "details": []}}
        )

@router.get("/models")
def get_model_runs(
    dataset_id: Optional[int] = None,
    current_user: User = Depends(require_viewer_or_admin),
    db: Session = Depends(get_db)
):
    """
    Daftar seluruh riwayat pelatihan model
    """
    query = db.query(ModelRun)
    if dataset_id:
        query = query.filter(ModelRun.dataset_id == dataset_id)
    runs = query.order_by(ModelRun.created_at.desc()).all()

    results = []
    for r in runs:
        nb_metric = db.query(ModelMetric).filter(
            ModelMetric.model_run_id == r.id,
            ModelMetric.algorithm == AlgorithmType.NAIVE_BAYES
        ).first()

        results.append({
            "id": r.id,
            "dataset_id": r.dataset_id,
            "created_at": r.created_at,
            "threshold_pct": r.threshold_pct,
            "min_abs_change": r.min_abs_change,
            "alpha": r.alpha,
            "status": r.status.value,
            "f1_macro_nb": nb_metric.f1_macro if nb_metric else None,
            "accuracy_nb": nb_metric.accuracy if nb_metric else None,
        })
    return results

@router.get("/models/{model_run_id}")
def get_model_run_detail(
    model_run_id: int,
    current_user: User = Depends(require_viewer_or_admin),
    db: Session = Depends(get_db)
):
    """
    Detail parameter, metrik 5 algoritma, dan pengaruh fitur
    """
    model_run = db.query(ModelRun).filter(ModelRun.id == model_run_id).first()
    if not model_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "NOT_FOUND", "message": "Model run tidak ditemukan", "details": []}}
        )

    metrics = db.query(ModelMetric).filter(ModelMetric.model_run_id == model_run_id).all()
    influences = db.query(FeatureInfluence).filter(FeatureInfluence.model_run_id == model_run_id).order_by(FeatureInfluence.delta_f1.desc()).all()

    # Periksa kesimpulan otomatis: jika F1 macro NB <= baseline terbaik (mayoritas / persistence)
    nb_metric = next((m for m in metrics if m.algorithm == AlgorithmType.NAIVE_BAYES), None)
    baseline_metrics = [m for m in metrics if m.algorithm in [AlgorithmType.MAJORITY, AlgorithmType.PERSISTENCE]]
    max_baseline_f1 = max([b.f1_macro for b in baseline_metrics]) if baseline_metrics else 0.0

    is_better = False
    if nb_metric and nb_metric.f1_macro > max_baseline_f1:
        is_better = True
        summary_text = f"Model Naive Bayes unggul dengan F1-macro {nb_metric.f1_macro} dibandingkan baseline terbaik ({max_baseline_f1})."
    else:
        summary_text = "Model belum lebih baik dari baseline"

    return {
        "id": model_run.id,
        "dataset_id": model_run.dataset_id,
        "created_at": model_run.created_at,
        "threshold_pct": model_run.threshold_pct,
        "min_abs_change": model_run.min_abs_change,
        "alpha": model_run.alpha,
        "test_ratio": model_run.test_ratio,
        "features": json.loads(model_run.features_json),
        "train_rows": model_run.train_rows,
        "test_rows": model_run.test_rows,
        "train_period": model_run.train_period,
        "test_period": model_run.test_period,
        "class_distribution": json.loads(model_run.class_distribution_json or "{}"),
        "status": model_run.status.value,
        "is_better_than_baseline": is_better,
        "summary_text": summary_text,
        "metrics": [
            {
                "algorithm": m.algorithm.value,
                "accuracy": m.accuracy,
                "f1_macro": m.f1_macro,
                "precision": json.loads(m.precision_json),
                "recall": json.loads(m.recall_json),
                "f1": json.loads(m.f1_json),
                "confusion_matrix": json.loads(m.confusion_matrix_json),
            } for m in metrics
        ],
        "feature_influences": [
            {
                "feature": fi.feature,
                "f1_macro_without": fi.f1_macro_without,
                "delta_f1": fi.delta_f1,
            } for fi in influences
        ]
    }

@router.get("/models/{model_run_id}/metrics")
def get_model_metrics(
    model_run_id: int,
    current_user: User = Depends(require_viewer_or_admin),
    db: Session = Depends(get_db)
):
    """Metrik perbandingan kelima algoritma"""
    metrics = db.query(ModelMetric).filter(ModelMetric.model_run_id == model_run_id).all()
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "NOT_FOUND", "message": "Metrik tidak ditemukan", "details": []}}
        )
    return [
        {
            "algorithm": m.algorithm.value,
            "accuracy": m.accuracy,
            "f1_macro": m.f1_macro,
            "precision": json.loads(m.precision_json),
            "recall": json.loads(m.recall_json),
            "f1": json.loads(m.f1_json),
            "confusion_matrix": json.loads(m.confusion_matrix_json),
        } for m in metrics
    ]

@router.get("/models/{model_run_id}/confusion-matrix")
def get_confusion_matrix(
    model_run_id: int,
    current_user: User = Depends(require_viewer_or_admin),
    db: Session = Depends(get_db)
):
    """Confusion matrix seluruh model pembanding"""
    metrics = db.query(ModelMetric).filter(ModelMetric.model_run_id == model_run_id).all()
    return {
        m.algorithm.value: json.loads(m.confusion_matrix_json) for m in metrics
    }

@router.get("/models/{model_run_id}/feature-influence")
def get_feature_influence(
    model_run_id: int,
    current_user: User = Depends(require_viewer_or_admin),
    db: Session = Depends(get_db)
):
    """Pengaruh atribut (drop-one-feature)"""
    influences = db.query(FeatureInfluence).filter(FeatureInfluence.model_run_id == model_run_id).order_by(FeatureInfluence.delta_f1.desc()).all()
    return [
        {
            "feature": fi.feature,
            "f1_macro_without": fi.f1_macro_without,
            "delta_f1": fi.delta_f1,
        } for fi in influences
    ]

@router.delete("/models/{model_run_id}")
def delete_model_run(
    model_run_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Menghapus riwayat pelatihan model (Khusus Admin)"""
    mr = db.query(ModelRun).filter(ModelRun.id == model_run_id).first()
    if not mr:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "NOT_FOUND", "message": "Model run tidak ditemukan", "details": []}}
        )
    db.delete(mr)
    db.commit()
    return {"status": "success", "message": f"Model run ID {model_run_id} berhasil dihapus."}
