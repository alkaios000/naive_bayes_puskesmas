import os
import json
from typing import Optional
from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pathlib import Path

from app.database import get_db
from app.models import (
    Dataset, Disease, MonthlyCase, ModelRun, ModelMetric, FeatureInfluence, 
    Prediction, AlgorithmType, User, UserRole
)
from app.security import get_current_user_optional
from app.config import settings
from app.services.features import ALL_AVAILABLE_FEATURES, FEATURE_GROUPS

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

router = APIRouter(include_in_schema=False)

@router.get("/login", response_class=HTMLResponse)
def page_login(request: Request, user=Depends(get_current_user_optional)):
    if user:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"app_name": settings.APP_NAME}
    )

@router.get("/logout")
def page_logout():
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("access_token")
    return response

@router.get("/", response_class=HTMLResponse)
@router.get("/dashboard", response_class=HTMLResponse)
def page_dashboard(
    request: Request,
    dataset_id: Optional[int] = None,
    user=Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    datasets = db.query(Dataset).order_by(Dataset.uploaded_at.desc()).all()
    selected_dataset = None
    if dataset_id:
        selected_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not selected_dataset and datasets:
        selected_dataset = datasets[0]

    latest_model = db.query(ModelRun).order_by(ModelRun.created_at.desc()).first()
    total_diseases = db.query(Disease).count()
    total_cases = db.query(MonthlyCase).count()
    
    # Ambil 10 penyakit terbanyak berdasarkan total kasus historis
    top_diseases = []
    if selected_dataset:
        cases_q = (
            db.query(Disease.icd10, Disease.name, MonthlyCase.cases)
            .join(Disease, MonthlyCase.disease_id == Disease.id)
            .filter(MonthlyCase.dataset_id == selected_dataset.id)
            .all()
        )
        disease_sum = {}
        for icd, name, cs in cases_q:
            disease_sum[icd] = disease_sum.get(icd, {"icd10": icd, "name": name, "total": 0})
            disease_sum[icd]["total"] += cs
        
        sorted_d = sorted(disease_sum.values(), key=lambda x: x["total"], reverse=True)
        top_diseases = sorted_d[:10]

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "app_name": settings.APP_NAME,
            "current_user": user,
            "datasets": datasets,
            "selected_dataset": selected_dataset,
            "latest_model": latest_model,
            "total_diseases": total_diseases,
            "total_cases": total_cases,
            "top_diseases": top_diseases,
            "active_menu": "dashboard"
        }
    )

@router.get("/upload", response_class=HTMLResponse)
def page_upload(request: Request, user=Depends(get_current_user_optional)):
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={
            "app_name": settings.APP_NAME,
            "current_user": user,
            "active_menu": "upload"
        }
    )

@router.get("/datasets", response_class=HTMLResponse)
def page_datasets(request: Request, user=Depends(get_current_user_optional), db: Session = Depends(get_db)):
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    datasets = db.query(Dataset).order_by(Dataset.uploaded_at.desc()).all()
    return templates.TemplateResponse(
        request=request,
        name="datasets.html",
        context={
            "app_name": settings.APP_NAME,
            "current_user": user,
            "datasets": datasets,
            "active_menu": "datasets"
        }
    )

@router.get("/users", response_class=HTMLResponse)
def page_users(
    request: Request,
    user=Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    if user.role != UserRole.ADMIN:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)

    users = db.query(User).order_by(User.id.asc()).all()
    user_list = []
    for u in users:
        ds_count = db.query(Dataset).filter(Dataset.uploaded_by == u.id).count()
        run_count = db.query(ModelRun).filter(ModelRun.created_by == u.id).count()
        user_list.append({
            "id": u.id,
            "username": u.username,
            "role": u.role.value,
            "created_at": u.created_at,
            "datasets_count": ds_count,
            "runs_count": run_count
        })

    return templates.TemplateResponse(
        request=request,
        name="users.html",
        context={
            "app_name": settings.APP_NAME,
            "current_user": user,
            "users": user_list,
            "active_menu": "users"
        }
    )

@router.get("/train", response_class=HTMLResponse)
def page_train(
    request: Request,
    dataset_id: Optional[int] = None,
    user=Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    datasets = db.query(Dataset).order_by(Dataset.uploaded_at.desc()).all()
    selected_ds = None
    if dataset_id:
        selected_ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not selected_ds and datasets:
        selected_ds = datasets[0]

    return templates.TemplateResponse(
        request=request,
        name="train.html",
        context={
            "app_name": settings.APP_NAME,
            "current_user": user,
            "datasets": datasets,
            "selected_dataset": selected_ds,
            "available_features": ALL_AVAILABLE_FEATURES,
            "feature_groups": FEATURE_GROUPS,
            "active_menu": "train"
        }
    )

@router.get("/results", response_class=HTMLResponse)
@router.get("/results/{model_run_id}", response_class=HTMLResponse)
def page_results(
    request: Request,
    model_run_id: Optional[int] = None,
    user=Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    runs = db.query(ModelRun).order_by(ModelRun.created_at.desc()).all()
    selected_run = None
    if model_run_id:
        selected_run = db.query(ModelRun).filter(ModelRun.id == model_run_id).first()
    if not selected_run and runs:
        selected_run = runs[0]

    metrics = []
    influences = []
    is_better = False
    summary_text = ""
    is_dummy = False
    cv_scores = None
    group_ablation = []
    disease_metrics = []
    preprocessing_report = None
    threshold_sensitivity = []
    walk_forward_cv = None
    conditional_probabilities = []

    if selected_run:
        ds = db.query(Dataset).filter(Dataset.id == selected_run.dataset_id).first()
        is_dummy = ds.is_dummy if ds else False

        metrics = db.query(ModelMetric).filter(ModelMetric.model_run_id == selected_run.id).all()
        influences = db.query(FeatureInfluence).filter(FeatureInfluence.model_run_id == selected_run.id).order_by(FeatureInfluence.delta_f1.desc()).all()

        nb_metric = next((m for m in metrics if m.algorithm == AlgorithmType.NAIVE_BAYES), None)
        baseline_metrics = [m for m in metrics if m.algorithm in [AlgorithmType.MAJORITY, AlgorithmType.PERSISTENCE]]
        max_baseline_f1 = max([b.f1_macro for b in baseline_metrics]) if baseline_metrics else 0.0

        if nb_metric and nb_metric.f1_macro > max_baseline_f1:
            is_better = True
            summary_text = f"Model Naive Bayes unggul dengan F1-macro {nb_metric.f1_macro:.4f} melampaui baseline terbaik ({max_baseline_f1:.4f})."
        else:
            summary_text = "Model belum lebih baik dari baseline"

        if selected_run.cv_scores_json:
            cv_scores = json.loads(selected_run.cv_scores_json)
        if selected_run.group_ablation_json:
            group_ablation = json.loads(selected_run.group_ablation_json)
        if selected_run.disease_metrics_json:
            disease_metrics = json.loads(selected_run.disease_metrics_json)
        if selected_run.preprocessing_report_json:
            preprocessing_report = json.loads(selected_run.preprocessing_report_json)
        if selected_run.threshold_sensitivity_json:
            threshold_sensitivity = json.loads(selected_run.threshold_sensitivity_json)
        if selected_run.walk_forward_json:
            walk_forward_cv = json.loads(selected_run.walk_forward_json)

        if selected_run.model_path and os.path.exists(selected_run.model_path):
            try:
                import joblib
                artifact = joblib.load(selected_run.model_path)
                conditional_probabilities = artifact.get("conditional_probabilities", [])
            except Exception:
                pass

    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={
            "app_name": settings.APP_NAME,
            "current_user": user,
            "runs": runs,
            "selected_run": selected_run,
            "metrics": metrics,
            "influences": influences,
            "cv_scores": cv_scores,
            "group_ablation": group_ablation,
            "disease_metrics": disease_metrics,
            "preprocessing_report": preprocessing_report,
            "threshold_sensitivity": threshold_sensitivity,
            "walk_forward_cv": walk_forward_cv,
            "conditional_probabilities": conditional_probabilities,
            "is_better": is_better,
            "summary_text": summary_text,
            "is_dummy": is_dummy,
            "active_menu": "results"
        }
    )

@router.get("/predictions", response_class=HTMLResponse)
@router.get("/predictions/{model_run_id}", response_class=HTMLResponse)
def page_predictions(
    request: Request,
    model_run_id: Optional[int] = None,
    user=Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    runs = db.query(ModelRun).order_by(ModelRun.created_at.desc()).all()
    selected_run = None
    if model_run_id:
        selected_run = db.query(ModelRun).filter(ModelRun.id == model_run_id).first()
    if not selected_run and runs:
        selected_run = runs[0]

    prediction_items = []
    is_dummy = False
    target_period = "-"

    if selected_run:
        ds = db.query(Dataset).filter(Dataset.id == selected_run.dataset_id).first()
        is_dummy = ds.is_dummy if ds else False

        predictions = (
            db.query(Prediction, Disease)
            .join(Disease, Prediction.disease_id == Disease.id)
            .filter(Prediction.model_run_id == selected_run.id)
            .order_by(Disease.icd10)
            .all()
        )
        if predictions:
            p0 = predictions[0][0]
            target_period = f"{p0.target_year}-{p0.target_month:02d}"

        for p, d in predictions:
            factors = json.loads(p.top_factors_json or "[]")
            prediction_items.append({
                "id": p.id,
                "disease_id": p.disease_id,
                "icd10": d.icd10,
                "name": d.name,
                "target_year": p.target_year,
                "target_month": p.target_month,
                "predicted_class": p.predicted_class,
                "p_naik": p.p_naik,
                "p_stabil": p.p_stabil,
                "p_turun": p.p_turun,
                "top_factors": factors
            })

    return templates.TemplateResponse(
        request=request,
        name="predictions.html",
        context={
            "app_name": settings.APP_NAME,
            "current_user": user,
            "runs": runs,
            "selected_run": selected_run,
            "predictions": prediction_items,
            "is_dummy": is_dummy,
            "target_period": target_period,
            "active_menu": "predictions"
        }
    )
