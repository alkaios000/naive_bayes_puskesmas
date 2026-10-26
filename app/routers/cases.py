from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.database import get_db
from app.models import Disease, MonthlyCase, Dataset, User
from app.schemas import DiseaseResponse, MonthlyCaseResponse
from app.security import require_viewer_or_admin

router = APIRouter(prefix="/api", tags=["cases"])

@router.get("/diseases", response_model=List[DiseaseResponse])
def get_diseases(
    dataset_id: Optional[int] = None,
    current_user: User = Depends(require_viewer_or_admin),
    db: Session = Depends(get_db)
):
    """
    Mengambil daftar diagnosis penyakit, opsional filter per dataset_id
    """
    query = db.query(Disease)
    if dataset_id:
        query = query.join(MonthlyCase, MonthlyCase.disease_id == Disease.id).filter(MonthlyCase.dataset_id == dataset_id).distinct()
    return query.order_by(Disease.icd10).all()

@router.get("/monthly-cases", response_model=List[MonthlyCaseResponse])
def get_monthly_cases(
    dataset_id: Optional[int] = None,
    icd10: Optional[str] = None,
    from_period: Optional[str] = Query(None, alias="from"),
    to_period: Optional[str] = Query(None, alias="to"),
    current_user: User = Depends(require_viewer_or_admin),
    db: Session = Depends(get_db)
):
    """
    Filter data rekap kasus bulanan berdasarkan dataset, kode diagnosis, dan rentang periode
    """
    query = (
        db.query(
            MonthlyCase.id,
            MonthlyCase.disease_id,
            Disease.icd10,
            Disease.name.label("disease_name"),
            MonthlyCase.year,
            MonthlyCase.month,
            MonthlyCase.cases,
            MonthlyCase.male,
            MonthlyCase.female,
            MonthlyCase.rainfall_mm,
            MonthlyCase.temp_avg,
        )
        .join(Disease, MonthlyCase.disease_id == Disease.id)
    )

    if dataset_id:
        query = query.filter(MonthlyCase.dataset_id == dataset_id)
    if icd10:
        query = query.filter(Disease.icd10 == icd10.strip().upper())
    if from_period:
        parts = from_period.split("-")
        if len(parts) == 2:
            fy, fm = int(parts[0]), int(parts[1])
            query = query.filter((MonthlyCase.year > fy) | ((MonthlyCase.year == fy) & (MonthlyCase.month >= fm)))
    if to_period:
        parts = to_period.split("-")
        if len(parts) == 2:
            ty, tm = int(parts[0]), int(parts[1])
            query = query.filter((MonthlyCase.year < ty) | ((MonthlyCase.year == ty) & (MonthlyCase.month <= tm)))

    results = query.order_by(MonthlyCase.year, MonthlyCase.month, Disease.icd10).all()
    return [
        MonthlyCaseResponse(
            id=r.id,
            disease_id=r.disease_id,
            icd10=r.icd10,
            disease_name=r.disease_name,
            year=r.year,
            month=r.month,
            cases=r.cases,
            male=r.male,
            female=r.female,
            rainfall_mm=r.rainfall_mm,
            temp_avg=r.temp_avg
        ) for r in results
    ]

@router.get("/dashboard/summary")
def get_dashboard_summary(
    dataset_id: Optional[int] = None,
    current_user: User = Depends(require_viewer_or_admin),
    db: Session = Depends(get_db)
):
    """
    Ringkasan metrik dashboard: Top-10 penyakit terbanyak dan total tren kasus bulanan
    """
    # Ambil dataset aktif
    if dataset_id:
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    else:
        ds = db.query(Dataset).order_by(Dataset.uploaded_at.desc()).first()

    if not ds:
        return {
            "dataset_id": None,
            "is_dummy": False,
            "period": "-",
            "total_cases": 0,
            "total_diseases": 0,
            "top_diseases": [],
            "monthly_totals": []
        }

    # Top-10 Penyakit
    top_diseases_query = (
        db.query(
            Disease.icd10,
            Disease.name,
            func.sum(MonthlyCase.cases).label("total_cases")
        )
        .join(MonthlyCase, MonthlyCase.disease_id == Disease.id)
        .filter(MonthlyCase.dataset_id == ds.id)
        .group_by(Disease.id, Disease.icd10, Disease.name)
        .order_by(desc("total_cases"))
        .limit(10)
        .all()
    )

    top_diseases = [
        {"icd10": r.icd10, "name": r.name, "total_cases": int(r.total_cases)}
        for r in top_diseases_query
    ]

    # Total kasus bulanan
    monthly_totals_query = (
        db.query(
            MonthlyCase.year,
            MonthlyCase.month,
            func.sum(MonthlyCase.cases).label("total_cases")
        )
        .filter(MonthlyCase.dataset_id == ds.id)
        .group_by(MonthlyCase.year, MonthlyCase.month)
        .order_by(MonthlyCase.year, MonthlyCase.month)
        .all()
    )

    monthly_totals = [
        {"period": f"{r.year}-{r.month:02d}", "total_cases": int(r.total_cases)}
        for r in monthly_totals_query
    ]

    total_sum = sum(m["total_cases"] for m in monthly_totals)

    return {
        "dataset_id": ds.id,
        "is_dummy": ds.is_dummy,
        "period": f"{ds.start_period} s.d. {ds.end_period}",
        "total_cases": total_sum,
        "total_diseases": ds.disease_count,
        "top_diseases": top_diseases,
        "monthly_totals": monthly_totals
    }
