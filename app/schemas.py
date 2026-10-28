from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# --- Autentikasi ---
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str

class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True

# --- Dataset & Validasi ---
class ValidationErrorItem(BaseModel):
    row: int
    type: str
    message: str

class DatasetUploadResponse(BaseModel):
    dataset_id: int
    name: str
    is_dummy: bool
    row_count: int
    disease_count: int
    start_period: str
    end_period: str
    message: str
    validation_report: List[ValidationErrorItem] = []

class DatasetResponse(BaseModel):
    id: int
    name: str
    original_filename: str
    is_dummy: bool
    row_count: int
    start_period: str
    end_period: str
    disease_count: int
    uploaded_at: datetime
    uploaded_by: Optional[int] = None

    class Config:
        from_attributes = True

# --- Penyakit & Kasus Bulanan ---
class DiseaseResponse(BaseModel):
    id: int
    icd10: str
    name: str

    class Config:
        from_attributes = True

class MonthlyCaseResponse(BaseModel):
    id: int
    disease_id: int
    icd10: str
    disease_name: str
    year: int
    month: int
    cases: int
    male: Optional[int] = None
    female: Optional[int] = None
    rainfall_mm: Optional[float] = None
    temp_avg: Optional[float] = None

    class Config:
        from_attributes = True

# --- Label Preview ---
class LabelPreviewRequest(BaseModel):
    dataset_id: int
    threshold_pct: float = Field(0.10, ge=0.01, le=1.0)
    min_abs_change: int = Field(3, ge=0, le=100)

class DiseaseLabelSummary(BaseModel):
    disease_id: int
    icd10: str
    disease_name: str
    total_months: int
    naik: int
    stabil: int
    turun: int

class LabelPreviewResponse(BaseModel):
    dataset_id: int
    threshold_pct: float
    min_abs_change: int
    total_samples: int
    overall_distribution: Dict[str, int]
    disease_summaries: List[DiseaseLabelSummary]

# --- Model Pelatihan & Metrik ---
class TrainModelRequest(BaseModel):
    dataset_id: int
    threshold_pct: float = Field(0.10, ge=0.01, le=1.0)
    min_abs_change: int = Field(3, ge=0, le=100)
    alpha: float = Field(1.0, ge=0.01, le=10.0)
    test_ratio: float = Field(0.20, ge=0.10, le=0.50)
    features: Optional[List[str]] = None
    top_n_diseases: Optional[int] = Field(None, ge=1, le=100)

class ModelMetricResponse(BaseModel):
    algorithm: str
    accuracy: float
    f1_macro: float
    precision_json: Dict[str, float]
    recall_json: Dict[str, float]
    f1_json: Dict[str, float]
    confusion_matrix_json: List[List[int]]

class FeatureInfluenceResponse(BaseModel):
    feature: str
    f1_macro_without: float
    delta_f1: float

class GroupAblationResponse(BaseModel):
    group: str
    features: List[str]
    f1_macro_without: float
    delta_f1: float

class ConditionalProbabilityItem(BaseModel):
    feature: str
    category: str
    p_naik: float
    p_stabil: float
    p_turun: float

class ModelRunDetailResponse(BaseModel):
    id: int
    dataset_id: int
    created_at: datetime
    threshold_pct: float
    min_abs_change: int
    alpha: float
    test_ratio: float
    top_n_diseases: Optional[int] = None
    features: List[str]
    train_rows: int
    test_rows: int
    train_period: Optional[str]
    test_period: Optional[str]
    class_distribution: Dict[str, int]
    status: str
    error: Optional[str]
    is_better_than_baseline: bool
    summary_text: str
    metrics: List[ModelMetricResponse]
    feature_influences: List[FeatureInfluenceResponse]
    group_ablation: Optional[List[GroupAblationResponse]] = None
    cv_scores: Optional[Dict[str, Any]] = None
    disease_metrics: Optional[List[Dict[str, Any]]] = None
    preprocessing_report: Optional[Dict[str, Any]] = None
    threshold_sensitivity: Optional[List[Dict[str, Any]]] = None
    walk_forward_cv: Optional[Dict[str, Any]] = None

# --- Prediksi ---
class PredictionRunRequest(BaseModel):
    model_run_id: int

class PredictionItemResponse(BaseModel):
    id: Optional[int] = None
    disease_id: int
    icd10: str
    disease_name: str
    target_year: int
    target_month: int
    predicted_class: str
    p_naik: float
    p_stabil: float
    p_turun: float
    top_factors: Optional[List[str]] = []

class PredictionRunResponse(BaseModel):
    model_run_id: int
    target_year: int
    target_month: int
    total_predictions: int
    predictions: List[PredictionItemResponse]

# --- Format Error Standar ---
class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[List[Any]] = None

class StandardErrorResponse(BaseModel):
    error: ErrorDetail
