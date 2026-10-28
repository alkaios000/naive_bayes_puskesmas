from datetime import datetime
import enum
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, 
    Text, UniqueConstraint, Enum, Index
)
from sqlalchemy.orm import relationship
from app.database import Base

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    VIEWER = "viewer"

class ModelRunStatus(str, enum.Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

class AlgorithmType(str, enum.Enum):
    NAIVE_BAYES = "naive_bayes"
    MAJORITY = "majority"
    PERSISTENCE = "persistence"
    DECISION_TREE = "decision_tree"
    KNN = "knn"

class TrendClass(str, enum.Enum):
    NAIK = "Naik"
    STABIL = "Stabil"
    TURUN = "Turun"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    datasets = relationship("Dataset", back_populates="uploader")
    model_runs = relationship("ModelRun", back_populates="creator")

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    original_filename = Column(String(255), nullable=False)
    is_dummy = Column(Boolean, default=False, nullable=False)
    row_count = Column(Integer, default=0, nullable=False)
    start_period = Column(String(7), nullable=False)  # format YYYY-MM
    end_period = Column(String(7), nullable=False)    # format YYYY-MM
    disease_count = Column(Integer, default=0, nullable=False)
    uploaded_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    uploader = relationship("User", back_populates="datasets")
    monthly_cases = relationship("MonthlyCase", back_populates="dataset", cascade="all, delete-orphan")
    model_runs = relationship("ModelRun", back_populates="dataset", cascade="all, delete-orphan")

class Disease(Base):
    __tablename__ = "diseases"
    
    id = Column(Integer, primary_key=True, index=True)
    icd10 = Column(String(20), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    
    monthly_cases = relationship("MonthlyCase", back_populates="disease")
    predictions = relationship("Prediction", back_populates="disease")

class MonthlyCase(Base):
    __tablename__ = "monthly_cases"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    disease_id = Column(Integer, ForeignKey("diseases.id", ondelete="CASCADE"), nullable=False)
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    cases = Column(Integer, nullable=False)
    male = Column(Integer, nullable=True)
    female = Column(Integer, nullable=True)
    age_lt1 = Column(Integer, nullable=True)
    age_1_5 = Column(Integer, nullable=True)
    age_6_14 = Column(Integer, nullable=True)
    age_15_59 = Column(Integer, nullable=True)
    age_gte60 = Column(Integer, nullable=True)
    total_visits = Column(Integer, nullable=True)
    rainfall_mm = Column(Float, nullable=True)
    temp_avg = Column(Float, nullable=True)
    
    dataset = relationship("Dataset", back_populates="monthly_cases")
    disease = relationship("Disease", back_populates="monthly_cases")
    
    __table_args__ = (
        UniqueConstraint("dataset_id", "disease_id", "year", "month", name="uq_dataset_disease_period"),
        Index("idx_cases_dataset_disease_period", "dataset_id", "disease_id", "year", "month"),
    )

class ModelRun(Base):
    __tablename__ = "model_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    threshold_pct = Column(Float, default=0.10, nullable=False)
    min_abs_change = Column(Integer, default=3, nullable=False)
    alpha = Column(Float, default=1.0, nullable=False)
    test_ratio = Column(Float, default=0.20, nullable=False)
    features_json = Column(Text, nullable=False)
    
    train_rows = Column(Integer, default=0, nullable=False)
    test_rows = Column(Integer, default=0, nullable=False)
    train_period = Column(String(50), nullable=True)
    test_period = Column(String(50), nullable=True)
    class_distribution_json = Column(Text, nullable=True)
    model_path = Column(String(255), nullable=True)
    status = Column(Enum(ModelRunStatus), default=ModelRunStatus.RUNNING, nullable=False)
    error = Column(Text, nullable=True)
    
    # Bidang analitik baru sesuai proposal
    top_n_diseases = Column(Integer, nullable=True)
    cv_scores_json = Column(Text, nullable=True)
    group_ablation_json = Column(Text, nullable=True)
    disease_metrics_json = Column(Text, nullable=True)
    preprocessing_report_json = Column(Text, nullable=True)
    threshold_sensitivity_json = Column(Text, nullable=True)
    walk_forward_json = Column(Text, nullable=True)
    
    dataset = relationship("Dataset", back_populates="model_runs")
    creator = relationship("User", back_populates="model_runs")
    metrics = relationship("ModelMetric", back_populates="model_run", cascade="all, delete-orphan")
    feature_influences = relationship("FeatureInfluence", back_populates="model_run", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="model_run", cascade="all, delete-orphan")

class ModelMetric(Base):
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=False)
    algorithm = Column(Enum(AlgorithmType), nullable=False)
    accuracy = Column(Float, nullable=False)
    f1_macro = Column(Float, nullable=False)
    precision_json = Column(Text, nullable=False)
    recall_json = Column(Text, nullable=False)
    f1_json = Column(Text, nullable=False)
    confusion_matrix_json = Column(Text, nullable=False)
    
    model_run = relationship("ModelRun", back_populates="metrics")

class FeatureInfluence(Base):
    __tablename__ = "feature_influence"
    
    id = Column(Integer, primary_key=True, index=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=False)
    feature = Column(String(100), nullable=False)
    f1_macro_without = Column(Float, nullable=False)
    delta_f1 = Column(Float, nullable=False)
    
    model_run = relationship("FeatureInfluence", back_populates="feature_influences") if False else relationship("ModelRun", back_populates="feature_influences")

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=False)
    disease_id = Column(Integer, ForeignKey("diseases.id", ondelete="CASCADE"), nullable=False)
    target_year = Column(Integer, nullable=False)
    target_month = Column(Integer, nullable=False)
    predicted_class = Column(Enum(TrendClass), nullable=False)
    p_naik = Column(Float, nullable=False)
    p_stabil = Column(Float, nullable=False)
    p_turun = Column(Float, nullable=False)
    
    # Analisis penjelas dan contoh hitung matematis
    top_factors_json = Column(Text, nullable=True)
    math_breakdown_json = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    model_run = relationship("ModelRun", back_populates="predictions")
    disease = relationship("Disease", back_populates="predictions")
    
    __table_args__ = (
        Index("idx_predictions_model_run", "model_run_id"),
    )
