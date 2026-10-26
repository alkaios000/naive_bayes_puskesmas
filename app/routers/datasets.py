from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dataset, User
from app.schemas import DatasetResponse, DatasetUploadResponse
from app.security import require_admin, require_viewer_or_admin
from app.services.ingest import validate_and_parse_file, save_dataset_to_db
from app.config import settings

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Unggah dan validasi file dataset CSV / XLSX (Khusus Admin)
    """
    filename = file.filename or "dataset.csv"
    if not (filename.endswith(".csv") or filename.endswith(".xlsx") or filename.endswith(".xls")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"code": "INVALID_FORMAT", "message": "Ekstensi berkas harus .csv atau .xlsx", "details": []}}
        )

    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"code": "FILE_TOO_LARGE", "message": f"Ukuran berkas melebihi batas maksimum {settings.MAX_UPLOAD_SIZE_MB}MB", "details": []}}
        )

    # Validasi dan parsing isi file
    df, errors, is_dummy = validate_and_parse_file(content, filename)
    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": {
                    "code": "VALIDATION_FAILED",
                    "message": "Validasi integritas data gagal ditemukan kesalahan.",
                    "details": errors
                }
            }
        )

    # Simpan dataset ke database
    dataset = save_dataset_to_db(df, filename, is_dummy, current_user.id, db)

    return DatasetUploadResponse(
        dataset_id=dataset.id,
        name=dataset.name,
        is_dummy=dataset.is_dummy,
        row_count=dataset.row_count,
        disease_count=dataset.disease_count,
        start_period=dataset.start_period,
        end_period=dataset.end_period,
        message="Dataset berhasil diunggah dan disimpan ke sistem.",
        validation_report=[]
    )

@router.get("", response_model=List[DatasetResponse])
def get_datasets(
    current_user: User = Depends(require_viewer_or_admin),
    db: Session = Depends(get_db)
):
    """
    Mengambil daftar seluruh dataset yang tersedia
    """
    datasets = db.query(Dataset).order_by(Dataset.uploaded_at.desc()).all()
    return datasets

@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset_detail(
    dataset_id: int,
    current_user: User = Depends(require_viewer_or_admin),
    db: Session = Depends(get_db)
):
    """
    Mendapatkan detail dataset spesifik
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "NOT_FOUND", "message": "Dataset tidak ditemukan", "details": []}}
        )
    return dataset

@router.delete("/{dataset_id}")
def delete_dataset(
    dataset_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Menghapus dataset beserta seluruh relasinya (Khusus Admin)
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "NOT_FOUND", "message": "Dataset tidak ditemukan", "details": []}}
        )

    db.delete(dataset)
    db.commit()
    return {"status": "success", "message": f"Dataset ID {dataset_id} berhasil dihapus."}
