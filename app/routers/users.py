from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, UserRole, Dataset, ModelRun
from app.schemas import UserResponse, UserCreate, UserUpdate
from app.security import require_admin, get_password_hash

router = APIRouter(prefix="/api/users", tags=["users"])

@router.get("", response_model=List[UserResponse])
def get_users(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Mengambil daftar seluruh pengguna sistem (Khusus Admin)
    """
    users = db.query(User).order_by(User.id.asc()).all()
    return users

@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: UserCreate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Membuat akun pengguna baru (Khusus Admin)
    """
    # Validasi role
    role_val = payload.role.strip().lower()
    if role_val not in [UserRole.ADMIN.value, UserRole.VIEWER.value]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": {"code": "INVALID_ROLE", "message": "Peran pengguna harus 'admin' atau 'viewer'", "details": []}}
        )

    # Cek apakah username sudah dipakai
    clean_username = payload.username.strip()
    existing = db.query(User).filter(User.username == clean_username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"code": "USER_EXISTS", "message": f"Nama pengguna '{clean_username}' sudah terdaftar", "details": []}}
        )

    new_user = User(
        username=clean_username,
        password_hash=get_password_hash(payload.password),
        role=UserRole(role_val)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@router.put("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    payload: UserUpdate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Memperbarui peran atau kata sandi pengguna (Khusus Admin)
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "NOT_FOUND", "message": "Pengguna tidak ditemukan", "details": []}}
        )

    # Validasi dan update role jika diberikan
    if payload.role:
        role_val = payload.role.strip().lower()
        if role_val not in [UserRole.ADMIN.value, UserRole.VIEWER.value]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": {"code": "INVALID_ROLE", "message": "Peran pengguna harus 'admin' atau 'viewer'", "details": []}}
            )

        # Mencegah admin mengubah perannya sendiri menjadi viewer jika tidak ada admin lain
        if user.id == current_user.id and role_val != UserRole.ADMIN.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"code": "CANNOT_DEMOTE_SELF", "message": "Anda tidak dapat menurunkan peran akun Anda sendiri menjadi viewer", "details": []}}
            )

        # Jika target admin diturunkan ke viewer, pastikan masih ada minimal 1 admin lain
        if user.role == UserRole.ADMIN and role_val == UserRole.VIEWER.value:
            admin_count = db.query(User).filter(User.role == UserRole.ADMIN).count()
            if admin_count <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": {"code": "LAST_ADMIN", "message": "Tidak dapat mengubah peran admin terakhir di sistem", "details": []}}
                )

        user.role = UserRole(role_val)

    # Update kata sandi jika diisi
    if payload.password and payload.password.strip():
        user.password_hash = get_password_hash(payload.password.strip())

    db.commit()
    db.refresh(user)
    return user

@router.delete("/{user_id}")
def delete_user(
    user_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Menghapus akun pengguna (Khusus Admin)
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"code": "NOT_FOUND", "message": "Pengguna tidak ditemukan", "details": []}}
        )

    # Mencegah menghapus akun sendiri
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"code": "CANNOT_DELETE_SELF", "message": "Anda tidak dapat menghapus akun Anda sendiri yang sedang aktif", "details": []}}
        )

    # Mencegah menghapus admin terakhir
    if user.role == UserRole.ADMIN:
        admin_count = db.query(User).filter(User.role == UserRole.ADMIN).count()
        if admin_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"code": "LAST_ADMIN", "message": "Tidak dapat menghapus admin terakhir di sistem", "details": []}}
            )

    # Lepaskan relasi foreign key sebelum delete
    db.query(Dataset).filter(Dataset.uploaded_by == user.id).update({Dataset.uploaded_by: None})
    db.query(ModelRun).filter(ModelRun.created_by == user.id).update({ModelRun.created_by: None})

    username_deleted = user.username
    db.delete(user)
    db.commit()

    return {
        "status": "success",
        "message": f"Pengguna '{username_deleted}' berhasil dihapus dari sistem."
    }
