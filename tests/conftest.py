import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models import User, UserRole
from app.security import get_password_hash, create_access_token

TEST_DB_URL = "sqlite:///./test_naive_bayes.db"

@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    # Hapus file db jika ada
    db_file = Path("./test_naive_bayes.db")
    if db_file.exists():
        try:
            db_file.unlink()
        except Exception:
            pass

@pytest.fixture(scope="function")
def db_session(test_engine):
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    yield session
    session.rollback()
    session.close()

@pytest.fixture(scope="function")
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def admin_token(db_session):
    admin = db_session.query(User).filter(User.username == "test_admin").first()
    if not admin:
        admin = User(
            username="test_admin",
            password_hash=get_password_hash("password123"),
            role=UserRole.ADMIN
        )
        db_session.add(admin)
        db_session.commit()
    token = create_access_token({"sub": admin.username, "role": "admin"})
    return token

@pytest.fixture
def viewer_token(db_session):
    viewer = db_session.query(User).filter(User.username == "test_viewer").first()
    if not viewer:
        viewer = User(
            username="test_viewer",
            password_hash=get_password_hash("password123"),
            role=UserRole.VIEWER
        )
        db_session.add(viewer)
        db_session.commit()
    token = create_access_token({"sub": viewer.username, "role": "viewer"})
    return token
