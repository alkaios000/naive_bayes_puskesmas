from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.database import engine, Base, init_db_schema
from app.routers import auth, datasets, cases, models, predictions, export, users, pages

# Inisialisasi Skema Basis Data
init_db_schema()

app = FastAPI(
    title=settings.APP_NAME,
    description="Sistem Prediksi Tren Kasus Penyakit Terbanyak dengan Naive Bayes - Puskesmas Sarang 1",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount Static Files
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Custom Exception Handler untuk Format Error Standar (PRD Bagian 9)
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if request.url.path.startswith("/api"):
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            return JSONResponse(status_code=exc.status_code, content=detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": str(detail),
                    "details": []
                }
            }
        )
    return JSONResponse(status_code=exc.status_code, content={"message": str(exc.detail)})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        loc = " -> ".join([str(l) for l in err.get("loc", [])])
        errors.append({
            "field": loc,
            "type": err.get("type", "VALIDATION_ERROR"),
            "message": err.get("msg", "Input tidak valid")
        })
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Data permintaan tidak valid.",
                "details": errors
            }
        }
    )

# Health Check Route (PRD Bagian 9)
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "database": "connected"
    }

# Daftarkan Router
app.include_router(auth.router)
app.include_router(datasets.router)
app.include_router(cases.router)
app.include_router(models.router)
app.include_router(predictions.router)
app.include_router(export.router)
app.include_router(users.router)
app.include_router(pages.router)
