# -*- coding: utf-8 -*-
"""
Script: app.py
Chức năng: Giao diện Web Dashboard hiện đại (FastAPI + Modern SPA) thay thế giao diện Streamlit cũ,
dự đoán sinh viên có nguy cơ bỏ học, phân tầng rủi ro 3 mức, mô phỏng What-If,
sàng lọc hàng loạt CSV, đối sánh mô hình Luồng A/B và phân tích lỗi (N = 885).

Cách chạy:
    python app.py
Hoặc:
    uvicorn app:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.utils import FIGURES_DIR, PROJECT_ROOT
from src.web_service import (
    get_batch_sample_csv_bytes,
    get_experiments_data,
    get_metadata,
    get_overview_data,
    get_resources,
    get_test_sample_detail,
    get_test_samples,
    predict_batch_csv,
    predict_single,
)

WEB_DIR: Path = PROJECT_ROOT / "web"


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(default_factory=dict)
    threshold: float = Field(default=0.50, ge=0.05, le=0.95)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Nạp sẵn dữ liệu và mô hình Random Forest Luồng B ngay khi khởi động server."""
    get_resources()
    yield


app = FastAPI(
    title="EduGuard AI — Cảnh báo sớm Sinh viên Bỏ học (FIT4013 - Nhóm 9)",
    description="API & Web Dashboard dự đoán nguy cơ bỏ học của sinh viên sử dụng Pipeline Luồng B + Random Forest.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Phục vụ trang giao diện Web SPA chính."""
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy giao diện web/index.html")
    return FileResponse(
        str(index_path),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "Student Dropout Prediction API"}


@app.get("/api/overview")
async def api_overview() -> Dict[str, Any]:
    return get_overview_data()


@app.get("/api/metadata")
async def api_metadata() -> Dict[str, Any]:
    return get_metadata()


@app.post("/api/predict")
async def api_predict(payload: PredictRequest) -> Dict[str, Any]:
    try:
        return predict_single(payload.features, threshold=payload.threshold)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Lỗi dự đoán: {exc}") from exc


@app.post("/api/predict-batch")
async def api_predict_batch(
    file: UploadFile = File(...),
    threshold: float = Query(default=0.50, ge=0.05, le=0.95),
) -> Dict[str, Any]:
    try:
        raw_bytes = await file.read()
        return predict_batch_csv(raw_bytes, threshold=threshold)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý file CSV: {exc}") from exc


@app.get("/api/batch-sample")
async def api_batch_sample(limit: int = Query(default=25, ge=5, le=200)):
    """Tải về file CSV mẫu (25 hồ sơ từ tập kiểm thử) để thử nghiệm sàng lọc hàng loạt."""
    csv_bytes = get_batch_sample_csv_bytes(limit=limit)
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="mau_ho_so_sinh_vien.csv"'},
    )


@app.get("/api/test-samples")
async def api_test_samples(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=15, ge=5, le=100),
    status: str = Query(default="ALL"),
    tier: str = Query(default="ALL"),
    course: Optional[int] = Query(default=None),
    search: str = Query(default=""),
    sort: str = Query(default="prob_desc"),
) -> Dict[str, Any]:
    return get_test_samples(
        page=page,
        page_size=page_size,
        status_filter=status,
        tier_filter=tier,
        course_filter=course,
        search=search,
        sort_by=sort,
    )


@app.get("/api/test-samples/{student_index}")
async def api_test_sample_detail(student_index: int) -> Dict[str, Any]:
    try:
        return get_test_sample_detail(student_index)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/experiments")
async def api_experiments() -> Dict[str, Any]:
    return get_experiments_data()


if __name__ == "__main__":
    print("=" * 72)
    print("  EduGuard AI — Hệ thống Cảnh báo sớm Sinh viên Bỏ học (FIT4013)")
    print("  Đang khởi động máy chủ Web tại: http://127.0.0.1:8000")
    print("=" * 72)
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
