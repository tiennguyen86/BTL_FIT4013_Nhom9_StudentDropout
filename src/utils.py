# -*- coding: utf-8 -*-
"""
Module: src/utils.py
Chức năng: Quản lý đường dẫn dự án, cố định hạt giống ngẫu nhiên (random_state=42),
từ điển nhãn tiếng Việt và hàm phân tầng rủi ro bỏ học 3 mức.
"""

from pathlib import Path
import random
from typing import Dict, Tuple
import numpy as np

# ==============================================================================
# 1. CỐ ĐỊNH HẠT GIỐNG NGẪU NHIÊN (REPRODUCIBILITY)
# ==============================================================================
RANDOM_STATE: int = 42


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Cố định seed cho thư viện random và numpy để tái lập 100% kết quả thực nghiệm."""
    random.seed(seed)
    np.random.seed(seed)


# ==============================================================================
# 2. QUẢN LÝ ĐƯỜNG DẪN DỰ ÁN (TƯƠNG THÍCH MỌI NƠI GỌI LỆNH)
# ==============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = PROJECT_ROOT / "data" / "data.csv"

MODELS_DIR: Path = PROJECT_ROOT / "models"
BEST_MODEL_PATH: Path = MODELS_DIR / "best_dropout_pipeline.pkl"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "Hinhve"

RESULTS_CSV_PATH: Path = OUTPUTS_DIR / "results_comparison.csv"
TUNING_CSV_PATH: Path = OUTPUTS_DIR / "tuning_results.csv"
ERROR_CSV_PATH: Path = OUTPUTS_DIR / "error_analysis.csv"


def ensure_directories() -> None:
    """Tự động kiểm tra và tạo các thư mục lưu trữ mô hình, bảng CSV và biểu đồ."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 3. TỪ ĐIỂN NHÃN TIẾNG VIỆT (PHỤC VỤ VẼ BIỂU ĐỒ & GIAO DIỆN DEMO)
# ==============================================================================
FEATURE_NAMES_VI: Dict[str, str] = {
    "Marital status": "Tình trạng hôn nhân",
    "Application mode": "Phương thức xét tuyển",
    "Application order": "Thứ tự nguyện vọng",
    "Course": "Ngành đào tạo",
    "Daytime/evening attendance": "Hệ học (Ban ngày/Buổi tối)",
    "Previous qualification": "Trình độ trước đại học",
    "Previous qualification (grade)": "Điểm trình độ trước ĐH",
    "Nacionality": "Quốc tịch",
    "Mother's qualification": "Trình độ học vấn của Mẹ",
    "Father's qualification": "Trình độ học vấn của Cha",
    "Mother's occupation": "Nghề nghiệp của Mẹ",
    "Father's occupation": "Nghề nghiệp của Cha",
    "Admission grade": "Điểm xét tuyển đầu vào",
    "Displaced": "Sinh viên xa nhà",
    "Educational special needs": "Nhu cầu giáo dục đặc biệt",
    "Debtor": "Tình trạng nợ học phí (Debtor)",
    "Tuition fees up to date": "Đóng học phí đúng hạn",
    "Gender": "Giới tính (1=Nam, 0=Nữ)",
    "Scholarship holder": "Sinh viên nhận học bổng",
    "Age at enrollment": "Tuổi khi nhập học",
    "International": "Sinh viên quốc tế",
    "Curricular units 1st sem (credited)": "Số môn được miễn HK1",
    "Curricular units 1st sem (enrolled)": "Số môn đăng ký HK1",
    "Curricular units 1st sem (evaluations)": "Số lượt dự thi HK1",
    "Curricular units 1st sem (approved)": "Số môn đậu HK1",
    "Curricular units 1st sem (grade)": "Điểm trung bình HK1",
    "Curricular units 1st sem (without evaluations)": "Số môn bỏ thi HK1",
    "Curricular units 2nd sem (credited)": "Số môn được miễn HK2",
    "Curricular units 2nd sem (enrolled)": "Số môn đăng ký HK2",
    "Curricular units 2nd sem (evaluations)": "Số lượt dự thi HK2",
    "Curricular units 2nd sem (approved)": "Số môn đậu HK2",
    "Curricular units 2nd sem (grade)": "Điểm trung bình HK2",
    "Curricular units 2nd sem (without evaluations)": "Số môn bỏ thi HK2",
    "Unemployment rate": "Tỷ lệ thất nghiệp (%)",
    "Inflation rate": "Tỷ lệ lạm phát (%)",
    "GDP": "Tăng trưởng GDP (%)",
    # 7 đặc trưng mới tạo ở Luồng B (Pipeline cải tiến)
    "Approval_rate_1st_sem": "Tỷ lệ qua môn HK1",
    "Approval_rate_2nd_sem": "Tỷ lệ qua môn HK2",
    "Grade_diff_2nd_1st": "Chênh lệch điểm HK2 - HK1",
    "Avg_grade_both_sem": "Điểm TB chung cả năm nhất",
    "Total_approved_units": "Tổng số môn đậu năm nhất",
    "Total_without_eval": "Tổng số môn bỏ thi năm nhất",
    "Financial_risk_score": "Chỉ số rủi ro tài chính",
}


def get_vi_name(col_name: str) -> str:
    """Trả về tên tiếng Việt của thuộc tính nếu có, ngược lại giữ nguyên tên gốc."""
    return FEATURE_NAMES_VI.get(col_name, col_name)


# ==============================================================================
# 4. HÀM PHÂN TẦNG RỦI RO BỎ HỌC 3 MỨC (THREE-TIER RISK STRATIFICATION)
# ==============================================================================
RISK_THRESHOLDS: Tuple[float, float] = (0.40, 0.70)


def assign_risk_tier(prob: float) -> str:
    """
    Phân tầng nguy cơ bỏ học dựa trên xác suất dự đoán P(Dropout = 1):
    - < 0.40       : Thấp (Theo dõi định kỳ)
    - 0.40 - 0.699 : Trung bình (Cố vấn học tập nhắc nhở sớm)
    - >= 0.70      : Cao (Can thiệp khẩn cấp học vụ & tài chính)
    """
    low_th, high_th = RISK_THRESHOLDS
    if prob >= high_th:
        return "Cao (>= 70%)"
    elif prob >= low_th:
        return "Trung bình (40% - 70%)"
    return "Thấp (< 40%)"


def get_risk_recommendation(prob: float) -> str:
    """Đưa ra khuyến nghị hành động cụ thể cho Cố vấn học tập ứng với từng tầng rủi ro."""
    low_th, high_th = RISK_THRESHOLDS
    if prob >= high_th:
        return (
            "CẢNH BÁO ĐỎ: Liên hệ trực tiếp sinh viên và gia đình; rà soát nợ học phí, "
            "xem xét hỗ trợ gia hạn học phí và kèm cặp học tập bắt buộc."
        )
    elif prob >= low_th:
        return (
            "CẢNH BÁO VÀNG: Cố vấn học tập gặp mặt đầu học kỳ tiếp theo; "
            "tư vấn điều chỉnh khối lượng tín chỉ đăng ký vừa sức."
        )
    return "AN TOÀN (XANH): Duy trì theo dõi tự động cuối mỗi học kỳ; khuyến khích xét học bổng."
