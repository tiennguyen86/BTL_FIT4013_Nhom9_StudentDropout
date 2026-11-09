# -*- coding: utf-8 -*-
"""
Module: src/preprocessing.py
Chức năng:
1. Đọc và làm sạch bộ dữ liệu gốc UCI (data/data.csv).
2. Thống kê khám phá dữ liệu (EDA): kiểm tra thiếu, trùng lặp, phân phối nhãn và ngoại lệ (IQR).
3. Chia tập dữ liệu Train/Test (80/20 Stratified, random_state=42) chống rò rỉ dữ liệu.
4. Xây dựng 2 luồng tiền xử lý chuẩn scikit-learn:
   - Luồng A (Pipeline cơ bản): Giữ 36 biến gốc + StandardScaler.
   - Luồng B (Pipeline cải tiến): Tạo 7 đặc trưng miền giáo dục + OneHotEncoder (9 biến định danh) + StandardScaler.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import DATA_PATH, RANDOM_STATE

# ==============================================================================
# 1. DANH SÁCH CÁC CỘT ĐẶC TRƯNG THEO BẢN CHẤT DỮ LIỆU
# ==============================================================================
# 9 thuộc tính phân loại định danh (Nominal) đang lưu dưới dạng mã số nguyên
NOMINAL_COLS: List[str] = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# 7 đặc trưng mới được kiến tạo trong Luồng B dựa trên tri thức miền giáo dục
NEW_ENGINEERED_COLS: List[str] = [
    "Approval_rate_1st_sem",
    "Approval_rate_2nd_sem",
    "Grade_diff_2nd_1st",
    "Avg_grade_both_sem",
    "Total_approved_units",
    "Total_without_eval",
    "Financial_risk_score",
]


# ==============================================================================
# 2. ĐỌC DỮ LIỆU, LÀM SẠCH & TÓM TẮT EDA
# ==============================================================================
def load_and_clean_data(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Đọc file data.csv gốc từ UCI (phân cách bằng dấu ';'),
    loại bỏ ký tự ẩn BOM (\\ufeff) và khoảng trắng/tab thừa ở tên cột,
    đồng thời tạo cột nhãn nhị phân Target_Binary (Dropout = 1, còn lại = 0).
    """
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    # Gán nhãn nhị phân: Dropout = 1 (Bỏ học), Graduate / Enrolled = 0 (Không bỏ học)
    df["Target_Binary"] = (df["Target"] == "Dropout").astype(int)
    return df


def get_outlier_summary_iqr(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Thống kê mô tả và phát hiện ngoại lệ theo quy tắc 1.5 * IQR cho các biến định lượng.
    Phục vụ phân tích EDA ở Chương 1 và Notebook.
    """
    rows = []
    n_total = len(df)
    for col in numeric_cols:
        s = df[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lower) | (s > upper)).sum())
        corr_val = float(s.corr(df["Target_Binary"])) if "Target_Binary" in df.columns else np.nan
        rows.append({
            "Feature": col,
            "Mean": round(float(s.mean()), 2),
            "Std": round(float(s.std()), 2),
            "Median": round(float(s.median()), 2),
            "Min": round(float(s.min()), 2),
            "Max": round(float(s.max()), 2),
            "IQR": round(float(iqr), 2),
            "Outliers_Count": n_out,
            "Outliers_Pct": round(n_out / n_total * 100, 2),
            "Corr_With_Dropout": round(corr_val, 3),
        })
    return pd.DataFrame(rows)


def print_eda_overview(df: pd.DataFrame) -> None:
    """In báo cáo tóm tắt khám phá dữ liệu (EDA) ra màn hình console/notebook."""
    print("=" * 72)
    print("TÓM TẮT KHÁM PHÁ DỮ LIỆU (EDA) - BỘ DỮ LIỆU SINH VIÊN UCI")
    print("=" * 72)
    print(f"- Kích thước dữ liệu gốc : {df.shape[0]} dòng x {df.shape[1] - 1} cột (36 đặc trưng + 1 cột nhãn)")
    print(f"- Tổng số giá trị thiếu  : {int(df.isnull().sum().sum())} ô")
    print(f"- Số dòng trùng lặp      : {int(df.duplicated().sum())} dòng")
    print("- Phân phối 3 nhãn gốc   :")
    for label, cnt in df["Target"].value_counts().items():
        print(f"    + {label:<10}: {cnt:4d} sinh viên ({cnt / len(df) * 100:.2f}%)")
    n_drop = int(df["Target_Binary"].sum())
    n_non = len(df) - n_drop
    print(f"- Phân phối nhị phân     : Dropout (1) = {n_drop} ({n_drop / len(df) * 100:.2f}%) | Non-Dropout (0) = {n_non} ({n_non / len(df) * 100:.2f}%)")
    print("=" * 72)


# ==============================================================================
# 3. CHIA TẬP HUẤN LUYỆN & KIỂM THỬ (STRATIFIED SPLIT)
# ==============================================================================
def split_train_test(
    df: pd.DataFrame, test_size: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Chia dữ liệu thành tập Train (80%) và Test (20%) có phân tầng theo nhãn nhị phân.
    Đảm bảo tách trước khi fit bất kỳ bộ biến đổi nào để chống data leakage.
    """
    feature_cols = [c for c in df.columns if c not in ("Target", "Target_Binary")]
    X = df[feature_cols].copy()
    y = df["Target_Binary"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


# ==============================================================================
# 4. THIẾT KẾ LUỒNG A (CƠ BẢN) VÀ LUỒNG B (CẢI TIẾN)
# ==============================================================================
class DomainFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Bộ biến đổi tự tạo 7 đặc trưng mới dựa trên kiến thức miền giáo dục cho Luồng B:
    1. Approval_rate_1st_sem : Tỷ lệ môn đậu trên số môn đăng ký ở Học kỳ 1.
    2. Approval_rate_2nd_sem : Tỷ lệ môn đậu trên số môn đăng ký ở Học kỳ 2.
    3. Grade_diff_2nd_1st    : Chênh lệch điểm trung bình giữa Học kỳ 2 và Học kỳ 1.
    4. Avg_grade_both_sem    : Điểm trung bình chung của cả năm học thứ nhất.
    5. Total_approved_units  : Tổng số học phần tích lũy thành công trong năm nhất.
    6. Total_without_eval    : Tổng số học phần đăng ký nhưng bỏ thi (không có điểm).
    7. Financial_risk_score  : Chỉ số áp lực tài chính = Debtor + (1 - Tuition_up_to_date) - Scholarship.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        d = X.copy()
        d["Approval_rate_1st_sem"] = np.where(
            d["Curricular units 1st sem (enrolled)"] > 0,
            d["Curricular units 1st sem (approved)"] / d["Curricular units 1st sem (enrolled)"],
            0.0,
        )
        d["Approval_rate_2nd_sem"] = np.where(
            d["Curricular units 2nd sem (enrolled)"] > 0,
            d["Curricular units 2nd sem (approved)"] / d["Curricular units 2nd sem (enrolled)"],
            0.0,
        )
        d["Grade_diff_2nd_1st"] = (
            d["Curricular units 2nd sem (grade)"] - d["Curricular units 1st sem (grade)"]
        )
        d["Avg_grade_both_sem"] = 0.5 * (
            d["Curricular units 1st sem (grade)"] + d["Curricular units 2nd sem (grade)"]
        )
        d["Total_approved_units"] = (
            d["Curricular units 1st sem (approved)"] + d["Curricular units 2nd sem (approved)"]
        )
        d["Total_without_eval"] = (
            d["Curricular units 1st sem (without evaluations)"]
            + d["Curricular units 2nd sem (without evaluations)"]
        )
        d["Financial_risk_score"] = (
            d["Debtor"] + (1 - d["Tuition fees up to date"]) - d["Scholarship holder"]
        )
        return d


def build_preprocessor_flow_a() -> Pipeline:
    """
    Khởi tạo Luồng A (Pipeline cơ bản):
    Chỉ chuẩn hóa Z-score (StandardScaler) trực tiếp trên 36 cột dữ liệu gốc.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
    ])


def build_preprocessor_flow_b(raw_columns: List[str]) -> Pipeline:
    """
    Khởi tạo Luồng B (Pipeline cải tiến):
    - Bước 1: Tạo thêm 7 đặc trưng miền giáo dục bằng DomainFeatureCreator.
    - Bước 2: Mã hóa One-Hot cho 9 biến phân loại định danh (gộp nhóm hiếm < 1%)
              và chuẩn hóa StandardScaler cho 34 biến số/nhị phân còn lại.
    """
    num_cols_b = [c for c in raw_columns if c not in NOMINAL_COLS] + NEW_ENGINEERED_COLS

    col_transformer = ColumnTransformer(
        transformers=[
            (
                "nom",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False),
                NOMINAL_COLS,
            ),
            ("num", StandardScaler(), num_cols_b),
        ],
        verbose_feature_names_out=False,
    )

    return Pipeline([
        ("feature_creator", DomainFeatureCreator()),
        ("column_transformer", col_transformer),
    ])


def get_flow_b_feature_names(fitted_preprocessor_b: Pipeline) -> List[str]:
    """Lấy danh sách tên các cột đặc trưng sau khi qua Luồng B (phục vụ vẽ Feature Importance)."""
    ct: ColumnTransformer = fitted_preprocessor_b.named_steps["column_transformer"]
    return list(ct.get_feature_names_out())
