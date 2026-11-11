# -*- coding: utf-8 -*-
"""
Module: src/web_service.py
Chức năng: Lớp dịch vụ dữ liệu và suy luận ML phục vụ giao diện FastAPI Web Dashboard.
Tái sử dụng hoàn toàn pipeline từ src/preprocessing.py, src/train_evaluate.py và src/utils.py.
"""

from __future__ import annotations

import io
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import src.preprocessing as prep
from src.domain_labels import (
    APPLICATION_MODE_LABELS,
    ARCHETYPE_PRESETS,
    COURSE_LABELS,
    FIGURE_METADATA,
    MARITAL_STATUS_LABELS,
    PREV_QUAL_LABELS,
)
from src.train_evaluate import run_experiment_pipeline
from src.utils import (
    BEST_MODEL_PATH,
    DATA_PATH,
    ERROR_CSV_PATH,
    FEATURE_NAMES_VI,
    RESULTS_CSV_PATH,
    RISK_THRESHOLDS,
    TUNING_CSV_PATH,
    assign_risk_tier,
    get_risk_recommendation,
    get_vi_name,
)


class AppResources:
    """Chứa toàn bộ dữ liệu, mô hình đã nạp và thống kê tiền tính toán cho Web API."""

    def __init__(self) -> None:
        self.df: pd.DataFrame = prep.load_and_clean_data(DATA_PATH)
        self.feature_cols: List[str] = [
            c for c in self.df.columns if c not in ("Target", "Target_Binary")
        ]
        self.X_train, self.X_test, self.y_train, self.y_test = prep.split_train_test(self.df)
        self.model = self._load_or_retrain_model()

        # Lưu trung vị gốc để điền giá trị mặc định cho các trường không truyền lên
        self.medians: Dict[str, float] = {
            k: float(v) for k, v in self.X_test.median(numeric_only=True).to_dict().items()
        }

        # Đọc các bảng thực nghiệm và phân tích lỗi
        if not RESULTS_CSV_PATH.exists() or not TUNING_CSV_PATH.exists() or not ERROR_CSV_PATH.exists():
            run_experiment_pipeline()

        self.results_df = pd.read_csv(RESULTS_CSV_PATH)
        self.tuning_df = pd.read_csv(TUNING_CSV_PATH)
        self.error_df = pd.read_csv(ERROR_CSV_PATH)
        # Gắn index gốc từ X_test để đối chiếu 1-1
        self.error_df["Student_Index"] = list(self.X_test.index)

        # Thống kê đối chuẩn theo nhóm Dropout vs Non-Dropout
        self.group_benchmarks = self._compute_group_benchmarks()

    def _load_or_retrain_model(self):
        """Nạp file .pkl, tự động huấn luyện lại nếu thiếu hoặc xung đột phiên bản thư viện."""
        if BEST_MODEL_PATH.exists():
            try:
                loaded = joblib.load(BEST_MODEL_PATH)
                _ = loaded.predict_proba(self.X_test.iloc[:2])
                return loaded
            except Exception as exc:
                print(f"[WARN] Không thể nạp {BEST_MODEL_PATH} ({exc}). Đang huấn luyện lại pipeline...")
        run_experiment_pipeline()
        return joblib.load(BEST_MODEL_PATH)

    def _compute_group_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Tính giá trị trung bình của nhóm An toàn (0) và Bỏ học (1) trên tập dữ liệu."""
        engineered_all = prep.DomainFeatureCreator().transform(self.df[self.feature_cols])
        engineered_all["Target_Binary"] = self.df["Target_Binary"].values
        cols = [
            "Approval_rate_1st_sem",
            "Approval_rate_2nd_sem",
            "Avg_grade_both_sem",
            "Total_approved_units",
            "Total_without_eval",
            "Financial_risk_score",
            "Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (grade)",
            "Tuition fees up to date",
            "Debtor",
            "Scholarship holder",
            "Age at enrollment",
        ]
        summary: Dict[str, Dict[str, float]] = {"non_dropout": {}, "dropout": {}}
        non_df = engineered_all[engineered_all["Target_Binary"] == 0]
        drop_df = engineered_all[engineered_all["Target_Binary"] == 1]
        for c in cols:
            summary["non_dropout"][c] = round(float(non_df[c].mean()), 3)
            summary["dropout"][c] = round(float(drop_df[c].mean()), 3)
        return summary


@lru_cache(maxsize=1)
def get_resources() -> AppResources:
    return AppResources()


def _label_options(codes: List[int], mapping: Dict[int, str], prefix: str = "Mã") -> List[Dict[str, Any]]:
    return [
        {
            "value": int(code),
            "label": f"{mapping.get(int(code), f'{prefix} {code}')} (#{int(code)})",
            "short_label": mapping.get(int(code), f"{prefix} {code}"),
        }
        for code in sorted(set(int(x) for x in codes))
    ]


def get_metadata() -> Dict[str, Any]:
    """Trả về cấu hình dropdown, giá trị mặc định, miền giá trị và các bộ hồ sơ mẫu."""
    res = get_resources()
    df = res.df

    ranges = {}
    range_cols = [
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)",
        "Age at enrollment",
        "Admission grade",
        "Previous qualification (grade)",
        "Unemployment rate",
        "Inflation rate",
        "GDP",
        "Application order",
    ]
    for col in range_cols:
        s = df[col]
        ranges[col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
        }

    options = {
        "Course": _label_options(df["Course"].unique().tolist(), COURSE_LABELS, "Ngành"),
        "Marital status": _label_options(df["Marital status"].unique().tolist(), MARITAL_STATUS_LABELS, "TT"),
        "Application mode": _label_options(df["Application mode"].unique().tolist(), APPLICATION_MODE_LABELS, "PT"),
        "Previous qualification": _label_options(df["Previous qualification"].unique().tolist(), PREV_QUAL_LABELS, "TĐ"),
    }

    return {
        "defaults": res.medians,
        "ranges": ranges,
        "options": options,
        "presets": ARCHETYPE_PRESETS,
        "feature_labels": FEATURE_NAMES_VI,
        "feature_cols": res.feature_cols,
        "risk_thresholds": {
            "low_max": RISK_THRESHOLDS[0],
            "high_min": RISK_THRESHOLDS[1],
        },
        "benchmarks": res.group_benchmarks,
    }


def _compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = len(y_true)
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
    }


def get_overview_data() -> Dict[str, Any]:
    """Tổng hợp toàn bộ KPI, EDA, tầm quan trọng đặc trưng và phân tầng rủi ro cho Dashboard."""
    res = get_resources()
    df = res.df
    err_df = res.error_df

    n_total = int(len(df))
    n_grad = int((df["Target"] == "Graduate").sum())
    n_drop = int((df["Target"] == "Dropout").sum())
    n_enr = int((df["Target"] == "Enrolled").sum())

    # Tầm quan trọng đặc trưng từ Random Forest (Luồng B)
    preproc_b = res.model.named_steps["preprocessor"]
    rf_clf = res.model.named_steps["classifier"]
    feat_names = prep.get_flow_b_feature_names(preproc_b)
    importances = rf_clf.feature_importances_
    top_idx = np.argsort(importances)[-12:][::-1]
    feature_importance = [
        {
            "feature": feat_names[i],
            "label": get_vi_name(feat_names[i]),
            "importance": round(float(importances[i]), 4),
            "is_engineered": feat_names[i] in prep.NEW_ENGINEERED_COLS,
        }
        for i in top_idx
    ]

    # Top tương quan Pearson với Target_Binary
    corr_s = df.drop(columns=["Target", "Target_Binary"]).corrwith(df["Target_Binary"]).dropna().sort_values()
    top_corr = pd.concat([corr_s.head(5), corr_s.tail(5)])
    correlations = [
        {
            "feature": k,
            "label": get_vi_name(k),
            "corr": round(float(v), 4),
        }
        for k, v in top_corr.items()
    ]

    # Tác động tài chính & học bổng
    financial_impact = [
        {
            "label": "Chưa đóng học phí",
            "dropout_rate": round(float(df[df["Tuition fees up to date"] == 0]["Target_Binary"].mean() * 100), 2),
            "count": int((df["Tuition fees up to date"] == 0).sum()),
            "tone": "danger",
        },
        {
            "label": "Đã đóng học phí",
            "dropout_rate": round(float(df[df["Tuition fees up to date"] == 1]["Target_Binary"].mean() * 100), 2),
            "count": int((df["Tuition fees up to date"] == 1).sum()),
            "tone": "safe",
        },
        {
            "label": "Đang nợ học phí (Debtor)",
            "dropout_rate": round(float(df[df["Debtor"] == 1]["Target_Binary"].mean() * 100), 2),
            "count": int((df["Debtor"] == 1).sum()),
            "tone": "warning",
        },
        {
            "label": "Không nợ học phí",
            "dropout_rate": round(float(df[df["Debtor"] == 0]["Target_Binary"].mean() * 100), 2),
            "count": int((df["Debtor"] == 0).sum()),
            "tone": "safe",
        },
        {
            "label": "Không có học bổng",
            "dropout_rate": round(float(df[df["Scholarship holder"] == 0]["Target_Binary"].mean() * 100), 2),
            "count": int((df["Scholarship holder"] == 0).sum()),
            "tone": "warning",
        },
        {
            "label": "Có học bổng",
            "dropout_rate": round(float(df[df["Scholarship holder"] == 1]["Target_Binary"].mean() * 100), 2),
            "count": int((df["Scholarship holder"] == 1).sum()),
            "tone": "info",
        },
    ]

    # Phân tầng rủi ro trên tập kiểm thử N = 885
    tier_order = ["Thấp (< 40%)", "Trung bình (40% - 70%)", "Cao (>= 70%)"]
    risk_tiers = []
    for t in tier_order:
        sub = err_df[err_df["Risk_Tier"] == t]
        cnt = int(len(sub))
        actual_drop = int(sub["Actual_Dropout"].sum()) if cnt else 0
        rate = round(actual_drop / cnt * 100, 2) if cnt else 0.0
        risk_tiers.append({
            "tier": t,
            "count": cnt,
            "pct_of_test": round(cnt / len(err_df) * 100, 2) if len(err_df) else 0.0,
            "actual_dropout": actual_drop,
            "actual_dropout_rate": rate,
        })

    # Ma trận nhầm lẫn tại 2 ngưỡng 0.50 và 0.40
    y_true = err_df["Actual_Dropout"].to_numpy()
    y_prob = err_df["Dropout_Probability"].to_numpy()
    cm_050 = _compute_metrics_at_threshold(y_true, y_prob, 0.50)
    cm_040 = _compute_metrics_at_threshold(y_true, y_prob, 0.40)

    # Tỷ lệ bỏ học theo top 8 ngành đào tạo đông sinh viên nhất
    course_stats = []
    for code, grp in df.groupby("Course"):
        c_int = int(code)
        course_stats.append({
            "code": c_int,
            "name": COURSE_LABELS.get(c_int, f"Ngành {c_int}"),
            "total": int(len(grp)),
            "dropout_rate": round(float(grp["Target_Binary"].mean() * 100), 2),
        })
    course_stats.sort(key=lambda x: x["total"], reverse=True)

    # Kết quả mô hình tốt nhất
    best_row = res.results_df[
        (res.results_df["Flow"].str.contains("Luồng B")) &
        (res.results_df["Model"].str.contains("Random Forest"))
    ].iloc[0].to_dict()

    return {
        "project": {
            "title": "Hệ thống Cảnh báo sớm Sinh viên có Nguy cơ Bỏ học",
            "subtitle": "Học phần Nhập môn Học máy (FIT4013) — Nhóm 9",
            "instructor": "ThS. Lê Thị Thuỳ Trang",
            "active_model": "Random Forest (Luồng B - Cải tiến)",
            "members": [
                {"id": "1871020569", "name": "Nguyễn Hữu Tiến", "class": "CNTT 18-10", "role": "Trưởng nhóm • Pipeline Luồng A/B & Web Demo"},
                {"id": "1871020502", "name": "Vương Xuân Sơn", "class": "CNTT 18-10", "role": "EDA, Xử lý ngoại lệ IQR, K-NN & Naive Bayes"},
                {"id": "1871020698", "name": "Lý Triệu Trung", "class": "CNTT 18-10", "role": "Decision Tree, Random Forest, Phân tích lỗi & Rủi ro"},
            ],
        },
        "dataset": {
            "total": n_total,
            "graduate": n_grad,
            "graduate_pct": round(n_grad / n_total * 100, 2),
            "dropout": n_drop,
            "dropout_pct": round(n_drop / n_total * 100, 2),
            "enrolled": n_enr,
            "enrolled_pct": round(n_enr / n_total * 100, 2),
            "non_dropout": n_grad + n_enr,
            "non_dropout_pct": round((n_grad + n_enr) / n_total * 100, 2),
            "train_size": int(len(res.X_train)),
            "test_size": int(len(res.X_test)),
            "raw_features": len(res.feature_cols),
            "engineered_features": len(prep.NEW_ENGINEERED_COLS),
            "flow_b_dimensions": len(feat_names),
        },
        "best_model_metrics": best_row,
        "feature_importance": feature_importance,
        "correlations": correlations,
        "financial_impact": financial_impact,
        "risk_tiers": risk_tiers,
        "confusion_matrices": {
            "0.50": cm_050,
            "0.40": cm_040,
        },
        "course_stats": course_stats[:8],
    }


def _explain_student_factors(engineered_row: pd.Series, res: AppResources) -> Dict[str, List[Dict[str, Any]]]:
    """Phân tích định lượng các yếu tố tăng rủi ro và yếu tố bảo vệ của hồ sơ sinh viên."""
    risk_drivers: List[Dict[str, Any]] = []
    protective_factors: List[Dict[str, Any]] = []

    app_rate_2 = float(engineered_row["Approval_rate_2nd_sem"])
    app_rate_1 = float(engineered_row["Approval_rate_1st_sem"])
    avg_grade = float(engineered_row["Avg_grade_both_sem"])
    tuition_ok = int(engineered_row["Tuition fees up to date"])
    debtor = int(engineered_row["Debtor"])
    scholarship = int(engineered_row["Scholarship holder"])
    no_eval = int(engineered_row["Total_without_eval"])
    total_app = int(engineered_row["Total_approved_units"])
    age = int(engineered_row["Age at enrollment"])
    grade_diff = float(engineered_row["Grade_diff_2nd_1st"])

    bench_safe = res.group_benchmarks["non_dropout"]
    bench_drop = res.group_benchmarks["dropout"]

    # Học phí & tài chính
    if tuition_ok == 0:
        risk_drivers.append({
            "title": "Chưa đóng học phí đúng hạn",
            "detail": "Nhóm chưa hoàn thành học phí có tỷ lệ bỏ học lên tới 86.6% trong tập dữ liệu.",
            "severity": "high",
            "value": "Chưa đóng (0)",
        })
    else:
        protective_factors.append({
            "title": "Đóng học phí đúng hạn",
            "detail": "Trạng thái tài chính ổn định giúp giảm đáng kể nguy cơ gián đoạn học tập.",
            "value": "Đúng hạn (1)",
        })

    if debtor == 1:
        risk_drivers.append({
            "title": "Đang có công nợ học phí (Debtor = 1)",
            "detail": "Áp lực tài chính là tín hiệu cảnh báo đỏ hàng đầu cần cố vấn hỗ trợ sớm.",
            "severity": "high",
            "value": "Đang nợ (1)",
        })

    if scholarship == 1:
        protective_factors.append({
            "title": "Sinh viên nhận học bổng",
            "detail": f"Tỷ lệ bỏ học ở nhóm có học bổng thấp hơn rất nhiều so với nhóm không học bổng.",
            "value": "Có học bổng (1)",
        })

    # Kết quả học tập HK2 & HK1
    if app_rate_2 < 0.5:
        risk_drivers.append({
            "title": "Tỷ lệ qua môn HK2 rất thấp",
            "detail": f"Đạt {app_rate_2*100:.1f}% (Trung bình nhóm An toàn là {bench_safe['Approval_rate_2nd_sem']*100:.1f}%).",
            "severity": "high" if app_rate_2 < 0.3 else "medium",
            "value": f"{app_rate_2*100:.1f}%",
        })
    elif app_rate_2 >= 0.8:
        protective_factors.append({
            "title": "Tỷ lệ qua môn HK2 tích cực",
            "detail": f"Hoàn thành {app_rate_2*100:.1f}% học phần đăng ký ở học kỳ 2.",
            "value": f"{app_rate_2*100:.1f}%",
        })

    if app_rate_1 < 0.5:
        risk_drivers.append({
            "title": "Tỷ lệ qua môn HK1 dưới 50%",
            "detail": f"Đạt {app_rate_1*100:.1f}% (Nhóm Bỏ học trung bình {bench_drop['Approval_rate_1st_sem']*100:.1f}%).",
            "severity": "medium",
            "value": f"{app_rate_1*100:.1f}%",
        })

    if avg_grade < 10.0:
        risk_drivers.append({
            "title": "Điểm trung bình năm nhất thấp",
            "detail": f"Điểm TB cả 2 kỳ đạt {avg_grade:.2f}/20 (Nhóm An toàn trung bình {bench_safe['Avg_grade_both_sem']:.2f}).",
            "severity": "high" if avg_grade < 7.0 else "medium",
            "value": f"{avg_grade:.2f}/20",
        })
    elif avg_grade >= 12.5:
        protective_factors.append({
            "title": "Điểm trung bình duy trì tốt",
            "detail": f"Đạt {avg_grade:.2f}/20 — cao hơn mức trung bình của nhóm hoàn thành chương trình.",
            "value": f"{avg_grade:.2f}/20",
        })

    if grade_diff <= -2.0:
        risk_drivers.append({
            "title": "Điểm số sụt giảm mạnh ở HK2",
            "detail": f"Chênh lệch điểm HK2 - HK1 là {grade_diff:+.2f} điểm.",
            "severity": "medium",
            "value": f"{grade_diff:+.2f}",
        })
    elif grade_diff >= 1.0:
        protective_factors.append({
            "title": "Điểm số cải thiện ở HK2",
            "detail": f"Tăng {grade_diff:+.2f} điểm so với học kỳ 1.",
            "value": f"{grade_diff:+.2f}",
        })

    if no_eval >= 1:
        risk_drivers.append({
            "title": "Có môn đăng ký nhưng bỏ thi",
            "detail": f"Tổng cộng {no_eval} môn không dự thi trong năm nhất — dấu hiệu mất động lực học tập.",
            "severity": "medium" if no_eval == 1 else "high",
            "value": f"{no_eval} môn",
        })

    if total_app >= 10:
        protective_factors.append({
            "title": "Tích lũy tín chỉ ổn định",
            "detail": f"Tổng số môn đậu cả 2 học kỳ đạt {total_app} học phần.",
            "value": f"{total_app} môn",
        })

    if age >= 25:
        risk_drivers.append({
            "title": "Độ tuổi nhập học cao (>= 25 tuổi)",
            "detail": f"Nhập học ở tuổi {age} thường chịu thêm áp lực công việc/gia đình.",
            "severity": "low",
            "value": f"{age} tuổi",
        })

    return {
        "risk_drivers": risk_drivers[:5],
        "protective_factors": protective_factors[:5],
    }


def predict_single(features: Dict[str, Any], threshold: float = 0.50) -> Dict[str, Any]:
    """Dự đoán xác suất bỏ học cho 1 hồ sơ sinh viên và trả về đầy đủ giải thích Luồng B."""
    res = get_resources()

    row_dict = dict(res.medians)
    for k, v in features.items():
        if k in row_dict and v is not None:
            row_dict[k] = float(v)

    input_df = pd.DataFrame([row_dict])[res.feature_cols]
    prob = float(res.model.predict_proba(input_df)[0, 1])
    pred_label = int(prob >= threshold)
    tier = assign_risk_tier(prob)
    rec = get_risk_recommendation(prob)

    engineered_df = prep.DomainFeatureCreator().transform(input_df)
    eng_row = engineered_df.iloc[0]

    engineered_summary = [
        {
            "key": col,
            "label": get_vi_name(col),
            "value": round(float(eng_row[col]), 3),
            "safe_avg": res.group_benchmarks["non_dropout"].get(col),
            "drop_avg": res.group_benchmarks["dropout"].get(col),
        }
        for col in prep.NEW_ENGINEERED_COLS
    ]

    explanation = _explain_student_factors(eng_row, res)

    return {
        "probability": round(prob, 4),
        "probability_pct": round(prob * 100, 2),
        "threshold": threshold,
        "predicted_label": pred_label,
        "predicted_text": "Dropout (1) — Nguy cơ Bỏ học" if pred_label == 1 else "Non-Dropout (0) — Tiếp tục học",
        "risk_tier": tier,
        "tier_code": "high" if prob >= RISK_THRESHOLDS[1] else ("medium" if prob >= RISK_THRESHOLDS[0] else "low"),
        "recommendation": rec,
        "engineered_features": engineered_summary,
        "explanation": explanation,
        "input_used": {k: round(float(row_dict[k]), 4) for k in res.feature_cols},
    }


def predict_batch_csv(file_bytes: bytes, threshold: float = 0.50) -> Dict[str, Any]:
    """Dự đoán hàng loạt từ file CSV tải lên (hỗ trợ dấu ';' hoặc ',')."""
    res = get_resources()

    text = file_bytes.decode("utf-8-sig", errors="replace")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    sep = ";" if first_line.count(";") > first_line.count(",") else ","

    df_in = pd.read_csv(io.StringIO(text), sep=sep)
    df_in.columns = [c.strip().replace("\ufeff", "") for c in df_in.columns]

    if df_in.empty:
        raise ValueError("File CSV không có dữ liệu sinh viên.")

    # Chuẩn bị DataFrame đủ 36 cột theo đúng thứ tự của Pipeline
    prepared = pd.DataFrame(index=df_in.index)
    matched_cols = 0
    for col in res.feature_cols:
        if col in df_in.columns:
            prepared[col] = pd.to_numeric(df_in[col], errors="coerce").fillna(res.medians[col])
            matched_cols += 1
        else:
            prepared[col] = res.medians[col]

    if matched_cols < 4:
        raise ValueError(
            "Không nhận diện được các cột dữ liệu UCI trong file CSV (cần ít nhất 4 cột đặc trưng khớp tên)."
        )

    probs = res.model.predict_proba(prepared[res.feature_cols])[:, 1]
    eng_df = prep.DomainFeatureCreator().transform(prepared)

    rows = []
    tier_counts = {"low": 0, "medium": 0, "high": 0}
    pred_dropouts = 0

    for idx in range(len(prepared)):
        p = float(probs[idx])
        pred = int(p >= threshold)
        pred_dropouts += pred
        tier = assign_risk_tier(p)
        tier_code = "high" if p >= RISK_THRESHOLDS[1] else ("medium" if p >= RISK_THRESHOLDS[0] else "low")
        tier_counts[tier_code] += 1

        course_val = int(prepared.iloc[idx]["Course"])
        actual_val = None
        if "Target" in df_in.columns:
            actual_val = str(df_in.iloc[idx]["Target"])
        elif "Actual_Dropout" in df_in.columns:
            actual_val = "Dropout" if int(df_in.iloc[idx]["Actual_Dropout"]) == 1 else "Non-Dropout"

        rows.append({
            "row_no": idx + 1,
            "course": COURSE_LABELS.get(course_val, f"Ngành {course_val}"),
            "age": int(prepared.iloc[idx]["Age at enrollment"]),
            "approved_1st": int(prepared.iloc[idx]["Curricular units 1st sem (approved)"]),
            "approved_2nd": int(prepared.iloc[idx]["Curricular units 2nd sem (approved)"]),
            "avg_grade": round(float(eng_df.iloc[idx]["Avg_grade_both_sem"]), 2),
            "tuition_ok": int(prepared.iloc[idx]["Tuition fees up to date"]),
            "debtor": int(prepared.iloc[idx]["Debtor"]),
            "probability": round(p, 4),
            "probability_pct": round(p * 100, 2),
            "predicted_label": pred,
            "risk_tier": tier,
            "tier_code": tier_code,
            "recommendation": get_risk_recommendation(p),
            "actual": actual_val,
        })

    return {
        "total": len(rows),
        "matched_columns": matched_cols,
        "predicted_dropout_count": pred_dropouts,
        "tier_counts": tier_counts,
        "avg_probability_pct": round(float(np.mean(probs) * 100), 2),
        "rows": rows,
    }


def get_batch_sample_csv_bytes(limit: int = 25) -> bytes:
    """Xuất nhanh 25 hồ sơ từ tập Test ra file CSV mẫu để người dùng tải về hoặc test batch."""
    res = get_resources()
    sample = res.X_test.head(limit).copy()
    sample["Actual_Dropout"] = res.y_test.head(limit).values
    return sample.to_csv(index=False).encode("utf-8-sig")


def get_test_samples(
    page: int = 1,
    page_size: int = 20,
    status_filter: str = "ALL",
    tier_filter: str = "ALL",
    course_filter: Optional[int] = None,
    search: str = "",
    sort_by: str = "prob_desc",
) -> Dict[str, Any]:
    """Truy vấn phân trang toàn bộ 885 sinh viên tập kiểm thử từ outputs/error_analysis.csv."""
    res = get_resources()
    df = res.error_df.copy()

    # Thống kê nhanh trạng thái trước khi lọc
    status_counts = df["Prediction_Status"].value_counts().to_dict()
    tier_counts = df["Risk_Tier"].value_counts().to_dict()

    if status_filter and status_filter != "ALL":
        df = df[df["Prediction_Status"].str.contains(status_filter, case=False, na=False)]

    if tier_filter and tier_filter != "ALL":
        df = df[df["Risk_Tier"] == tier_filter]

    if course_filter is not None:
        df = df[df["Course"] == course_filter]

    if search.strip():
        q = search.strip().lower()
        df["_course_name"] = df["Course"].map(lambda c: COURSE_LABELS.get(int(c), str(c)).lower())
        mask = (
            df["Student_Index"].astype(str).str.contains(q, na=False)
            | df["_course_name"].str.contains(q, na=False)
        )
        df = df[mask]

    if sort_by == "prob_desc":
        df = df.sort_values("Dropout_Probability", ascending=False)
    elif sort_by == "prob_asc":
        df = df.sort_values("Dropout_Probability", ascending=True)
    else:
        df = df.sort_values("Student_Index", ascending=True)

    total_filtered = int(len(df))
    start = max((page - 1) * page_size, 0)
    end = start + page_size
    page_df = df.iloc[start:end]

    items = []
    for _, row in page_df.iterrows():
        prob = float(row["Dropout_Probability"])
        c_code = int(row["Course"])
        items.append({
            "student_index": int(row["Student_Index"]),
            "course_code": c_code,
            "course_name": COURSE_LABELS.get(c_code, f"Ngành {c_code}"),
            "age": int(row["Age at enrollment"]),
            "approved_1st": int(row["Curricular units 1st sem (approved)"]),
            "enrolled_1st": int(row["Curricular units 1st sem (enrolled)"]),
            "grade_1st": round(float(row["Curricular units 1st sem (grade)"]), 2),
            "approved_2nd": int(row["Curricular units 2nd sem (approved)"]),
            "enrolled_2nd": int(row["Curricular units 2nd sem (enrolled)"]),
            "grade_2nd": round(float(row["Curricular units 2nd sem (grade)"]), 2),
            "tuition_ok": int(row["Tuition fees up to date"]),
            "debtor": int(row["Debtor"]),
            "scholarship": int(row["Scholarship holder"]),
            "actual_dropout": int(row["Actual_Dropout"]),
            "predicted_dropout": int(row["Predicted_Dropout"]),
            "probability": round(prob, 4),
            "probability_pct": round(prob * 100, 2),
            "risk_tier": str(row["Risk_Tier"]),
            "tier_code": "high" if prob >= RISK_THRESHOLDS[1] else ("medium" if prob >= RISK_THRESHOLDS[0] else "low"),
            "prediction_status": str(row["Prediction_Status"]),
        })

    return {
        "total": total_filtered,
        "total_all": int(len(res.error_df)),
        "page": page,
        "page_size": page_size,
        "total_pages": max((total_filtered + page_size - 1) // page_size, 1),
        "status_counts": status_counts,
        "tier_counts": tier_counts,
        "items": items,
    }


def get_test_sample_detail(student_index: int) -> Dict[str, Any]:
    """Lấy đầy đủ 36 thuộc tính gốc + phân tích mô hình cho 1 sinh viên cụ thể trong tập Test."""
    res = get_resources()
    matches = res.error_df[res.error_df["Student_Index"] == student_index]
    if matches.empty:
        raise KeyError(f"Không tìm thấy sinh viên #{student_index} trong tập kiểm thử.")

    row = matches.iloc[0]
    raw_features = {col: float(row[col]) for col in res.feature_cols}
    pred_result = predict_single(raw_features, threshold=0.50)

    c_code = int(row["Course"])
    return {
        "student_index": student_index,
        "course_name": COURSE_LABELS.get(c_code, f"Ngành {c_code}"),
        "actual_dropout": int(row["Actual_Dropout"]),
        "actual_text": "Bỏ học (Dropout = 1)" if int(row["Actual_Dropout"]) == 1 else "Không bỏ học (Non-Dropout = 0)",
        "prediction_status": str(row["Prediction_Status"]),
        "raw_features": raw_features,
        "prediction": pred_result,
    }


def get_experiments_data() -> Dict[str, Any]:
    """Trả về bảng đối sánh Luồng A/B, kết quả khảo sát siêu tham số và danh sách 7 biểu đồ."""
    res = get_resources()

    results = res.results_df.to_dict(orient="records")
    tuning = res.tuning_df.to_dict(orient="records")

    # Tính mức cải thiện F1-score từ Luồng A -> Luồng B theo từng thuật toán
    improvements = []
    for model_name in res.results_df["Model"].unique():
        sub_a = res.results_df[(res.results_df["Model"] == model_name) & (res.results_df["Flow"].str.contains("Luồng A"))]
        sub_b = res.results_df[(res.results_df["Model"] == model_name) & (res.results_df["Flow"].str.contains("Luồng B"))]
        if not sub_a.empty and not sub_b.empty:
            f1_a = float(sub_a.iloc[0]["F1-score"])
            f1_b = float(sub_b.iloc[0]["F1-score"])
            auc_a = float(sub_a.iloc[0]["ROC-AUC"])
            auc_b = float(sub_b.iloc[0]["ROC-AUC"])
            improvements.append({
                "model": model_name,
                "f1_flow_a": f1_a,
                "f1_flow_b": f1_b,
                "f1_delta": round(f1_b - f1_a, 4),
                "auc_flow_a": auc_a,
                "auc_flow_b": auc_b,
                "auc_delta": round(auc_b - auc_a, 4),
            })

    return {
        "results": results,
        "tuning": tuning,
        "improvements": improvements,
        "figures": FIGURE_METADATA,
    }

