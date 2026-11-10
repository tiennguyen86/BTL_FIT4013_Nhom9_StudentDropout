# -*- coding: utf-8 -*-
"""
Script: app.py
Chức năng: Giao diện Web Demo (Streamlit) dự đoán sinh viên có nguy cơ bỏ học
và phân tầng rủi ro 3 mức (Thấp - Trung bình - Cao).
Cách chạy:
    streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

# Import từ src/ để joblib nhận diện đúng class DomainFeatureCreator trong pipeline
import src.preprocessing as prep
from src.train_evaluate import run_experiment_pipeline
from src.utils import (
    BEST_MODEL_PATH,
    DATA_PATH,
    ERROR_CSV_PATH,
    RESULTS_CSV_PATH,
    TUNING_CSV_PATH,
    assign_risk_tier,
    get_risk_recommendation,
)

st.set_page_config(
    page_title="Cảnh báo sớm Sinh viên Bỏ học - Nhóm 9",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource
def load_model_and_data():
    """Nạp mô hình đã huấn luyện và dữ liệu gốc (tự động chạy pipeline nếu chưa có file .pkl)."""
    if not BEST_MODEL_PATH.exists():
        run_experiment_pipeline()
    model = joblib.load(BEST_MODEL_PATH)
    df = prep.load_and_clean_data(DATA_PATH)
    _, X_test, _, y_test = prep.split_train_test(df)
    return model, df, X_test, y_test


def main():
    st.title("🎓 Hệ thống Cảnh báo sớm Sinh viên có Nguy cơ Bỏ học")
    st.caption("Học phần: Nhập môn Học máy (FIT4013) — Nhóm 9 | Mô hình: Luồng B + Random Forest")

    model, df, X_test, y_test = load_model_and_data()
    feature_cols = [c for c in df.columns if c not in ("Target", "Target_Binary")]

    tab1, tab2, tab3 = st.tabs([
        "🔍 Dự đoán Hồ sơ Sinh viên",
        "📊 Kết quả Thực nghiệm (Luồng A vs B)",
        "⚠️ Phân tích Lỗi & Phân tầng Rủi ro",
    ])

    # ==========================================================================
    # TAB 1: DỰ ĐOÁN TRỰC TIẾP HỒ SƠ SINH VIÊN
    # ==========================================================================
    with tab1:
        col_left, col_right = st.columns([1.15, 1.0])

        with col_left:
            st.subheader("1. Chọn hoặc Nhập thông tin sinh viên")
            mode = st.radio(
                "Chế độ kiểm thử:",
                ["Nhập thông số trực tiếp", "Chọn hồ sơ thực tế từ Tập kiểm thử (Test set)"],
                horizontal=True,
            )

            # Lấy dòng trung vị làm giá trị mặc định cho các cột phụ
            base_row = X_test.median(numeric_only=True).to_dict()

            if mode == "Chọn hồ sơ thực tế từ Tập kiểm thử (Test set)":
                sample_idx = st.selectbox(
                    "Chọn mã dòng sinh viên trong tập Test (N = 885):",
                    options=list(X_test.index[:100]),
                    format_func=lambda idx: (
                        f"Sinh viên #{idx} — Thực tế: "
                        f"{'BỎ HỌC (1)' if y_test.loc[idx] == 1 else 'KHÔNG BỎ HỌC (0)'}"
                    ),
                )
                input_df = X_test.loc[[sample_idx]].copy()
                actual_label = int(y_test.loc[sample_idx])
                st.info(
                    f"**Nhãn thực tế trong dữ liệu:** "
                    f"{'🔴 Bỏ học (Dropout = 1)' if actual_label == 1 else '🟢 Không bỏ học (Non-Dropout = 0)'}"
                )
            else:
                c1, c2 = st.columns(2)
                with c1:
                    enr_1 = st.number_input("Số môn đăng ký HK1", 0, 26, 6)
                    app_1 = st.number_input("Số môn đậu HK1", 0, 26, 5)
                    grd_1 = st.slider("Điểm trung bình HK1 (thang 0-20)", 0.0, 20.0, 12.0, 0.5)
                    no_eval_1 = st.number_input("Số môn bỏ thi HK1", 0, 12, 0)
                    tuition_ok = st.selectbox("Đóng học phí đúng hạn", [1, 0], format_func=lambda x: "Có (1)" if x == 1 else "Chưa đóng (0)")
                    debtor = st.selectbox("Có nợ học phí (Debtor)", [0, 1], format_func=lambda x: "Không nợ (0)" if x == 0 else "Đang nợ (1)")

                with c2:
                    enr_2 = st.number_input("Số môn đăng ký HK2", 0, 23, 6)
                    app_2 = st.number_input("Số môn đậu HK2", 0, 20, 5)
                    grd_2 = st.slider("Điểm trung bình HK2 (thang 0-20)", 0.0, 20.0, 12.0, 0.5)
                    no_eval_2 = st.number_input("Số môn bỏ thi HK2", 0, 12, 0)
                    scholarship = st.selectbox("Có học bổng", [0, 1], format_func=lambda x: "Không (0)" if x == 0 else "Có học bổng (1)")
                    age = st.number_input("Tuổi khi nhập học", 17, 70, 20)

                base_row.update({
                    "Curricular units 1st sem (enrolled)": enr_1,
                    "Curricular units 1st sem (approved)": app_1,
                    "Curricular units 1st sem (grade)": grd_1,
                    "Curricular units 1st sem (without evaluations)": no_eval_1,
                    "Curricular units 2nd sem (enrolled)": enr_2,
                    "Curricular units 2nd sem (approved)": app_2,
                    "Curricular units 2nd sem (grade)": grd_2,
                    "Curricular units 2nd sem (without evaluations)": no_eval_2,
                    "Tuition fees up to date": tuition_ok,
                    "Debtor": debtor,
                    "Scholarship holder": scholarship,
                    "Age at enrollment": age,
                })
                input_df = pd.DataFrame([base_row])[feature_cols]

        with col_right:
            st.subheader("2. Kết quả Dự báo & Khuyến nghị")
            prob = float(model.predict_proba(input_df)[0, 1])
            pred_label = int(prob >= 0.50)
            tier = assign_risk_tier(prob)
            rec = get_risk_recommendation(prob)

            st.metric("Xác suất Bỏ học dự đoán P(Dropout = 1)", f"{prob * 100:.2f}%")
            st.progress(min(max(prob, 0.0), 1.0))

            if prob >= 0.70:
                st.error(f"**Mức phân tầng rủi ro:** {tier}")
            elif prob >= 0.40:
                st.warning(f"**Mức phân tầng rủi ro:** {tier}")
            else:
                st.success(f"**Mức phân tầng rủi ro:** {tier}")

            st.markdown(f"**Kết luận phân lớp (ngưỡng 0.50):** `{'Dropout (1)' if pred_label == 1 else 'Non-Dropout (0)'}`")
            st.markdown(f"**Khuyến nghị cho Cố vấn học tập:**\n> {rec}")

            st.markdown("**Tóm tắt các chỉ số học tập & tài chính đã tính trong Luồng B:**")
            engineered = prep.DomainFeatureCreator().transform(input_df)
            show_cols = [
                "Approval_rate_1st_sem",
                "Approval_rate_2nd_sem",
                "Avg_grade_both_sem",
                "Total_approved_units",
                "Total_without_eval",
                "Financial_risk_score",
            ]
            st.dataframe(engineered[show_cols].round(3), use_container_width=True)

    # ==========================================================================
    # TAB 2: KẾT QUẢ THỰC NGHIỆM (LUỒNG A VS LUỒNG B & SIÊU THAM SỐ)
    # ==========================================================================
    with tab2:
        st.subheader("Bảng đối sánh Luồng A (Cơ bản) và Luồng B (Cải tiến)")
        if RESULTS_CSV_PATH.exists():
            res_df = pd.read_csv(RESULTS_CSV_PATH)
            st.dataframe(res_df, use_container_width=True)

        st.subheader("Kết quả khảo sát Siêu tham số (Hyperparameter Tuning)")
        if TUNING_CSV_PATH.exists():
            tun_df = pd.read_csv(TUNING_CSV_PATH)
            st.dataframe(tun_df, use_container_width=True)

    # ==========================================================================
    # TAB 3: PHÂN TÍCH LỖI (ERROR ANALYSIS)
    # ==========================================================================
    with tab3:
        st.subheader("Danh sách các trường hợp dự đoán trên tập Kiểm thử (N = 885)")
        if ERROR_CSV_PATH.exists():
            err_df = pd.read_csv(ERROR_CSV_PATH)
            err_col = "Error_Type" if "Error_Type" in err_df.columns else "Prediction_Status"
            filter_type = st.selectbox("Lọc theo trạng thái dự đoán:", ["Tất cả"] + list(err_df[err_col].unique()))
            if filter_type != "Tất cả":
                err_df = err_df[err_df[err_col] == filter_type]
            st.dataframe(err_df.head(50), use_container_width=True)


if __name__ == "__main__":
    main()
