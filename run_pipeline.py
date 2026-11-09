# -*- coding: utf-8 -*-
"""
File: run_pipeline.py
Chạy toàn bộ quy trình từ dữ liệu thô đến kết quả (không cần mở notebook):

    python run_pipeline.py

Kết quả:
    outputs/results_comparison.csv      bảng so sánh Luồng A/B, baseline/tinh chỉnh
    outputs/tuning_results.csv          điểm kiểm định chéo của từng tổ hợp siêu tham số
    outputs/error_analysis.csv          dự đoán từng sinh viên trong tập kiểm thử
    outputs/error_group_summary.csv     tỉ lệ bỏ sót / báo nhầm theo nhóm sinh viên
    outputs/risk_tiers_test.csv         phân tầng rủi ro
    outputs/feature_importance.csv      độ quan trọng đặc trưng
    outputs/Hinhve/*.png                biểu đồ để chèn vào báo cáo
    models/best_dropout_pipeline.pkl    mô hình tốt nhất cho app demo
"""

import time

from src.preprocessing import load_and_clean_data, print_eda_overview
from src.train_evaluate import generate_eda_plots, run_experiment_pipeline


def main() -> None:
    t0 = time.time()
    df = load_and_clean_data()
    print_eda_overview(df)
    generate_eda_plots(df)
    run_experiment_pipeline()
    print(f"Hoàn thành sau {time.time() - t0:.0f} giây.")


if __name__ == "__main__":
    main()
