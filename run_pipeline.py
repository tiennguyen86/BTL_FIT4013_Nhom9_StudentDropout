# -*- coding: utf-8 -*-
"""
Script: run_pipeline.py
Chức năng:
Điểm chạy chính (Entry-point) của dự án BTL Nhập môn Học máy (FIT4013) - Nhóm 9.
Chỉ cần chạy 1 lệnh:
    python3 run_pipeline.py
Chương trình sẽ tự động thực hiện toàn bộ quy trình End-to-End:
1. Kiểm tra dữ liệu đầu vào (data/data.csv) và in báo cáo khám phá dữ liệu (EDA).
2. Vẽ đầy đủ bộ 7 biểu đồ khoa học (Hình 1 -> Hình 7) lưu vào outputs/Hinhve/.
3. Huấn luyện & đối sánh Luồng A (Cơ bản) vs. Luồng B (Cải tiến) trên K-NN, Naive Bayes,
   Decision Tree và Random Forest.
4. Khảo sát siêu tham số (Hyperparameter Tuning) và lưu outputs/tuning_results.csv.
5. Phân tích lỗi (Error Analysis), phân tầng rủi ro 3 mức và lưu outputs/error_analysis.csv.
6. Xuất mô hình tốt nhất ra models/best_dropout_pipeline.pkl phục vụ ứng dụng Web Demo (app.py).
"""

import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.preprocessing import (
    get_flow_b_feature_names,
    load_and_clean_data,
)
from src.train_evaluate import run_experiment_pipeline
from src.utils import (
    BEST_MODEL_PATH,
    DATA_PATH,
    ERROR_CSV_PATH,
    FIGURES_DIR,
    RESULTS_CSV_PATH,
    TUNING_CSV_PATH,
    ensure_directories,
    get_vi_name,
    set_seed,
)

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def ensure_all_figures_exist(df: pd.DataFrame) -> None:
    """
    Đảm bảo đủ 7 biểu đồ (hinh1 -> hinh7) trong outputs/Hinhve/
    ngay cả khi bản src/train_evaluate.py trước đó chỉ mới vẽ hinh4, hinh5, hinh6.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Hình 1: Phân phối nhãn 3 lớp gốc và nhãn nhị phân
    h1_path = FIGURES_DIR / "hinh1_phan_phoi_nhan.png"
    if not h1_path.exists():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=200)
        classes_3 = ["Graduate\n(Tốt nghiệp)", "Dropout\n(Bỏ học)", "Enrolled\n(Đang học)"]
        counts_3 = [
            int((df["Target"] == "Graduate").sum()),
            int((df["Target"] == "Dropout").sum()),
            int((df["Target"] == "Enrolled").sum()),
        ]
        bars1 = axes[0].bar(classes_3, counts_3, color=["#2b5c8f", "#d95f02", "#7570b3"], width=0.55, edgecolor="black")
        axes[0].set_title("a) Phân phối 3 nhãn gốc (N = 4.424)", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Số lượng sinh viên")
        axes[0].set_ylim(0, 2600)
        axes[0].grid(axis="y", linestyle="--", alpha=0.4)
        for bar, c in zip(bars1, counts_3):
            axes[0].text(bar.get_x() + bar.get_width() / 2, c + 50, f"{c}\n({c/len(df)*100:.2f}%)", ha="center", fontsize=9, fontweight="bold")

        classes_2 = ["Không bỏ học (0)\n[Graduate + Enrolled]", "Bỏ học (1)\n[Dropout]"]
        counts_2 = [int((df["Target_Binary"] == 0).sum()), int((df["Target_Binary"] == 1).sum())]
        bars2 = axes[1].bar(classes_2, counts_2, color=["#388e3c", "#d32f2f"], width=0.48, edgecolor="black")
        axes[1].set_title("b) Phân phối nhãn nhị phân mục tiêu", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Số lượng sinh viên")
        axes[1].set_ylim(0, 3500)
        axes[1].grid(axis="y", linestyle="--", alpha=0.4)
        for bar, c in zip(bars2, counts_2):
            axes[1].text(bar.get_x() + bar.get_width() / 2, c + 70, f"{c}\n({c/len(df)*100:.2f}%)", ha="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.savefig(h1_path, dpi=200)
        plt.close()

    # 2. Hình 2: Tỷ lệ bỏ học theo tài chính & Top 10 tương quan Pearson
    h2_path = FIGURES_DIR / "hinh2_tai_chinh_tuong_quan.png"
    if not h2_path.exists():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=200)
        cat_fin = [
            "Chưa đóng\nhọc phí", "Đã đóng\nhọc phí",
            "Có nợ\n(Debtor=1)", "Không nợ\n(Debtor=0)",
            "Không có\nhọc bổng", "Có\nhọc bổng",
        ]
        rates_fin = [
            df[df["Tuition fees up to date"] == 0]["Target_Binary"].mean() * 100,
            df[df["Tuition fees up to date"] == 1]["Target_Binary"].mean() * 100,
            df[df["Debtor"] == 1]["Target_Binary"].mean() * 100,
            df[df["Debtor"] == 0]["Target_Binary"].mean() * 100,
            df[df["Scholarship holder"] == 0]["Target_Binary"].mean() * 100,
            df[df["Scholarship holder"] == 1]["Target_Binary"].mean() * 100,
        ]
        colors_fin = ["#c62828", "#2e7d32", "#ef6c00", "#2e7d32", "#f57c00", "#1565c0"]
        bars_f = axes[0].bar(cat_fin, rates_fin, color=colors_fin, width=0.55, edgecolor="black")
        axes[0].set_title("a) Tỷ lệ bỏ học theo Tài chính & Học bổng", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Tỷ lệ bỏ học (%)")
        axes[0].set_ylim(0, 102)
        axes[0].grid(axis="y", linestyle="--", alpha=0.4)
        for b, r in zip(bars_f, rates_fin):
            axes[0].text(b.get_x() + b.get_width() / 2, r + 2, f"{r:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

        corr_s = df.drop(columns=["Target", "Target_Binary"]).corrwith(df["Target_Binary"]).sort_values()
        top_corr = pd.concat([corr_s.head(5), corr_s.tail(5)])
        y_labels = [get_vi_name(k) for k in top_corr.index]
        c_bars = ["#1976d2" if v < 0 else "#d32f2f" for v in top_corr.values]
        bars_c = axes[1].barh(y_labels, top_corr.values, color=c_bars, height=0.62, edgecolor="black")
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].set_title("b) Top 10 thuộc tính tương quan với Bỏ học", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Hệ số tương quan Pearson (r)")
        axes[1].set_xlim(-0.70, 0.38)
        axes[1].grid(axis="x", linestyle="--", alpha=0.4)
        for b, v in zip(bars_c, top_corr.values):
            offset = -0.02 if v < 0 else 0.02
            ha = "right" if v < 0 else "left"
            axes[1].text(v + offset, b.get_y() + b.get_height() / 2, f"{v:+.2f}", ha=ha, va="center", fontsize=8.5, fontweight="bold")

        plt.tight_layout()
        plt.savefig(h2_path, dpi=200)
        plt.close()

    # 3. Hình 3: Boxplot kết quả học tập HK1 và HK2
    h3_path = FIGURES_DIR / "hinh3_boxplot_hoc_tap.png"
    if not h3_path.exists():
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=200)
        df_plot = df.copy()
        df_plot["Nhãn"] = df_plot["Target_Binary"].map({0: "Không bỏ học (0)", 1: "Bỏ học (1)"})
        pal = {"Không bỏ học (0)": "#4caf50", "Bỏ học (1)": "#e53935"}

        sns.boxplot(data=df_plot, x="Nhãn", y="Curricular units 1st sem (approved)", ax=axes[0], palette=pal, width=0.45)
        axes[0].set_title("a) Số môn đậu Học kỳ 1", fontsize=10.5, fontweight="bold")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Số học phần đạt")
        axes[0].grid(axis="y", linestyle="--", alpha=0.4)

        sns.boxplot(data=df_plot, x="Nhãn", y="Curricular units 2nd sem (approved)", ax=axes[1], palette=pal, width=0.45)
        axes[1].set_title("b) Số môn đậu Học kỳ 2", fontsize=10.5, fontweight="bold")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Số học phần đạt")
        axes[1].grid(axis="y", linestyle="--", alpha=0.4)

        sns.boxplot(data=df_plot, x="Nhãn", y="Curricular units 2nd sem (grade)", ax=axes[2], palette=pal, width=0.45)
        axes[2].set_title("c) Điểm trung bình Học kỳ 2", fontsize=10.5, fontweight="bold")
        axes[2].set_xlabel("")
        axes[2].set_ylabel("Điểm trung bình (thang 0-20)")
        axes[2].grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(h3_path, dpi=200)
        plt.close()

    # 4. Hình 7: Top 12 đặc trưng quan trọng nhất (Feature Importance từ best_dropout_pipeline.pkl)
    h7_path = FIGURES_DIR / "hinh7_feature_importance.png"
    if not h7_path.exists() and BEST_MODEL_PATH.exists():
        best_pipe = joblib.load(BEST_MODEL_PATH)
        preproc_b = best_pipe.named_steps["preprocessor"]
        rf_clf = best_pipe.named_steps["classifier"]
        if hasattr(rf_clf, "feature_importances_"):
            feat_names = get_flow_b_feature_names(preproc_b)
            importances = rf_clf.feature_importances_
            idx = np.argsort(importances)[-12:]
            top_feats = [get_vi_name(feat_names[i]) for i in idx]
            top_vals = importances[idx]

            plt.figure(figsize=(8.5, 4.8), dpi=200)
            bars = plt.barh(top_feats, top_vals, color="#2b5c8f", edgecolor="black", height=0.65)
            plt.title("Top 12 đặc trưng quan trọng nhất trong dự đoán bỏ học (Luồng B)", fontsize=11, fontweight="bold")
            plt.xlabel("Mức độ đóng góp (Gini Feature Importance)")
            plt.xlim(0, max(top_vals) * 1.18)
            plt.grid(axis="x", linestyle="--", alpha=0.4)
            for b, v in zip(bars, top_vals):
                plt.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8.5, fontweight="bold")
            plt.tight_layout()
            plt.savefig(h7_path, dpi=200)
            plt.close()


def main() -> None:
    """Hàm điều phối chạy toàn bộ pipeline từ dữ liệu thô đến kết quả cuối cùng."""
    start_time = time.time()
    set_seed()
    ensure_directories()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp dữ liệu tại {DATA_PATH}. "
            "Vui lòng kiểm tra lại file data/data.csv!"
        )

    # Bước 1: Đọc dữ liệu & chạy thực nghiệm huấn luyện, đánh giá, tuning và phân tích lỗi
    df = load_and_clean_data(DATA_PATH)
    pipeline_outputs = run_experiment_pipeline()
    if len(pipeline_outputs) == 3:
        results_df, tuning_df, error_df = pipeline_outputs
    else:
        results_df, tuning_df = pipeline_outputs
        error_df = pd.read_csv(ERROR_CSV_PATH)

    # Bước 2: Đảm bảo đủ cả 7 biểu đồ trong outputs/Hinhve/
    ensure_all_figures_exist(df)

    # Bước 3: In bảng tổng kết trực quan ra màn hình Terminal
    print("\n" + "=" * 78)
    print("1. BẢNG SO SÁNH HIỆU NĂNG LUỒNG A (CƠ BẢN) VÀ LUỒNG B (CẢI TIẾN)")
    print("=" * 78)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("2. CẤU HÌNH SIÊU THAM SỐ TỐT NHẤT CỦA TỪNG THUẬT TOÁN (TRÊN LUỒNG B)")
    print("=" * 78)
    best_tuning = (
        tuning_df.sort_values(by="F1-score", ascending=False)
        .groupby("Algorithm", as_index=False)
        .first()
    )
    print(best_tuning.to_string(index=False))

    print("\n" + "=" * 78)
    print("3. THỐNG KÊ PHÂN TẦNG RỦI RO BỎ HỌC TRÊN TẬP KIỂM THỬ (N = 885)")
    print("=" * 78)
    tier_summary = (
        error_df.groupby("Risk_Tier")["Actual_Dropout"]
        .agg(So_luong_SV="count", So_SV_bo_hoc_that="sum", Ty_le_bo_hoc_that="mean")
        .reset_index()
    )
    tier_summary["Ty_le_bo_hoc_that"] = (tier_summary["Ty_le_bo_hoc_that"] * 100).round(2).astype(str) + "%"
    print(tier_summary.to_string(index=False))

    elapsed = time.time() - start_time
    print("\n" + "=" * 78)
    print(f"HOÀN TẤT TOÀN BỘ PIPELINE TRONG {elapsed:.2f} GIÂY!")
    print(f"- Mô hình lưu tại       : {BEST_MODEL_PATH}")
    print(f"- Bảng so sánh Luồng A/B: {RESULTS_CSV_PATH}")
    print(f"- Bảng khảo sát tham số : {TUNING_CSV_PATH}")
    print(f"- Bảng phân tích lỗi    : {ERROR_CSV_PATH}")
    print(f"- Thư mục ảnh báo cáo   : {FIGURES_DIR} ({len(list(FIGURES_DIR.glob('*.png')))} ảnh .png)")
    print("=" * 78)


if __name__ == "__main__":
    main()
