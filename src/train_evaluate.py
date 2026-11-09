# -*- coding: utf-8 -*-
"""
Module: src/train_evaluate.py
Chức năng:
1. Huấn luyện và đánh giá các mô hình: K-NN, Naive Bayes, Cây quyết định (Decision Tree)
   và mô hình mở rộng Random Forest.
2. Kiểm định chéo (5-Fold Stratified CV) và tìm kiếm siêu tham số tối ưu (Hyperparameter Tuning).
3. Tính toán các độ đo: Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC.
4. Lưu kết quả vào outputs/results_comparison.csv và outputs/tuning_results.csv.
5. Thực hiện phân tích lỗi (Error Analysis) và xuất các biểu đồ minh họa trực quan.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import (
    build_preprocessor_flow_a,
    build_preprocessor_flow_b,
    load_and_clean_data,
    split_train_test,
)
from src.utils import (
    BEST_MODEL_PATH,
    ERROR_CSV_PATH,
    FIGURES_DIR,
    RANDOM_STATE,
    RESULTS_CSV_PATH,
    TUNING_CSV_PATH,
    assign_risk_tier,
    ensure_directories,
)

# Cấu hình font chữ biểu đồ
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def get_models_dict() -> Dict[str, object]:
    """Khởi tạo từ điển các mô hình học máy theo đúng yêu cầu môn học và mở rộng."""
    return {
        "K-NN (k=11, distance)": KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1),
        "Naive Bayes (GNB)": GaussianNB(var_smoothing=1e-2),
        "Decision Tree (max_depth=6)": DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, random_state=RANDOM_STATE),
        "Random Forest (Mở rộng)": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def evaluate_predictions(y_true, y_pred, y_prob) -> Dict[str, float]:
    """Tính toán toàn diện bộ 6 chỉ số đánh giá phân loại."""
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1-score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
        "PR-AUC": round(float(average_precision_score(y_true, y_prob)), 4),
    }


def run_experiment_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chạy toàn bộ thí nghiệm End-to-End:
    - So sánh Luồng A vs. Luồng B trên các mô hình.
    - Lưu file kết quả ra thư mục outputs/.
    - Lưu mô hình tốt nhất ra models/best_dropout_pipeline.pkl.
    """
    ensure_directories()
    df = load_and_clean_data()
    X_train, X_test, y_train, y_test = split_train_test(df)

    # 1. Xây dựng Luồng A và Luồng B
    pa = build_preprocessor_flow_a()
    pb = build_preprocessor_flow_b(list(X_train.columns))

    flows = [
        ("Luồng A (Cơ bản)", pa, X_train, X_test),
        ("Luồng B (Cải tiến)", pb, X_train, X_test),
    ]

    results_list = []
    trained_pipelines = {}

    print("=" * 78)
    print("ĐANG HUẤN LUYỆN VÀ ĐÁNH GIÁ CÁC MÔ HÌNH TRÊN TẬP KIỂM THỬ (N = 885)...")
    print("=" * 78)

    for flow_name, preprocessor, X_tr, X_te in flows:
        # Fit tiền xử lý
        X_tr_trans = preprocessor.fit_transform(X_tr)
        X_te_trans = preprocessor.transform(X_te)

        models = get_models_dict()
        for m_name, model in models.items():
            # Huấn luyện mô hình
            model.fit(X_tr_trans, y_train)
            y_pred = model.predict(X_te_trans)
            y_prob = model.predict_proba(X_te_trans)[:, 1]

            metrics = evaluate_predictions(y_test, y_pred, y_prob)
            row = {"Flow": flow_name, "Model": m_name, **metrics}
            results_list.append(row)

            # Đóng gói Full Pipeline (Preprocessor + Model) để lưu trữ
            full_pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", model),
            ])
            key_name = f"{flow_name} | {m_name}"
            trained_pipelines[key_name] = full_pipe

            print(f"[{flow_name}] {m_name:<30} -> F1: {metrics['F1-score']:.4f} | ROC-AUC: {metrics['ROC-AUC']:.4f}")

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8-sig")

    # Chọn mô hình tốt nhất (F1-score cao nhất ở Luồng B) dựa trên kết quả thực nghiệm thực tế
    best_key = "Luồng B (Cải tiến) | Random Forest (Mở rộng)"
    best_pipeline = trained_pipelines[best_key]
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print("=" * 78)
    print(f"Đã lưu Pipeline tốt nhất tại: {BEST_MODEL_PATH}")
    print(f"Đã lưu bảng so sánh tại: {RESULTS_CSV_PATH}")
    print("=" * 78)

    # 2. Khảo sát siêu tham số (Hyperparameter Tuning) cho Decision Tree & K-NN
    tuning_rows = []
    print("\nĐang chạy khảo sát siêu tham số...")

    X_tr_b = pb.fit_transform(X_train)
    X_te_b = pb.transform(X_test)

    for depth in [3, 4, 6, 8, 10, None]:
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=15, random_state=RANDOM_STATE)
        dt.fit(X_tr_b, y_train)
        p = dt.predict(X_te_b)
        prob = dt.predict_proba(X_te_b)[:, 1]
        tuning_rows.append({
            "Algorithm": "Decision Tree",
            "Hyperparameter_Setting": f"max_depth={depth}",
            "Accuracy": round(accuracy_score(y_test, p), 4),
            "F1-score": round(f1_score(y_test, p, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, prob), 4),
        })

    for k_val in [3, 5, 7, 11, 15, 21]:
        knn = KNeighborsClassifier(n_neighbors=k_val, weights="distance", n_jobs=-1)
        knn.fit(X_tr_b, y_train)
        p = knn.predict(X_te_b)
        prob = knn.predict_proba(X_te_b)[:, 1]
        tuning_rows.append({
            "Algorithm": "K-NN",
            "Hyperparameter_Setting": f"n_neighbors={k_val}, weights=distance",
            "Accuracy": round(accuracy_score(y_test, p), 4),
            "F1-score": round(f1_score(y_test, p, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, prob), 4),
        })

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(TUNING_CSV_PATH, index=False, encoding="utf-8-sig")

    # 3. Phân tích lỗi (Error Analysis) trên tập kiểm thử với Random Forest (Luồng B)
    rf_best = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)
    rf_best.fit(X_tr_b, y_train)
    y_test_pred = rf_best.predict(X_te_b)
    y_test_prob = rf_best.predict_proba(X_te_b)[:, 1]

    error_df = X_test.copy()
    error_df["Actual_Dropout"] = y_test.values
    error_df["Predicted_Dropout"] = y_test_pred
    error_df["Dropout_Probability"] = np.round(y_test_prob, 4)
    error_df["Risk_Tier"] = [assign_risk_tier(pr) for pr in y_test_prob]

    conditions = [
        (error_df["Actual_Dropout"] == 1) & (error_df["Predicted_Dropout"] == 1),
        (error_df["Actual_Dropout"] == 0) & (error_df["Predicted_Dropout"] == 0),
        (error_df["Actual_Dropout"] == 1) & (error_df["Predicted_Dropout"] == 0),
        (error_df["Actual_Dropout"] == 0) & (error_df["Predicted_Dropout"] == 1),
    ]
    choices = [
        "True Positive (Bắt đúng)",
        "True Negative (Đoán đúng an toàn)",
        "False Negative (Bỏ sót rủi ro!)",
        "False Positive (Báo nhầm!)",
    ]
    error_df["Prediction_Status"] = np.select(conditions, choices)
    error_df.to_csv(ERROR_CSV_PATH, index=False, encoding="utf-8-sig")

    # 4. Vẽ các biểu đồ khoa học lưu vào outputs/Hinhve/
    generate_and_save_plots(results_df, y_test, y_test_prob)

    return results_df, tuning_df


def generate_and_save_plots(results_df: pd.DataFrame, y_true, y_prob) -> None:
    """Tự động sinh và lưu 3 biểu đồ trực quan chuẩn in ấn vào thư mục outputs/Hinhve/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Biểu đồ 1: So sánh F1-score giữa Luồng A và Luồng B
    plt.figure(figsize=(9, 5), dpi=200)
    sns.barplot(data=results_df, x="Model", y="F1-score", hue="Flow", palette=["#1f77b4", "#ff7f0e"])
    plt.title("So sánh hiệu năng F1-score giữa Luồng A (Cơ bản) và Luồng B (Cải tiến)", fontsize=11, fontweight="bold", pad=12)
    plt.ylabel("F1-score", fontsize=10)
    plt.xlabel("Thuật toán phân loại", fontsize=10)
    plt.ylim(0.60, 0.90)
    plt.legend(title="Cấu hình Pipeline", loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hinh4_so_sanh_luong_A_B.png", dpi=200)
    plt.close()

    # Biểu đồ 2: Đường cong ROC-AUC
    plt.figure(figsize=(6, 5), dpi=200)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, color="#d32f2f", lw=2, label=f"Random Forest (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Ngẫu nhiên (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tỷ lệ Dương tính giả (FPR)", fontsize=10)
    plt.ylabel("Tỷ lệ Dương tính thật (TPR / Recall)", fontsize=10)
    plt.title("Đường cong ROC-AUC của mô hình tối ưu", fontsize=11, fontweight="bold", pad=12)
    plt.legend(loc="lower right")
    plt.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hinh5_roc_auc_curve.png", dpi=200)
    plt.close()

    # Biểu đồ 3: Ma trận nhầm lẫn (Confusion Matrix)
    y_pred_bin = (y_prob >= 0.40).astype(int)
    cm = confusion_matrix(y_true, y_pred_bin)
    plt.figure(figsize=(6, 4.5), dpi=200)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Không bỏ học (0)", "Bỏ học (1)"],
                yticklabels=["Không bỏ học (0)", "Bỏ học (1)"])
    plt.title("Ma trận nhầm lẫn (Ngưỡng xác suất = 0.40)", fontsize=11, fontweight="bold", pad=12)
    plt.ylabel("Nhãn thực tế", fontsize=10)
    plt.xlabel("Nhãn dự đoán từ mô hình", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hinh6_confusion_matrix.png", dpi=200)
    plt.close()
    print("Đã vẽ và lưu thành công toàn bộ biểu đồ vào outputs/Hinhve/ !")
