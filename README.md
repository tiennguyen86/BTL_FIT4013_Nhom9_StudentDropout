# 🎓 Hệ thống Cảnh báo sớm Sinh viên có Nguy cơ Bỏ học (Student Dropout Prediction)

> **Học phần:** Nhập môn Học máy (FIT4013) — Năm học 2026–2027
> **Giảng viên hướng dẫn:** ThS. Lê Thị Thuỳ Trang
> **Nhóm thực hiện:** Nhóm 9

---

## 👥 Danh sách thành viên nhóm

| Mã sinh viên | Họ và tên | Lớp | Phân công công việc chính |
| :---: | :--- | :---: | :--- |
| **1871020569** | Nguyễn Hữu Tiến | CNTT 18-10 | Trưởng nhóm; Xây dựng pipeline tiền xử lý (Luồng A/B); Đóng gói model `.pkl` và Giao diện Web Demo Streamlit (`app.py`). |
| **1871020502** | Vương Xuân Sơn | CNTT 18-10 | Khám phá dữ liệu (EDA), thống kê mô tả, phát hiện ngoại lệ IQR, huấn luyện thuật toán K-NN & Naive Bayes. |
| **1871020698** | Lý Triệu Trung | CNTT 18-10 | Huấn luyện Cây quyết định & Random Forest, đánh giá đa chỉ số, phân tích lỗi (Error Analysis) và phân tầng rủi ro. |

---

## 📂 Cấu trúc Thư mục Dự án

```text
BTL_FIT4013_Nhom9_StudentDropout/
│
├── data/
│   └── data.csv                  # Dữ liệu gốc UCI (4.424 dòng x 37 cột)
│
├── notebooks/
│   └── BTL_Nhom9_EndToEnd.ipynb  # Jupyter Notebook chạy trực quan từ EDA đến kết luận
│
├── src/
│   ├── __init__.py               # Khởi tạo package src
│   ├── utils.py                  # Cố định random_state=42, quản lý đường dẫn, phân tầng rủi ro
│   ├── preprocessing.py          # Làm sạch dữ liệu, tạo 7 đặc trưng miền giáo dục, Luồng A & B
│   └── train_evaluate.py         # Huấn luyện K-NN, Naive Bayes, Decision Tree, Random Forest & vẽ biểu đồ
│
├── models/
│   └── best_dropout_pipeline.pkl # Pipeline hoàn chỉnh (Preprocessor + Model Random Forest) tốt nhất
│
├── outputs/
│   ├── Hinhve/                   # Thư mục chứa 7 ảnh biểu đồ (.png) độ phân giải cao cho Overleaf
│   ├── results_comparison.csv    # Bảng so sánh Luồng A vs Luồng B (Accuracy, Precision, Recall, F1, AUC)
│   ├── tuning_results.csv        # Bảng khảo sát thay đổi siêu tham số (Hyperparameter Tuning)
│   └── error_analysis.csv        # Danh sách chi tiết các ca dự đoán sai (False Negative / False Positive)
│
├── report/                       # Thư mục chứa báo cáo LaTeX và file PDF hoàn chỉnh
│   └── BaoCao_BTL_Nhom9.pdf
│
├── run_pipeline.py               # Điểm chạy chính (Entry-point): Chạy 1 lệnh ra toàn bộ kết quả
├── app.py                        # Giao diện Web Demo tương tác bằng Streamlit (Điểm thưởng GUI)
├── requirements.txt              # Khai báo các thư viện Python cần thiết
├── AI_USAGE.md                   # Khai báo minh bạch việc sử dụng công cụ AI
└── README.md                     # Tài liệu hướng dẫn sử dụng dự án
