# -*- coding: utf-8 -*-
"""
Kiểm thử tích hợp toàn bộ API và giao diện Web Dashboard (FastAPI TestClient).
Chạy:
    python -m unittest tests.test_web_app -v
"""

import unittest
from fastapi.testclient import TestClient

from app import app


class TestStudentDropoutWebApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_cm.__exit__(None, None, None)

    def test_01_static_and_health(self) -> None:
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertIn("EduGuard AI", res_root.text)

        res_css = self.client.get("/static/styles.css")
        self.assertEqual(res_css.status_code, 200)

        res_js = self.client.get("/static/app.js")
        self.assertEqual(res_js.status_code, 200)

        res_health = self.client.get("/api/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "ok")

    def test_02_overview_and_metadata(self) -> None:
        ov = self.client.get("/api/overview").json()
        self.assertEqual(ov["dataset"]["total"], 4424)
        self.assertEqual(ov["dataset"]["test_size"], 885)
        self.assertEqual(len(ov["risk_tiers"]), 3)
        self.assertGreaterEqual(len(ov["feature_importance"]), 10)

        meta = self.client.get("/api/metadata").json()
        self.assertEqual(len(meta["feature_cols"]), 36)
        self.assertEqual(len(meta["presets"]), 3)
        self.assertIn("Course", meta["options"])

    def test_03_single_prediction_risk_tiers(self) -> None:
        # Hồ sơ rủi ro cao
        high_res = self.client.post(
            "/api/predict",
            json={
                "features": {
                    "Curricular units 1st sem (enrolled)": 6,
                    "Curricular units 1st sem (approved)": 0,
                    "Curricular units 1st sem (grade)": 0.0,
                    "Curricular units 2nd sem (enrolled)": 6,
                    "Curricular units 2nd sem (approved)": 0,
                    "Curricular units 2nd sem (grade)": 0.0,
                    "Tuition fees up to date": 0,
                    "Debtor": 1,
                    "Scholarship holder": 0,
                },
                "threshold": 0.50,
            },
        )
        self.assertEqual(high_res.status_code, 200)
        high_data = high_res.json()
        self.assertGreaterEqual(high_data["probability"], 0.70)
        self.assertEqual(high_data["tier_code"], "high")
        self.assertEqual(len(high_data["engineered_features"]), 7)

        # Hồ sơ an toàn
        low_res = self.client.post(
            "/api/predict",
            json={
                "features": {
                    "Curricular units 1st sem (enrolled)": 6,
                    "Curricular units 1st sem (approved)": 6,
                    "Curricular units 1st sem (grade)": 14.5,
                    "Curricular units 2nd sem (enrolled)": 6,
                    "Curricular units 2nd sem (approved)": 6,
                    "Curricular units 2nd sem (grade)": 14.8,
                    "Tuition fees up to date": 1,
                    "Debtor": 0,
                    "Scholarship holder": 1,
                },
                "threshold": 0.50,
            },
        )
        self.assertEqual(low_res.status_code, 200)
        low_data = low_res.json()
        self.assertLess(low_data["probability"], 0.40)
        self.assertEqual(low_data["tier_code"], "low")

    def test_04_batch_sample_and_predict_batch(self) -> None:
        sample_resp = self.client.get("/api/batch-sample?limit=15")
        self.assertEqual(sample_resp.status_code, 200)
        batch_resp = self.client.post(
            "/api/predict-batch?threshold=0.50",
            files={"file": ("sample.csv", sample_resp.content, "text/csv")},
        )
        self.assertEqual(batch_resp.status_code, 200)
        batch_data = batch_resp.json()
        self.assertEqual(batch_data["total"], 15)
        self.assertEqual(len(batch_data["rows"]), 15)

    def test_05_experiments_figures_and_test_samples(self) -> None:
        exp = self.client.get("/api/experiments").json()
        self.assertEqual(len(exp["results"]), 8)
        self.assertEqual(len(exp["figures"]), 7)

        fig_resp = self.client.get("/figures/hinh1_phan_phoi_nhan.png")
        self.assertEqual(fig_resp.status_code, 200)

        samples = self.client.get("/api/test-samples?page=1&page_size=10").json()
        self.assertEqual(samples["total_all"], 885)
        self.assertEqual(len(samples["items"]), 10)

        first_id = samples["items"][0]["student_index"]
        detail = self.client.get(f"/api/test-samples/{first_id}").json()
        self.assertEqual(detail["student_index"], first_id)


if __name__ == "__main__":
    unittest.main()

