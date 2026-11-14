# -*- coding: utf-8 -*-
"""
Test file: tests/test_e2e_frontend.py
Validates all backend APIs, static assets delivery, and integration points for the new UI.
"""

import io
import unittest
from fastapi.testclient import TestClient

from app import app


class TestFrontendE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_01_static_and_health(self):
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertIn("EduGuard AI", res_root.text)
        self.assertIn("Hệ thống Giám sát & Cảnh báo sớm Rủi ro Bỏ học", res_root.text)

        res_css = self.client.get("/static/styles.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertIn("--brand-primary", res_css.text)

        res_js = self.client.get("/static/app.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn("EduGuard AI", res_js.text)

        res_health = self.client.get("/api/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "ok")

    def test_02_overview_and_metadata(self):
        ov = self.client.get("/api/overview").json()
        self.assertEqual(ov["dataset"]["total"], 4424)
        self.assertEqual(ov["dataset"]["test_size"], 885)
        self.assertEqual(len(ov["risk_tiers"]), 3)
        self.assertGreaterEqual(len(ov["feature_importance"]), 10)

        meta = self.client.get("/api/metadata").json()
        self.assertEqual(len(meta["feature_cols"]), 36)
        self.assertEqual(len(meta["presets"]), 3)
        self.assertIn("Course", meta["options"])

    def test_03_prediction_and_interventions(self):
        # Hồ sơ rủi ro cao
        payload_high = {
            "features": {
                "Curricular units 1st sem (approved)": 0,
                "Curricular units 2nd sem (approved)": 0,
                "Tuition fees up to date": 0,
                "Debtor": 1,
                "Scholarship holder": 0,
            },
            "threshold": 0.50,
        }
        res_high = self.client.post("/api/predict", json=payload_high).json()
        self.assertGreaterEqual(res_high["probability"], 0.65)
        self.assertEqual(res_high["tier_code"], "high")

        # Hồ sơ can thiệp (hỗ trợ đóng học phí + xóa nợ)
        payload_intervened = {
            "features": {
                "Curricular units 1st sem (approved)": 0,
                "Curricular units 2nd sem (approved)": 0,
                "Tuition fees up to date": 1,
                "Debtor": 0,
                "Scholarship holder": 1,
            },
            "threshold": 0.50,
        }
        res_intervened = self.client.post("/api/predict", json=payload_intervened).json()
        self.assertLess(res_intervened["probability"], res_high["probability"])

    def test_04_batch_sample_and_screening(self):
        sample_resp = self.client.get("/api/batch-sample?limit=15")
        self.assertEqual(sample_resp.status_code, 200)
        csv_bytes = sample_resp.content

        files = {"file": ("students.csv", io.BytesIO(csv_bytes), "text/csv")}
        batch_resp = self.client.post("/api/predict-batch?threshold=0.50", files=files)
        self.assertEqual(batch_resp.status_code, 200)
        batch_data = batch_resp.json()
        self.assertEqual(batch_data["total"], 15)
        self.assertIn("rows", batch_data)
        self.assertEqual(len(batch_data["rows"]), 15)

    def test_05_experiments_and_figures(self):
        exp = self.client.get("/api/experiments").json()
        self.assertGreaterEqual(len(exp["results"]), 8)
        self.assertGreaterEqual(len(exp["tuning"]), 10)
        self.assertEqual(len(exp["figures"]), 7)

        # Check figure delivery
        fig_filename = exp["figures"][0]["filename"]
        fig_resp = self.client.get(f"/figures/{fig_filename}")
        self.assertEqual(fig_resp.status_code, 200)

    def test_06_error_analysis_and_detail(self):
        res = self.client.get("/api/test-samples?page=1&page_size=10&status=ALL").json()
        self.assertEqual(len(res["items"]), 10)
        self.assertEqual(res["total_all"], 885)

        first_id = res["items"][0]["student_index"]
        detail_res = self.client.get(f"/api/test-samples/{first_id}").json()
        self.assertEqual(detail_res["student_index"], first_id)
        self.assertIn("raw_features", detail_res)
        self.assertIn("prediction", detail_res)


if __name__ == "__main__":
    unittest.main()
