/* ==========================================================================
   EduGuard AI — Frontend Application Logic
   ========================================================================== */

(() => {
  "use strict";

  const VIEW_META = {
    dashboard: {
      title: "Tổng quan Hệ thống",
      subtitle: "Cảnh báo sớm Sinh viên có Nguy cơ Bỏ học • Tập dữ liệu UCI (N = 4.424)",
    },
    predictor: {
      title: "Dự đoán Hồ sơ & Mô phỏng What-If",
      subtitle: "Tinh chỉnh thông số học vụ, tài chính theo thời gian thực với mô hình Random Forest (Luồng B)",
    },
    batch: {
      title: "Sàng lọc Nguy cơ Hàng loạt (CSV)",
      subtitle: "Tải danh sách sinh viên để phân tầng rủi ro 3 mức và xuất kế hoạch can thiệp",
    },
    experiments: {
      title: "Thực nghiệm Mô hình & Thư viện Biểu đồ",
      subtitle: "Đối sánh Luồng A vs Luồng B, khảo sát siêu tham số và bộ 7 biểu đồ khoa học",
    },
    errors: {
      title: "Phân tích Lỗi & Tập Kiểm thử (N = 885)",
      subtitle: "Kiểm tra chi tiết các ca True Positive, True Negative, False Negative và False Positive",
    },
  };

  const state = {
    theme: localStorage.getItem("eduguard-theme") || "light",
    overview: null,
    metadata: null,
    experiments: null,
    batchResult: null,
    currentFeatures: {},
    cmThreshold: "0.50",
    expMetric: "F1-score",
    tuningAlgo: "Decision Tree",
    galleryCategory: "ALL",
    errorQuery: {
      page: 1,
      pageSize: 15,
      status: "ALL",
      tier: "ALL",
      search: "",
      sort: "prob_desc",
    },
    modalStudentDetail: null,
    charts: {},
  };

  // =========================================================================
  // Helpers
  // =========================================================================
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  function esc(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function apiGet(url) {
    const res = await fetch(url);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  }

  async function apiPostJson(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  }

  function debounce(fn, ms) {
    let timer = null;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), ms);
    };
  }

  function getChartColors() {
    const isDark = state.theme === "dark";
    return {
      text: isDark ? "#cbd5e1" : "#334155",
      muted: isDark ? "#94a3b8" : "#64748b",
      grid: isDark ? "rgba(148, 163, 184, 0.12)" : "rgba(15, 23, 42, 0.07)",
      surface: isDark ? "#0f172a" : "#ffffff",
      indigo: isDark ? "#818cf8" : "#4f46e5",
      indigoSoft: isDark ? "rgba(129, 140, 248, 0.72)" : "rgba(79, 70, 229, 0.78)",
      emerald: isDark ? "#34d399" : "#059669",
      amber: isDark ? "#fbbf24" : "#d97706",
      rose: isDark ? "#fb7185" : "#e11d48",
      slate: isDark ? "#64748b" : "#94a3b8",
    };
  }

  function mountChart(key, canvasId, config) {
    if (typeof Chart === "undefined") return null;
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    if (state.charts[key]) {
      state.charts[key].destroy();
    }
    state.charts[key] = new Chart(canvas, config);
    return state.charts[key];
  }

  // =========================================================================
  // Theme & Navigation
  // =========================================================================
  function applyTheme(theme) {
    state.theme = theme;
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("eduguard-theme", theme);

    const isDark = theme === "dark";
    $("#themeIconLight").classList.toggle("hidden", !isDark);
    $("#themeIconDark").classList.toggle("hidden", isDark);
    $("#themeLabel").textContent = isDark ? "Chế độ Sáng" : "Chế độ Tối";

    // Re-render charts with updated theme colors
    if (state.overview) renderDashboardCharts();
    if (state.experiments) renderExperimentCharts();
  }

  function switchView(viewName) {
    $$(".nav-item").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.view === viewName);
    });
    $$(".view-section").forEach((sec) => {
      sec.classList.toggle("active", sec.id === `view-${viewName}`);
    });
    const meta = VIEW_META[viewName] || VIEW_META.dashboard;
    $("#viewTitle").textContent = meta.title;
    $("#viewSubtitle").textContent = meta.subtitle;

    // Close mobile drawer
    $("#sidebar").classList.remove("open");
    $("#sidebarBackdrop").classList.remove("show");

    if (viewName === "experiments" && !state.experiments) {
      loadExperiments();
    }
    if (viewName === "errors") {
      loadErrorAnalysis();
    }
  }

  // =========================================================================
  // View 1: Dashboard Overview
  // =========================================================================
  async function loadOverview() {
    const data = await apiGet("/api/overview");
    state.overview = data;

    const ds = data.dataset;
    const best = data.best_model_metrics;

    $("#kpiTotalStudents").textContent = ds.total.toLocaleString("en-US");
    $("#kpiSplitMeta").textContent = `Train: ${ds.train_size.toLocaleString("en-US")} (80%) • Test: ${ds.test_size.toLocaleString("en-US")} (20%)`;

    $("#kpiDropoutRate").textContent = `${ds.dropout_pct}%`;
    $("#kpiDropoutCount").textContent = `${ds.dropout.toLocaleString("en-US")} bỏ học / ${ds.non_dropout.toLocaleString("en-US")} không bỏ học`;

    $("#kpiBestF1").textContent = Number(best["F1-score"]).toFixed(4);
    $("#kpiBestModelMeta").textContent = `Random Forest • Accuracy ${(Number(best.Accuracy) * 100).toFixed(2)}%`;

    $("#kpiBestAuc").textContent = Number(best["ROC-AUC"]).toFixed(4);
    $("#kpiBestPrAuc").textContent = `PR-AUC: ${Number(best["PR-AUC"]).toFixed(4)} • Recall: ${(Number(best.Recall) * 100).toFixed(2)}%`;

    renderConfusionMatrix();
    renderRiskTiers();
    renderCourseStats();
    renderDashboardCharts();
    renderTeamMembers();
  }

  function renderConfusionMatrix() {
    const cm = state.overview.confusion_matrices[state.cmThreshold];
    if (!cm) return;
    $("#cmTn").textContent = cm.tn;
    $("#cmFp").textContent = cm.fp;
    $("#cmFn").textContent = cm.fn;
    $("#cmTp").textContent = cm.tp;

    $("#cmMetricsStrip").innerHTML = `
      <div class="cm-metric-pill"><span>Accuracy</span><strong>${(cm.accuracy * 100).toFixed(2)}%</strong></div>
      <div class="cm-metric-pill"><span>Precision</span><strong>${(cm.precision * 100).toFixed(2)}%</strong></div>
      <div class="cm-metric-pill"><span>Recall</span><strong>${(cm.recall * 100).toFixed(2)}%</strong></div>
      <div class="cm-metric-pill"><span>F1-score</span><strong>${cm.f1_score.toFixed(4)}</strong></div>
    `;
  }

  function renderRiskTiers() {
    const container = $("#riskTierBars");
    const tiers = state.overview.risk_tiers;
    const toneClass = (tier) => {
      if (tier.includes("Cao")) return "high";
      if (tier.includes("Trung")) return "medium";
      return "low";
    };

    container.innerHTML = tiers
      .map(
        (t) => `
      <div>
        <div class="tier-row-head">
          <span><strong>${esc(t.tier)}</strong> — ${t.count} SV (${t.pct_of_test}%)</span>
          <span>Bỏ học thật: <strong>${t.actual_dropout_rate}%</strong> (${t.actual_dropout} SV)</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill ${toneClass(t.tier)}" style="width: ${Math.min(100, t.actual_dropout_rate)}%"></div>
        </div>
      </div>`
      )
      .join("");
  }

  function renderCourseStats() {
    const items = state.overview.course_stats;
    $("#courseStatsGrid").innerHTML = items
      .map((c) => {
        const cls = c.dropout_rate >= 40 ? "high" : c.dropout_rate >= 25 ? "medium" : "low";
        return `
          <div class="course-stat-item">
            <div class="course-stat-name" title="${esc(c.name)}">${esc(c.name)}</div>
            <div class="course-stat-meta">
              <span>${c.total} SV</span>
              <strong>${c.dropout_rate}% bỏ học</strong>
            </div>
            <div class="progress-track">
              <div class="progress-fill ${cls}" style="width: ${Math.min(100, c.dropout_rate)}%"></div>
            </div>
          </div>
        `;
      })
      .join("");
  }

  function renderDashboardCharts() {
    const o = state.overview;
    const c = getChartColors();

    // 1. Target Distribution Doughnut
    mountChart("targetDist", "chartTargetDist", {
      type: "doughnut",
      data: {
        labels: [
          `Graduate (Tốt nghiệp): ${o.dataset.graduate_pct}%`,
          `Dropout (Bỏ học): ${o.dataset.dropout_pct}%`,
          `Enrolled (Đang học): ${o.dataset.enrolled_pct}%`,
        ],
        datasets: [
          {
            data: [o.dataset.graduate, o.dataset.dropout, o.dataset.enrolled],
            backgroundColor: [c.emerald, c.rose, c.indigo],
            borderColor: c.surface,
            borderWidth: 3,
            hoverOffset: 8,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "62%",
        plugins: {
          legend: {
            position: "bottom",
            labels: { color: c.text, padding: 14, font: { family: "Plus Jakarta Sans", size: 12, weight: "600" } },
          },
        },
      },
    });

    // 2. Financial Impact Horizontal Bar
    mountChart("financialImpact", "chartFinancialImpact", {
      type: "bar",
      data: {
        labels: o.financial_impact.map((x) => x.label),
        datasets: [
          {
            label: "Tỷ lệ bỏ học (%)",
            data: o.financial_impact.map((x) => x.dropout_rate),
            backgroundColor: o.financial_impact.map((x) =>
              x.tone === "danger" ? c.rose : x.tone === "warning" ? c.amber : x.tone === "info" ? c.indigo : c.emerald
            ),
            borderRadius: 7,
            barPercentage: 0.65,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            max: 100,
            grid: { color: c.grid },
            ticks: { color: c.muted, callback: (v) => `${v}%` },
          },
          y: {
            grid: { display: false },
            ticks: { color: c.text, font: { size: 12, weight: "600" } },
          },
        },
      },
    });

    // 3. Top 12 Feature Importance
    mountChart("featureImportance", "chartFeatureImportance", {
      type: "bar",
      data: {
        labels: o.feature_importance.map((f) =>
          f.is_engineered ? `★ ${f.label}` : f.label
        ),
        datasets: [
          {
            label: "Gini Importance",
            data: o.feature_importance.map((f) => f.importance),
            backgroundColor: o.feature_importance.map((f) =>
              f.is_engineered ? c.emerald : c.indigoSoft
            ),
            borderRadius: 6,
            barPercentage: 0.72,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => {
                const item = o.feature_importance[ctx.dataIndex];
                return item.is_engineered ? "Đặc trưng mới tạo ở Luồng B" : "Đặc trưng gốc UCI";
              },
            },
          },
        },
        scales: {
          x: { grid: { color: c.grid }, ticks: { color: c.muted } },
          y: { grid: { display: false }, ticks: { color: c.text, font: { size: 11.5 } } },
        },
      },
    });
  }

  // =========================================================================
  // View 2: Student Predictor & What-If Simulator
  // =========================================================================
  async function loadMetadataAndInitPredictor() {
    const meta = await apiGet("/api/metadata");
    state.metadata = meta;

    // Populate dropdowns
    const fillSelect = (id, opts) => {
      const sel = $(id);
      if (!sel || !opts) return;
      sel.innerHTML = opts
        .map((o) => `<option value="${o.value}">${esc(o.label)}</option>`)
        .join("");
    };
    fillSelect("#selectCourse", meta.options.Course);
    fillSelect("#selectMarital", meta.options["Marital status"]);
    fillSelect("#selectAppMode", meta.options["Application mode"]);
    fillSelect("#selectPrevQual", meta.options["Previous qualification"]);

    // Render presets
    $("#presetButtons").innerHTML = meta.presets
      .map(
        (p) => `
      <button type="button" class="preset-btn" data-preset="${esc(p.id)}" title="${esc(p.description)}">
        <span class="preset-dot ${esc(p.id)}"></span>
        <span>${esc(p.name.split(" (")[0])}</span>
      </button>`
      )
      .join("");

    $$("[data-preset]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const preset = meta.presets.find((p) => p.id === btn.dataset.preset);
        if (!preset) return;
        $("#selectedTestActualBanner").classList.add("hidden");
        $("#testSampleSelect").value = "";
        applyFeaturesToForm({ ...meta.defaults, ...preset.values });
        runPrediction();
      });
    });

    // Populate top 120 test samples into quick dropdown
    const testRes = await apiGet("/api/test-samples?page=1&page_size=100&sort=id_asc");
    const testSel = $("#testSampleSelect");
    testSel.innerHTML =
      '<option value="">-- Chọn sinh viên tập Test (N=885) --</option>' +
      testRes.items
        .map(
          (item) =>
            `<option value="${item.student_index}">SV #${item.student_index} — ${
              item.actual_dropout === 1 ? "🔴 Bỏ học thật" : "🟢 Không bỏ học"
            } (${item.probability_pct}%)</option>`
        )
        .join("");

    // Initialize form with Safe or Median defaults
    applyFeaturesToForm(meta.defaults);
    await runPrediction();
  }

  function applyFeaturesToForm(featureObj) {
    state.currentFeatures = { ...state.metadata.defaults, ...featureObj };

    $$(".pred-input").forEach((el) => {
      const field = el.dataset.field;
      if (state.currentFeatures[field] !== undefined) {
        el.value = state.currentFeatures[field];
        if (el.dataset.display) {
          const disp = document.getElementById(el.dataset.display);
          if (disp) disp.textContent = Number(el.value).toFixed(2);
        }
      }
    });

    $$(".segmented-control").forEach((grp) => {
      const field = grp.dataset.field;
      const val = String(Math.round(Number(state.currentFeatures[field] ?? 0)));
      grp.querySelectorAll(".seg-opt").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.val === val);
      });
    });
  }

  function collectFeaturesFromForm() {
    const features = { ...state.currentFeatures };
    $$(".pred-input").forEach((el) => {
      const val = parseFloat(el.value);
      if (!Number.isNaN(val)) {
        features[el.dataset.field] = val;
      }
    });
    return features;
  }

  async function runPrediction() {
    const features = collectFeaturesFromForm();
    state.currentFeatures = features;
    const threshold = parseFloat($("#predThresholdSlider").value || "0.50");

    const res = await apiPostJson("/api/predict", { features, threshold });
    renderPredictionResult(res);
  }

  const debouncedPredict = debounce(() => {
    if ($("#autoPredictToggle").checked) {
      runPrediction();
    }
  }, 140);

  function renderPredictionResult(res) {
    // Animate SVG semi-circle arc (radius = 96, arc length = pi * 96 = 301.59)
    const maxLen = 301.59;
    const pct = Math.max(0, Math.min(1, res.probability));
    const offset = maxLen * (1 - pct);
    $("#gaugeProgressArc").style.strokeDashoffset = offset.toFixed(2);
    $("#gaugePercentText").textContent = `${res.probability_pct.toFixed(1)}%`;

    const badge = $("#predTierBadge");
    badge.textContent = res.risk_tier;
    badge.className = `tier-pill ${res.tier_code}`;

    const classBadge = $("#predClassBadge");
    classBadge.textContent = res.predicted_text;
    classBadge.className = `class-badge ${res.predicted_label === 1 ? "dropout" : "safe"}`;

    const recBox = $("#recommendationBox");
    recBox.className = `recommendation-box ${res.tier_code}`;
    $("#recommendationText").textContent = res.recommendation;

    // Drivers
    const renderDriverList = (elId, list, emptyMsg) => {
      const el = $(elId);
      if (!list || list.length === 0) {
        el.innerHTML = `<li class="driver-item"><span>${esc(emptyMsg)}</span></li>`;
        return;
      }
      el.innerHTML = list
        .map(
          (item) => `
        <li class="driver-item">
          <strong>${esc(item.title)} (${esc(item.value)})</strong>
          <span>${esc(item.detail)}</span>
        </li>`
        )
        .join("");
    };

    renderDriverList("#riskDriversList", res.explanation.risk_drivers, "Không phát hiện yếu tố rủi ro nổi bật.");
    renderDriverList("#protectiveList", res.explanation.protective_factors, "Chưa có yếu tố bảo vệ rõ rệt.");

    // 7 Engineered Features
    $("#engineeredFeaturesGrid").innerHTML = res.engineered_features
      .map(
        (ef) => `
      <div class="eng-card">
        <div class="eng-name">${esc(ef.label)}</div>
        <div class="eng-val">${ef.value}</div>
        <div class="eng-bench">TB An toàn: ${ef.safe_avg ?? "—"} • Bỏ học: ${ef.drop_avg ?? "—"}</div>
      </div>`
      )
      .join("");
  }

  async function loadTestStudentIntoPredictor(studentIndex) {
    const detail = await apiGet(`/api/test-samples/${studentIndex}`);
    applyFeaturesToForm(detail.raw_features);
    const banner = $("#selectedTestActualBanner");
    banner.classList.remove("hidden");
    const isDrop = detail.actual_dropout === 1;
    banner.innerHTML = `
      <div>
        <strong>Hồ sơ Tập kiểm thử #${detail.student_index}</strong> (${esc(detail.course_name)}) —
        Nhãn thực tế: <strong>${isDrop ? "🔴 Bỏ học (1)" : "🟢 Không bỏ học (0)"}</strong> •
        Trạng thái gốc: <code>${esc(detail.prediction_status)}</code>
      </div>
      <span class="chip">Tập Test N=885</span>
    `;
    await runPrediction();
  }

  // =========================================================================
  // View 3: Batch CSV Screening
  // =========================================================================
  async function submitBatchFile(fileOrBlob, filename = "batch.csv") {
    const threshold = $("#batchThresholdSelect").value || "0.50";
    const statusEl = $("#batchStatusMsg");
    statusEl.textContent = `Đang phân tích ${filename}...`;

    const formData = new FormData();
    formData.append("file", fileOrBlob, filename);

    const res = await fetch(`/api/predict-batch?threshold=${encodeURIComponent(threshold)}`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      statusEl.textContent = `Lỗi: ${err.detail || "Không thể đọc file CSV"}`;
      return;
    }

    const data = await res.json();
    state.batchResult = data;
    statusEl.textContent = `Đã sàng lọc xong ${data.total} sinh viên (khớp ${data.matched_columns}/36 cột UCI).`;
    renderBatchResults();
  }

  function renderBatchResults() {
    const data = state.batchResult;
    if (!data) return;

    $("#batchResultsContainer").classList.remove("hidden");
    $("#batchTotalCount").textContent = data.total;
    $("#batchMatchedCols").textContent = `Khớp ${data.matched_columns}/36 cột đặc trưng`;
    $("#batchHighCount").textContent = data.tier_counts.high;
    $("#batchMedCount").textContent = data.tier_counts.medium;
    $("#batchLowCount").textContent = data.tier_counts.low;
    $("#batchAvgProb").textContent = `${data.avg_probability_pct}%`;

    const tierFilter = $("#batchTierFilter").value;
    let rows = [...data.rows].sort((a, b) => b.probability - a.probability);
    if (tierFilter !== "all") {
      rows = rows.filter((r) => r.tier_code === tierFilter);
    }

    $("#batchTableBody").innerHTML = rows
      .map(
        (r) => `
      <tr>
        <td class="mono-val">#${r.row_no}</td>
        <td>${esc(r.course)}</td>
        <td class="mono-val">${r.age}</td>
        <td class="mono-val">${r.approved_1st}</td>
        <td class="mono-val">${r.approved_2nd}</td>
        <td class="mono-val">${r.avg_grade.toFixed(2)}</td>
        <td>${r.tuition_ok === 1 ? '<span class="chip emerald">Đủ</span>' : '<span class="chip rose">Chưa đóng</span>'}</td>
        <td class="mono-val"><strong>${r.probability_pct}%</strong></td>
        <td><span class="tier-pill ${r.tier_code}">${esc(r.risk_tier)}</span></td>
        <td>${esc(r.recommendation)}</td>
      </tr>`
      )
      .join("");
  }

  function exportBatchCsv() {
    if (!state.batchResult) return;
    const headers = [
      "STT",
      "Nganh_dao_tao",
      "Tuoi",
      "Mon_dau_HK1",
      "Mon_dau_HK2",
      "Diem_TB_2_ky",
      "Dong_hoc_phi",
      "Dang_no",
      "Xac_suat_bo_hoc_%",
      "Muc_rui_ro",
      "Khuyen_nghi",
    ];
    const lines = [headers.join(",")];
    for (const r of state.batchResult.rows) {
      const vals = [
        r.row_no,
        `"${String(r.course).replace(/"/g, '""')}"`,
        r.age,
        r.approved_1st,
        r.approved_2nd,
        r.avg_grade,
        r.tuition_ok,
        r.debtor,
        r.probability_pct,
        `"${r.risk_tier}"`,
        `"${String(r.recommendation).replace(/"/g, '""')}"`,
      ];
      lines.push(vals.join(","));
    }
    const blob = new Blob(["\uFEFF" + lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ket_qua_sang_loc_bo_hoc.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  // =========================================================================
  // View 4: Experiments & Gallery
  // =========================================================================
  async function loadExperiments() {
    const data = await apiGet("/api/experiments");
    state.experiments = data;
    renderResultsTable();
    renderGallery();
    renderExperimentCharts();
  }

  function renderResultsTable() {
    const rows = state.experiments.results;
    $("#resultsTableBody").innerHTML = rows
      .map((r) => {
        const isBest = String(r.Flow).includes("Luồng B") && String(r.Model).includes("Random Forest");
        return `
          <tr class="${isBest ? "row-highlight" : ""}">
            <td><strong>${esc(r.Flow)}</strong></td>
            <td>${esc(r.Model)} ${isBest ? '<span class="chip emerald">Tốt nhất</span>' : ""}</td>
            <td class="mono-val">${Number(r.Accuracy).toFixed(4)}</td>
            <td class="mono-val">${Number(r.Precision).toFixed(4)}</td>
            <td class="mono-val">${Number(r.Recall).toFixed(4)}</td>
            <td class="mono-val"><strong>${Number(r["F1-score"]).toFixed(4)}</strong></td>
            <td class="mono-val">${Number(r["ROC-AUC"]).toFixed(4)}</td>
            <td class="mono-val">${Number(r["PR-AUC"]).toFixed(4)}</td>
          </tr>`;
      })
      .join("");
  }

  function renderExperimentCharts() {
    if (!state.experiments) return;
    const c = getChartColors();
    const metric = state.expMetric;
    const results = state.experiments.results;

    const flowA = results.filter((r) => String(r.Flow).includes("Luồng A"));
    const flowB = results.filter((r) => String(r.Flow).includes("Luồng B"));
    const models = flowA.map((r) => r.Model);

    mountChart("flowCompare", "chartFlowCompare", {
      type: "bar",
      data: {
        labels: models,
        datasets: [
          {
            label: "Luồng A (Cơ bản - 36 biến)",
            data: flowA.map((r) => r[metric]),
            backgroundColor: c.slate,
            borderRadius: 6,
          },
          {
            label: "Luồng B (Cải tiến - +7 đặc trưng & OHE)",
            data: flowB.map((r) => r[metric]),
            backgroundColor: c.indigo,
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: c.text } },
        },
        scales: {
          y: {
            min: 0.45,
            max: 1.0,
            grid: { color: c.grid },
            ticks: { color: c.muted },
          },
          x: {
            grid: { display: false },
            ticks: { color: c.text, font: { size: 11 } },
          },
        },
      },
    });

    // Hyperparameter tuning chart
    const tuningRows = state.experiments.tuning.filter((t) => t.Algorithm === state.tuningAlgo);
    mountChart("tuning", "chartTuning", {
      type: "line",
      data: {
        labels: tuningRows.map((t) =>
          t.Hyperparameter_Setting.replace(", weights=distance", "").replace("n_neighbors=", "k=").replace("max_depth=", "depth=")
        ),
        datasets: [
          {
            label: "F1-score",
            data: tuningRows.map((t) => t["F1-score"]),
            borderColor: c.indigo,
            backgroundColor: c.indigo,
            tension: 0.3,
            pointRadius: 5,
          },
          {
            label: "ROC-AUC",
            data: tuningRows.map((t) => t["ROC-AUC"]),
            borderColor: c.emerald,
            backgroundColor: c.emerald,
            tension: 0.3,
            pointRadius: 5,
          },
          {
            label: "Accuracy",
            data: tuningRows.map((t) => t.Accuracy),
            borderColor: c.amber,
            backgroundColor: c.amber,
            tension: 0.3,
            pointRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: c.text } } },
        scales: {
          y: { min: 0.65, max: 0.96, grid: { color: c.grid }, ticks: { color: c.muted } },
          x: { grid: { color: c.grid }, ticks: { color: c.text } },
        },
      },
    });
  }

  function renderGallery() {
    const cat = state.galleryCategory;
    const figs = state.experiments.figures.filter((f) => cat === "ALL" || f.category === cat);

    $("#figuresGallery").innerHTML = figs
      .map(
        (f) => `
      <div class="figure-card" data-fig="${esc(f.filename)}">
        <div class="figure-thumb">
          <img src="/figures/${esc(f.filename)}" alt="${esc(f.title)}" loading="lazy" />
        </div>
        <div class="figure-body">
          <span class="chip">${esc(f.category)}</span>
          <h4>${esc(f.title)}</h4>
          <p>${esc(f.description)}</p>
        </div>
      </div>`
      )
      .join("");

    $$(".figure-card").forEach((card) => {
      card.addEventListener("click", () => {
        const fig = state.experiments.figures.find((x) => x.filename === card.dataset.fig);
        if (!fig) return;
        $("#lightboxTitle").textContent = fig.title;
        $("#lightboxDesc").textContent = fig.description;
        $("#lightboxImg").src = `/figures/${fig.filename}`;
        $("#imageModal").classList.remove("hidden");
      });
    });
  }

  // =========================================================================
  // View 5: Error Analysis (N = 885)
  // =========================================================================
  async function loadErrorAnalysis() {
    const q = state.errorQuery;
    const params = new URLSearchParams({
      page: String(q.page),
      page_size: String(q.pageSize),
      status: q.status,
      tier: q.tier,
      search: q.search,
      sort: q.sort,
    });
    const data = await apiGet(`/api/test-samples?${params.toString()}`);
    renderErrorStatusCards(data.status_counts, data.total_all);
    renderErrorTable(data);
  }

  function renderErrorStatusCards(counts, totalAll) {
    const findCount = (prefix) => {
      const k = Object.keys(counts).find((key) => key.includes(prefix));
      return k ? counts[k] : 0;
    };
    const cards = [
      { key: "ALL", label: "Tất cả tập Test", val: totalAll },
      { key: "True Positive", label: "True Positive (Bắt đúng)", val: findCount("True Positive") },
      { key: "True Negative", label: "True Negative (An toàn)", val: findCount("True Negative") },
      { key: "False Negative", label: "False Negative (Bỏ sót!)", val: findCount("False Negative") },
      { key: "False Positive", label: "False Positive (Báo nhầm)", val: findCount("False Positive") },
    ];

    $("#errorStatusCards").innerHTML = cards
      .map(
        (c) => `
      <button type="button" class="status-card ${state.errorQuery.status === c.key ? "active" : ""}" data-err-status="${esc(c.key)}">
        <div class="status-card-label">${esc(c.label)}</div>
        <div class="status-card-val">${c.val}</div>
      </button>`
      )
      .join("");

    $$("[data-err-status]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.errorQuery.status = btn.dataset.errStatus;
        state.errorQuery.page = 1;
        loadErrorAnalysis();
      });
    });
  }

  function renderErrorTable(data) {
    const statusChip = (s) => {
      if (s.includes("True Positive")) return '<span class="chip indigo">TP • Bắt đúng</span>';
      if (s.includes("True Negative")) return '<span class="chip emerald">TN • An toàn</span>';
      if (s.includes("False Negative")) return '<span class="chip rose">FN • Bỏ sót!</span>';
      return '<span class="chip amber">FP • Báo nhầm</span>';
    };

    $("#errorTableBody").innerHTML = data.items
      .map(
        (item) => `
      <tr>
        <td class="mono-val"><strong>#${item.student_index}</strong></td>
        <td>${esc(item.course_name)}</td>
        <td class="mono-val">${item.approved_1st}/${item.enrolled_1st} • ${item.approved_2nd}/${item.enrolled_2nd}</td>
        <td class="mono-val">${item.grade_2nd.toFixed(1)}</td>
        <td>${item.tuition_ok === 1 ? "Đã đóng" : '<span class="text-rose">Chưa đóng</span>'}${item.debtor === 1 ? ' • <span class="text-rose">Nợ</span>' : ""}</td>
        <td>${item.actual_dropout === 1 ? '<span class="chip rose">Bỏ học (1)</span>' : '<span class="chip emerald">An toàn (0)</span>'}</td>
        <td class="mono-val"><strong>${item.probability_pct}%</strong></td>
        <td>${statusChip(item.prediction_status)}</td>
        <td>
          <div class="btn-row">
            <button type="button" class="btn btn-ghost btn-sm" data-inspect-sv="${item.student_index}">Chi tiết</button>
          </div>
        </td>
      </tr>`
      )
      .join("");

    $("#errPageInfo").textContent = `Trang ${data.page} / ${data.total_pages} • Tổng ${data.total} sinh viên phù hợp`;
    $("#errPrevBtn").disabled = data.page <= 1;
    $("#errNextBtn").disabled = data.page >= data.total_pages;

    $$("[data-inspect-sv]").forEach((btn) => {
      btn.addEventListener("click", () => openStudentModal(Number(btn.dataset.inspectSv)));
    });
  }

  async function openStudentModal(studentIndex) {
    const detail = await apiGet(`/api/test-samples/${studentIndex}`);
    state.modalStudentDetail = detail;

    $("#modalStudentTitle").textContent = `Hồ sơ Sinh viên #${detail.student_index} — ${detail.course_name}`;
    $("#modalStudentSub").textContent = `Thực tế: ${detail.actual_text} • Trạng thái: ${detail.prediction_status}`;

    const pred = detail.prediction;
    const raw = detail.raw_features;

    $("#modalStudentBody").innerHTML = `
      <div class="kpi-grid" style="grid-template-columns: repeat(3, 1fr); margin-bottom: 14px;">
        <div class="kpi-card">
          <div class="kpi-header"><span>Xác suất dự đoán</span></div>
          <div class="kpi-value">${pred.probability_pct}%</div>
          <div class="kpi-sub">${esc(pred.risk_tier)}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-header"><span>HK1 (Đậu/ĐK • TB)</span></div>
          <div class="kpi-value">${raw["Curricular units 1st sem (approved)"]}/${raw["Curricular units 1st sem (enrolled)"]}</div>
          <div class="kpi-sub">Điểm TB: ${Number(raw["Curricular units 1st sem (grade)"]).toFixed(2)}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-header"><span>HK2 (Đậu/ĐK • TB)</span></div>
          <div class="kpi-value">${raw["Curricular units 2nd sem (approved)"]}/${raw["Curricular units 2nd sem (enrolled)"]}</div>
          <div class="kpi-sub">Điểm TB: ${Number(raw["Curricular units 2nd sem (grade)"]).toFixed(2)}</div>
        </div>
      </div>

      <div class="recommendation-box ${pred.tier_code}">
        <div class="rec-title">Khuyến nghị Cố vấn học tập</div>
        <p>${esc(pred.recommendation)}</p>
      </div>

      <h4 class="mini-section-title">7 Đặc trưng Luồng B</h4>
      <div class="engineered-grid">
        ${pred.engineered_features
          .map(
            (ef) => `
          <div class="eng-card">
            <div class="eng-name">${esc(ef.label)}</div>
            <div class="eng-val">${ef.value}</div>
          </div>`
          )
          .join("")}
      </div>
    `;

    $("#studentModal").classList.remove("hidden");
  }

  function renderTeamMembers() {
    if (!state.overview) return;
    const members = state.overview.project.members;
    $("#teamMembersList").innerHTML = members
      .map(
        (m) => `
      <div class="team-member">
        <div>
          <strong>${esc(m.name)}</strong> <span class="chip">${esc(m.class)}</span>
          <div class="card-sub">${esc(m.role)}</div>
        </div>
        <span class="mono-val">${esc(m.id)}</span>
      </div>`
      )
      .join("");
  }

  // =========================================================================
  // Event Wiring & Bootstrap
  // =========================================================================
  function bindEvents() {
    // Sidebar navigation
    $$(".nav-item").forEach((btn) => {
      btn.addEventListener("click", () => switchView(btn.dataset.view));
    });
    $$("[data-jump]").forEach((btn) => {
      btn.addEventListener("click", () => switchView(btn.dataset.jump));
    });

    // Mobile drawer
    $("#mobileMenuBtn").addEventListener("click", () => {
      $("#sidebar").classList.add("open");
      $("#sidebarBackdrop").classList.add("show");
    });
    $("#sidebarBackdrop").addEventListener("click", () => {
      $("#sidebar").classList.remove("open");
      $("#sidebarBackdrop").classList.remove("show");
    });

    // Theme toggle
    $("#themeToggleBtn").addEventListener("click", () => {
      applyTheme(state.theme === "dark" ? "light" : "dark");
    });

    // Confusion matrix threshold toggle
    $$("#cmThresholdToggle .seg-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        $$("#cmThresholdToggle .seg-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.cmThreshold = btn.dataset.th;
        renderConfusionMatrix();
      });
    });

    // Predictor inputs
    $$(".pred-input").forEach((el) => {
      const evt = el.type === "range" || el.type === "number" ? "input" : "change";
      el.addEventListener(evt, () => {
        if (el.dataset.display) {
          const disp = document.getElementById(el.dataset.display);
          if (disp) disp.textContent = Number(el.value).toFixed(2);
        }
        debouncedPredict();
      });
    });

    $$(".segmented-control").forEach((grp) => {
      grp.querySelectorAll(".seg-opt").forEach((btn) => {
        btn.addEventListener("click", () => {
          grp.querySelectorAll(".seg-opt").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          state.currentFeatures[grp.dataset.field] = Number(btn.dataset.val);
          debouncedPredict();
        });
      });
    });

    $("#predThresholdSlider").addEventListener("input", () => {
      $("#predThresholdVal").textContent = Number($("#predThresholdSlider").value).toFixed(2);
      debouncedPredict();
    });

    $$("[data-set-th]").forEach((btn) => {
      btn.addEventListener("click", () => {
        $("#predThresholdSlider").value = btn.dataset.setTh;
        $("#predThresholdVal").textContent = btn.dataset.setTh;
        runPrediction();
      });
    });

    $("#runPredictBtn").addEventListener("click", runPrediction);

    $("#resetMedianBtn").addEventListener("click", () => {
      $("#selectedTestActualBanner").classList.add("hidden");
      $("#testSampleSelect").value = "";
      applyFeaturesToForm(state.metadata.defaults);
      runPrediction();
    });

    $("#testSampleSelect").addEventListener("change", (e) => {
      if (e.target.value) {
        loadTestStudentIntoPredictor(Number(e.target.value));
      }
    });

    // Batch upload
    const dropzone = $("#csvDropzone");
    const fileInput = $("#csvFileInput");
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      const file = e.dataTransfer.files[0];
      if (file) submitBatchFile(file, file.name);
    });
    fileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) submitBatchFile(file, file.name);
    });

    $("#runDemoBatchBtn").addEventListener("click", async () => {
      const resp = await fetch("/api/batch-sample?limit=25");
      const blob = await resp.blob();
      await submitBatchFile(blob, "mau_25_sinh_vien.csv");
    });

    $("#batchTierFilter").addEventListener("change", renderBatchResults);
    $("#exportBatchCsvBtn").addEventListener("click", exportBatchCsv);

    // Experiments controls
    $("#expMetricSelect").addEventListener("change", (e) => {
      state.expMetric = e.target.value;
      renderExperimentCharts();
    });

    $$("#tuningAlgoToggle .seg-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        $$("#tuningAlgoToggle .seg-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.tuningAlgo = btn.dataset.algo;
        renderExperimentCharts();
      });
    });

    $$("#galleryFilter .seg-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        $$("#galleryFilter .seg-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.galleryCategory = btn.dataset.cat;
        renderGallery();
      });
    });

    // Error analysis filters
    const debouncedErrSearch = debounce(() => {
      state.errorQuery.search = $("#errSearchInput").value;
      state.errorQuery.page = 1;
      loadErrorAnalysis();
    }, 250);
    $("#errSearchInput").addEventListener("input", debouncedErrSearch);

    $("#errTierFilter").addEventListener("change", (e) => {
      state.errorQuery.tier = e.target.value;
      state.errorQuery.page = 1;
      loadErrorAnalysis();
    });
    $("#errSortSelect").addEventListener("change", (e) => {
      state.errorQuery.sort = e.target.value;
      state.errorQuery.page = 1;
      loadErrorAnalysis();
    });
    $("#errPrevBtn").addEventListener("click", () => {
      if (state.errorQuery.page > 1) {
        state.errorQuery.page -= 1;
        loadErrorAnalysis();
      }
    });
    $("#errNextBtn").addEventListener("click", () => {
      state.errorQuery.page += 1;
      loadErrorAnalysis();
    });

    // Modals
    $$("[data-close-modal]").forEach((el) => {
      el.addEventListener("click", () => {
        document.getElementById(el.dataset.closeModal).classList.add("hidden");
      });
    });
    $("#openTeamBtn").addEventListener("click", () => {
      $("#teamModal").classList.remove("hidden");
    });
    $("#modalLoadWhatIfBtn").addEventListener("click", async () => {
      if (!state.modalStudentDetail) return;
      $("#studentModal").classList.add("hidden");
      switchView("predictor");
      await loadTestStudentIntoPredictor(state.modalStudentDetail.student_index);
    });
  }

  async function init() {
    applyTheme(state.theme);
    bindEvents();
    try {
      await Promise.all([loadOverview(), loadMetadataAndInitPredictor()]);
    } catch (err) {
      console.error("Initialization error:", err);
      alert("Không thể kết nối tới API máy chủ: " + err.message);
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();

