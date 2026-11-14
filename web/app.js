/* ==========================================================================
   EduGuard AI — Frontend Application Logic (v2.0)
   Academic Retention & Early-Warning Machine Learning Cockpit
   FIT4013 - Nhóm 9 (2026–2027)
   ========================================================================== */

(() => {
  "use strict";

  // Navigation View Metadata
  const VIEW_TITLES = {
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

  // Application Global State
  const state = {
    theme: localStorage.getItem("eduguard-theme") || "light",
    activeView: "dashboard",
    overview: null,
    metadata: null,
    experiments: null,
    batchData: null,
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
      course: null,
      search: "",
      sort: "prob_desc",
    },
    currentModalStudent: null,
    charts: {},
  };

  // DOM Helpers
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));

  function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function debounce(fn, delayMs) {
    let timer = null;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), delayMs);
    };
  }

  // Toast Notification System
  function showToast(message, type = "info") {
    const container = $("#toastContainer");
    if (!container) return;
    const toast = document.createElement("div");
    toast.className = `toast-item toast-${type}`;
    toast.innerHTML = `<span>${escapeHtml(message)}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(20px)";
      toast.style.transition = "all 0.25s ease";
      setTimeout(() => toast.remove(), 250);
    }, 3500);
  }

  // API Call Helpers
  async function apiGet(url) {
    const resp = await fetch(url);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `Lỗi máy chủ HTTP ${resp.status}`);
    }
    return resp.json();
  }

  async function apiPostJson(url, data) {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `Lỗi máy chủ HTTP ${resp.status}`);
    }
    return resp.json();
  }

  // Color Tokens for Chart.js
  function getThemeColors() {
    const isDark = state.theme === "dark";
    return {
      text: isDark ? "#cbd5e1" : "#475569",
      textMuted: isDark ? "#64748b" : "#94a3b8",
      grid: isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(15, 23, 42, 0.06)",
      tooltipBg: isDark ? "#1e293b" : "#ffffff",
      tooltipText: isDark ? "#f8fafc" : "#0f172a",
      tooltipBorder: isDark ? "#334155" : "#e2e8f0",
      blue: isDark ? "#60a5fa" : "#2563eb",
      emerald: isDark ? "#34d399" : "#059669",
      amber: isDark ? "#fbbf24" : "#d97706",
      crimson: isDark ? "#f87171" : "#dc2626",
      purple: isDark ? "#c084fc" : "#7c3aed",
      slate: isDark ? "#94a3b8" : "#64748b",
    };
  }

  // =========================================================================
  // Theme Management (Light / Dark)
  // =========================================================================
  function applyTheme(newTheme) {
    state.theme = newTheme;
    document.documentElement.setAttribute("data-theme", newTheme);
    localStorage.setItem("eduguard-theme", newTheme);

    const isDark = newTheme === "dark";
    const lightIcon = $("#themeIconLight");
    const darkIcon = $("#themeIconDark");
    const themeLabel = $("#themeLabel");

    if (lightIcon && darkIcon) {
      if (isDark) {
        lightIcon.classList.add("hidden");
        darkIcon.classList.remove("hidden");
        if (themeLabel) themeLabel.textContent = "Chế độ Sáng";
      } else {
        lightIcon.classList.remove("hidden");
        darkIcon.classList.add("hidden");
        if (themeLabel) themeLabel.textContent = "Chế độ Tối";
      }
    }

    // Refresh active charts with updated theme colors
    Object.values(state.charts).forEach((chart) => {
      if (chart && typeof chart.update === "function") {
        updateChartTheme(chart);
        chart.update();
      }
    });
  }

  function updateChartTheme(chart) {
    const c = getThemeColors();
    if (chart.options.scales) {
      Object.values(chart.options.scales).forEach((scale) => {
        if (scale.ticks) scale.ticks.color = c.text;
        if (scale.grid) scale.grid.color = c.grid;
      });
    }
    if (chart.options.plugins?.legend?.labels) {
      chart.options.plugins.legend.labels.color = c.text;
    }
  }

  function toggleTheme() {
    applyTheme(state.theme === "dark" ? "light" : "dark");
  }

  // =========================================================================
  // View Router
  // =========================================================================
  function switchView(viewName) {
    if (!VIEW_TITLES[viewName]) return;
    state.activeView = viewName;

    // Update navigation active state
    $$(".nav-tab").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.view === viewName);
    });

    // Update view containers
    $$(".page-view").forEach((view) => {
      view.classList.toggle("active", view.id === `view-${viewName}`);
    });

    // Update titles
    const meta = VIEW_TITLES[viewName];
    const viewTitle = $("#viewTitle");
    const viewSubtitle = $("#viewSubtitle");
    if (viewTitle) viewTitle.textContent = meta.title;
    if (viewSubtitle) viewSubtitle.textContent = meta.subtitle;

    // Close mobile sidebar if open
    closeMobileSidebar();

    // Trigger lazy loading or layout updates per view
    if (viewName === "experiments" && !state.experiments) {
      loadExperimentsData();
    } else if (viewName === "errors") {
      fetchErrorSamples();
    }
  }

  function openMobileSidebar() {
    $("#sidebar")?.classList.add("mobile-open");
    $("#sidebarBackdrop")?.classList.add("mobile-open");
  }

  function closeMobileSidebar() {
    $("#sidebar")?.classList.remove("mobile-open");
    $("#sidebarBackdrop")?.classList.remove("mobile-open");
  }

  // =========================================================================
  // View 1: Executive Dashboard Initialization
  // =========================================================================
  async function loadOverviewData() {
    try {
      const data = await apiGet("/api/overview");
      state.overview = data;
      renderOverviewKPIs(data);
      renderTargetDistributionChart(data.dataset);
      renderFinancialImpactChart(data.financial_impact);
      renderFeatureImportanceChart(data.feature_importance);
      renderConfusionMatrixAndTiers(data);
      renderCourseStats(data.course_stats);
      populateTeamModal(data.project);
    } catch (err) {
      console.error("Lỗi nạp dữ liệu Overview:", err);
      showToast("Không thể nạp dữ liệu Tổng quan: " + err.message, "danger");
    }
  }

  function renderOverviewKPIs(data) {
    const ds = data.dataset;
    const bm = data.best_model_metrics;

    if ($("#kpiTotalStudents")) $("#kpiTotalStudents").textContent = ds.total.toLocaleString("vi-VN");
    if ($("#kpiSplitMeta")) {
      $("#kpiSplitMeta").textContent = `Huấn luyện: ${ds.train_size.toLocaleString("vi-VN")} (80%) • Kiểm thử: ${ds.test_size.toLocaleString("vi-VN")} (20%)`;
    }

    if ($("#kpiDropoutRate")) $("#kpiDropoutRate").textContent = `${ds.dropout_pct}%`;
    if ($("#kpiDropoutCount")) {
      $("#kpiDropoutCount").textContent = `${ds.dropout.toLocaleString("vi-VN")} ca bỏ học / ${ds.non_dropout.toLocaleString("vi-VN")} an toàn`;
    }

    if ($("#kpiBestF1")) $("#kpiBestF1").textContent = bm["F1-score"].toFixed(4);
    if ($("#kpiBestModelMeta")) {
      $("#kpiBestModelMeta").textContent = `Random Forest • Accuracy: ${(bm["Accuracy"] * 100).toFixed(2)}%`;
    }

    if ($("#kpiBestAuc")) $("#kpiBestAuc").textContent = bm["ROC-AUC"].toFixed(4);
    if ($("#kpiBestPrAuc")) {
      $("#kpiBestPrAuc").textContent = `PR-AUC: ${bm["PR-AUC"].toFixed(4)} • Recall: ${(bm["Recall"] * 100).toFixed(2)}%`;
    }
  }

  function renderTargetDistributionChart(dataset) {
    const ctx = $("#chartTargetDist");
    if (!ctx) return;
    const colors = getThemeColors();

    if (state.charts.targetDist) state.charts.targetDist.destroy();

    state.charts.targetDist = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: ["Tốt nghiệp (Graduate)", "Bỏ học (Dropout)", "Đang theo học (Enrolled)"],
        datasets: [
          {
            data: [dataset.graduate, dataset.dropout, dataset.enrolled],
            backgroundColor: [colors.emerald, colors.crimson, colors.blue],
            borderWidth: 2,
            borderColor: colors.tooltipBg,
            hoverOffset: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "68%",
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              boxWidth: 12,
              padding: 12,
              font: { family: "Plus Jakarta Sans", size: 12, weight: 600 },
              color: colors.text,
            },
          },
          tooltip: {
            backgroundColor: colors.tooltipBg,
            titleColor: colors.tooltipText,
            bodyColor: colors.tooltipText,
            borderColor: colors.tooltipBorder,
            borderWidth: 1,
            callbacks: {
              label: (item) => {
                const total = dataset.total;
                const val = item.raw;
                const pct = ((val / total) * 100).toFixed(1);
                return ` ${val.toLocaleString("vi-VN")} SV (${pct}%)`;
              },
            },
          },
        },
      },
    });
  }

  function renderFinancialImpactChart(financialItems) {
    const ctx = $("#chartFinancialImpact");
    if (!ctx) return;
    const colors = getThemeColors();

    if (state.charts.financialImpact) state.charts.financialImpact.destroy();

    const labels = financialItems.map((item) => item.label);
    const rates = financialItems.map((item) => item.dropout_rate);
    const bgColors = financialItems.map((item) => {
      if (item.tone === "danger") return colors.crimson;
      if (item.tone === "warning") return colors.amber;
      if (item.tone === "info") return colors.blue;
      return colors.emerald;
    });

    state.charts.financialImpact = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Tỷ lệ Bỏ học (%)",
            data: rates,
            backgroundColor: bgColors,
            borderRadius: 6,
            maxBarThickness: 28,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: "y",
        scales: {
          x: {
            min: 0,
            max: 100,
            ticks: {
              callback: (val) => `${val}%`,
              color: colors.text,
              font: { family: "JetBrains Mono", size: 11 },
            },
            grid: { color: colors.grid },
          },
          y: {
            ticks: {
              color: colors.text,
              font: { family: "Plus Jakarta Sans", size: 12, weight: 600 },
            },
            grid: { display: false },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: colors.tooltipBg,
            titleColor: colors.tooltipText,
            bodyColor: colors.tooltipText,
            borderColor: colors.tooltipBorder,
            borderWidth: 1,
            callbacks: {
              label: (item) => ` Tỷ lệ bỏ học: ${item.raw}%`,
            },
          },
        },
      },
    });
  }

  function renderFeatureImportanceChart(features) {
    const ctx = $("#chartFeatureImportance");
    if (!ctx) return;
    const colors = getThemeColors();

    if (state.charts.featureImp) state.charts.featureImp.destroy();

    const labels = features.map((f) => f.label);
    const values = features.map((f) => (f.importance * 100).toFixed(2));
    const barColors = features.map((f) => (f.is_engineered ? colors.blue : colors.slate));

    state.charts.featureImp = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Trọng số Đóng góp (%)",
            data: values,
            backgroundColor: barColors,
            borderRadius: 6,
            maxBarThickness: 20,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: "y",
        scales: {
          x: {
            ticks: {
              callback: (val) => `${val}%`,
              color: colors.text,
              font: { family: "JetBrains Mono", size: 11 },
            },
            grid: { color: colors.grid },
          },
          y: {
            ticks: {
              color: colors.text,
              font: { family: "Plus Jakarta Sans", size: 11, weight: 600 },
            },
            grid: { display: false },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: colors.tooltipBg,
            titleColor: colors.tooltipText,
            bodyColor: colors.tooltipText,
            borderColor: colors.tooltipBorder,
            borderWidth: 1,
            callbacks: {
              afterLabel: (item) => {
                const feat = features[item.dataIndex];
                return feat.is_engineered ? "★ Đặc trưng Miền giáo dục tạo mới (Luồng B)" : "Thuộc tính gốc UCI";
              },
            },
          },
        },
      },
    });
  }

  function renderConfusionMatrixAndTiers(data) {
    updateConfusionMatrixView(state.cmThreshold);

    // Render Risk Tiers Distribution
    const tierWrap = $("#riskTierBars");
    if (!tierWrap || !data.risk_tiers) return;

    tierWrap.innerHTML = data.risk_tiers
      .map((tier) => {
        const fillClass = tier.tier.includes("Thấp") ? "fill-low" : tier.tier.includes("Trung bình") ? "fill-med" : "fill-high";
        return `
          <div class="tier-bar-row">
            <div class="tier-bar-header">
              <span><strong>${escapeHtml(tier.tier)}</strong></span>
              <span class="numeric-val">${tier.count} SV (${tier.pct_of_test}%) • Bỏ học thật: <strong>${tier.actual_dropout_rate}%</strong></span>
            </div>
            <div class="tier-bar-track">
              <div class="tier-bar-fill ${fillClass}" style="width: ${tier.pct_of_test}%;"></div>
            </div>
          </div>
        `;
      })
      .join("");
  }

  function updateConfusionMatrixView(threshold) {
    if (!state.overview || !state.overview.confusion_matrices) return;
    const cm = state.overview.confusion_matrices[threshold];
    if (!cm) return;

    if ($("#cmTn")) $("#cmTn").textContent = cm.tn.toLocaleString("vi-VN");
    if ($("#cmFp")) $("#cmFp").textContent = cm.fp.toLocaleString("vi-VN");
    if ($("#cmFn")) $("#cmFn").textContent = cm.fn.toLocaleString("vi-VN");
    if ($("#cmTp")) $("#cmTp").textContent = cm.tp.toLocaleString("vi-VN");

    const strip = $("#cmMetricsStrip");
    if (strip) {
      strip.innerHTML = `
        <div class="strip-stat-item">
          <span class="stat-title">Độ chính xác (Acc)</span>
          <span class="stat-num">${(cm.accuracy * 100).toFixed(2)}%</span>
        </div>
        <div class="strip-stat-item">
          <span class="stat-title">Độ chuẩn xác (Prec)</span>
          <span class="stat-num">${(cm.precision * 100).toFixed(2)}%</span>
        </div>
        <div class="strip-stat-item">
          <span class="stat-title">Độ nhạy (Recall)</span>
          <span class="stat-num">${(cm.recall * 100).toFixed(2)}%</span>
        </div>
        <div class="strip-stat-item">
          <span class="stat-title">F1-Score</span>
          <span class="stat-num">${cm.f1_score.toFixed(4)}</span>
        </div>
      `;
    }
  }

  function renderCourseStats(courses) {
    const grid = $("#courseStatsGrid");
    if (!grid || !courses) return;

    grid.innerHTML = courses
      .map((c) => {
        const rateColor = c.dropout_rate >= 40 ? "text-crimson" : c.dropout_rate >= 25 ? "text-amber" : "text-emerald";
        return `
          <div class="course-card-item">
            <div class="course-name-title" title="${escapeHtml(c.name)}">${escapeHtml(c.name)}</div>
            <div class="course-meta-row">
              <span class="tile-sub">Quy mô: <strong>${c.total}</strong> SV</span>
              <span class="course-dropout-pct ${rateColor}">${c.dropout_rate}% bỏ học</span>
            </div>
          </div>
        `;
      })
      .join("");
  }

  // =========================================================================
  // View 2: Predictive Studio & What-If Simulator
  // =========================================================================
  async function loadPredictorMetadata() {
    try {
      const data = await apiGet("/api/metadata");
      state.metadata = data;
      state.currentFeatures = { ...data.defaults };

      populateSelectOptions(data.options);
      renderPresetButtons(data.presets);
      loadTestSampleDropdown();
      bindInputEvents();

      // Trigger initial prediction
      runSinglePrediction();
    } catch (err) {
      console.error("Lỗi nạp metadata:", err);
      showToast("Không thể tải cấu hình dự báo: " + err.message, "danger");
    }
  }

  function populateSelectOptions(options) {
    const courseSelect = $("#selectCourse");
    if (courseSelect && options.Course) {
      courseSelect.innerHTML = options.Course.map((opt) => `<option value="${opt.value}">${escapeHtml(opt.label)}</option>`).join("");
    }
    const maritalSelect = $("#selectMarital");
    if (maritalSelect && options["Marital status"]) {
      maritalSelect.innerHTML = options["Marital status"].map((opt) => `<option value="${opt.value}">${escapeHtml(opt.label)}</option>`).join("");
    }
    const appModeSelect = $("#selectAppMode");
    if (appModeSelect && options["Application mode"]) {
      appModeSelect.innerHTML = options["Application mode"].map((opt) => `<option value="${opt.value}">${escapeHtml(opt.label)}</option>`).join("");
    }
    const prevQualSelect = $("#selectPrevQual");
    if (prevQualSelect && options["Previous qualification"]) {
      prevQualSelect.innerHTML = options["Previous qualification"].map((opt) => `<option value="${opt.value}">${escapeHtml(opt.label)}</option>`).join("");
    }
  }

  function renderPresetButtons(presets) {
    const container = $("#presetButtons");
    if (!container || !presets) return;

    container.innerHTML = presets
      .map(
        (p, idx) => `
          <button type="button" class="preset-pill-btn ${idx === 0 ? "active" : ""}" data-preset-idx="${idx}">
            ${escapeHtml(p.name)}
          </button>
        `
      )
      .join("");

    container.addEventListener("click", (e) => {
      const btn = e.target.closest(".preset-pill-btn");
      if (!btn) return;
      $$(".preset-pill-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      const idx = parseInt(btn.dataset.presetIdx, 10);
      const preset = state.metadata.presets[idx];
      if (preset) {
        applyFeaturesToForm(preset.features);
        $("#selectedTestActualBanner")?.classList.add("hidden");
        showToast(`Đã áp dụng mẫu: ${preset.name}`, "info");
        runSinglePrediction();
      }
    });
  }

  async function loadTestSampleDropdown() {
    try {
      const data = await apiGet("/api/test-samples?page=1&page_size=35&sort=prob_desc");
      const sel = $("#testSampleSelect");
      if (!sel || !data.items) return;

      sel.innerHTML =
        '<option value="">-- Chọn sinh viên từ tập Test (N=885) --</option>' +
        data.items
          .map(
            (item) => `
            <option value="${item.student_index}">
              #${item.student_index} • ${escapeHtml(item.course_name)} (${item.actual_dropout === 1 ? "Bỏ học" : "An toàn"} - P: ${(item.probability * 100).toFixed(0)}%)
            </option>
          `
          )
          .join("");
    } catch (err) {
      console.warn("Không thể nạp danh sách SV kiểm thử:", err);
    }
  }

  function bindInputEvents() {
    // Inputs with debounce for What-If
    const debouncedPredict = debounce(() => {
      if ($("#autoPredictToggle")?.checked) {
        runSinglePrediction();
      }
    }, 280);

    $$(".pred-input").forEach((input) => {
      input.addEventListener("input", (e) => {
        const field = e.target.dataset.field;
        const val = parseFloat(e.target.value) || 0;
        state.currentFeatures[field] = val;

        // Update slider display value if linked
        const displayId = e.target.dataset.display;
        if (displayId) {
          const disp = $(`#${displayId}`);
          if (disp) disp.textContent = val.toFixed(1);
        }

        debouncedPredict();
      });
    });

    // Segmented switches (Finance: Tuition, Debtor, Scholarship)
    $$(".segmented-control").forEach((ctrl) => {
      ctrl.addEventListener("click", (e) => {
        const btn = e.target.closest(".seg-item");
        if (!btn) return;
        ctrl.querySelectorAll(".seg-item").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");

        const field = ctrl.dataset.field;
        const val = parseInt(btn.dataset.val, 10);
        state.currentFeatures[field] = val;

        if ($("#autoPredictToggle")?.checked) {
          runSinglePrediction();
        }
      });
    });

    // Decision Threshold Slider
    const thSlider = $("#predThresholdSlider");
    if (thSlider) {
      thSlider.addEventListener("input", (e) => {
        const th = parseFloat(e.target.value);
        if ($("#predThresholdVal")) $("#predThresholdVal").textContent = th.toFixed(2);
        runSinglePrediction();
      });
    }

    // Quick threshold buttons
    $$(".threshold-quick-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const th = parseFloat(btn.dataset.setTh);
        if (thSlider) thSlider.value = th;
        if ($("#predThresholdVal")) $("#predThresholdVal").textContent = th.toFixed(2);
        runSinglePrediction();
      });
    });

    // Reset Median Button
    $("#resetMedianBtn")?.addEventListener("click", () => {
      if (state.metadata?.defaults) {
        applyFeaturesToForm(state.metadata.defaults);
        $("#selectedTestActualBanner")?.classList.add("hidden");
        showToast("Đã khôi phục toàn bộ thông số về giá trị trung vị gốc", "info");
        runSinglePrediction();
      }
    });

    // Manual Run Prediction Button
    $("#runPredictBtn")?.addEventListener("click", () => {
      runSinglePrediction();
    });

    // Test sample select change
    $("#testSampleSelect")?.addEventListener("change", async (e) => {
      const idx = e.target.value;
      if (!idx) return;
      try {
        const detail = await apiGet(`/api/test-samples/${idx}`);
        applyFeaturesToForm(detail.raw_features || detail.features);

        // Show actual ground truth banner
        const banner = $("#selectedTestActualBanner");
        if (banner) {
          banner.classList.remove("hidden");
          const isDrop = detail.actual_dropout === 1;
          banner.innerHTML = `
            <div>
              <strong>Hồ sơ #${detail.student_index} (${escapeHtml(detail.course_name)})</strong> — 
              Thực tế: <span class="table-risk-pill ${isDrop ? "pill-high" : "pill-low"}">${isDrop ? "Bỏ học (Dropout)" : "Tốt nghiệp / An toàn"}</span>
            </div>
            <button class="btn btn-outline btn-sm" id="closeActualBanner">✕ Đóng</button>
          `;
          $("#closeActualBanner")?.addEventListener("click", () => banner.classList.add("hidden"));
        }

        showToast(`Đã nạp hồ sơ sinh viên #${idx}`, "info");
        runSinglePrediction();
      } catch (err) {
        showToast("Lỗi nạp hồ sơ SV: " + err.message, "danger");
      }
    });

    // What-If Intervention Action Triggers
    $("#actionPayTuition")?.addEventListener("click", () => {
      const oldProb = state.lastProbability || 0.5;
      state.currentFeatures["Tuition fees up to date"] = 1;
      state.currentFeatures["Debtor"] = 0;
      updateControlVisual("Tuition fees up to date", 1);
      updateControlVisual("Debtor", 0);

      runSinglePrediction().then((newProb) => {
        if (newProb !== null) {
          const diff = ((oldProb - newProb) * 100).toFixed(1);
          showToast(`Đã áp dụng hỗ trợ học phí! Rủi ro giảm ${diff}%`, "success");
        }
      });
    });

    $("#actionGrantScholarship")?.addEventListener("click", () => {
      const oldProb = state.lastProbability || 0.5;
      state.currentFeatures["Scholarship holder"] = 1;
      updateControlVisual("Scholarship holder", 1);

      runSinglePrediction().then((newProb) => {
        if (newProb !== null) {
          const diff = ((oldProb - newProb) * 100).toFixed(1);
          showToast(`Đã cấp học bổng sinh viên! Rủi ro giảm ${diff}%`, "success");
        }
      });
    });

    $("#actionBoostSem2")?.addEventListener("click", () => {
      const oldProb = state.lastProbability || 0.5;
      const curApproved = parseFloat(state.currentFeatures["Curricular units 2nd sem (approved)"]) || 0;
      const curGrade = parseFloat(state.currentFeatures["Curricular units 2nd sem (grade)"]) || 10;

      const newApproved = Math.min(curApproved + 3, 7);
      const newGrade = Math.min(curGrade + 2.5, 18.0);

      state.currentFeatures["Curricular units 2nd sem (approved)"] = newApproved;
      state.currentFeatures["Curricular units 2nd sem (grade)"] = newGrade;

      // Update input fields
      const inputApp = $(`[data-field="Curricular units 2nd sem (approved)"]`);
      if (inputApp) inputApp.value = newApproved;
      const inputGrade = $(`[data-field="Curricular units 2nd sem (grade)"]`);
      if (inputGrade) {
        inputGrade.value = newGrade;
        if ($("#grade2Val")) $("#grade2Val").textContent = newGrade.toFixed(1);
      }

      runSinglePrediction().then((newProb) => {
        if (newProb !== null) {
          const diff = ((oldProb - newProb) * 100).toFixed(1);
          showToast(`Đã phụ đạo học vụ HK2! Rủi ro giảm ${diff}%`, "success");
        }
      });
    });
  }

  function updateControlVisual(field, val) {
    const ctrl = $(`[data-field="${field}"]`);
    if (!ctrl) return;
    if (ctrl.classList.contains("segmented-control")) {
      ctrl.querySelectorAll(".seg-item").forEach((btn) => {
        btn.classList.toggle("active", parseInt(btn.dataset.val, 10) === val);
      });
    }
  }

  function applyFeaturesToForm(features) {
    state.currentFeatures = { ...state.currentFeatures, ...features };

    Object.entries(features).forEach(([key, val]) => {
      const input = $(`[data-field="${key}"]`);
      if (!input) return;

      if (input.classList.contains("segmented-control")) {
        input.querySelectorAll(".seg-item").forEach((btn) => {
          btn.classList.toggle("active", parseInt(btn.dataset.val, 10) === parseInt(val, 10));
        });
      } else if (input.tagName === "SELECT" || input.tagName === "INPUT") {
        input.value = val;
        const displayId = input.dataset.display;
        if (displayId) {
          const disp = $(`#${displayId}`);
          if (disp) disp.textContent = parseFloat(val).toFixed(1);
        }
      }
    });
  }

  async function runSinglePrediction() {
    const th = parseFloat($("#predThresholdSlider")?.value || "0.50");
    try {
      const res = await apiPostJson("/api/predict", {
        features: state.currentFeatures,
        threshold: th,
      });

      state.lastProbability = res.probability;
      renderPredictionResult(res);
      return res.probability;
    } catch (err) {
      console.error("Lỗi suy luận mô hình:", err);
      showToast("Lỗi dự báo: " + err.message, "danger");
      return null;
    }
  }

  function renderPredictionResult(res) {
    const pct = (res.probability * 100).toFixed(1);
    const gaugeText = $("#gaugePercentText");
    if (gaugeText) gaugeText.textContent = `${pct}%`;

    // Animate SVG Tachometer Arc
    const arc = $("#gaugeProgressArc");
    if (arc) {
      const totalDash = 301.59;
      const offset = totalDash - totalDash * res.probability;
      arc.style.strokeDashoffset = offset;
    }

    // Tier badge
    const tierBadge = $("#predTierBadge");
    if (tierBadge) {
      tierBadge.className = `tier-indicator-pill tier-${res.tier_code}`;
      tierBadge.textContent = res.risk_tier || (res.tier_code === "high" ? "Cao (>= 70%)" : res.tier_code === "medium" ? "Trung bình (40-70%)" : "Thấp (< 40%)");
    }

    // Decision Class
    const classBadge = $("#predClassBadge");
    if (classBadge) {
      const isDropout = res.predicted_label === 1;
      classBadge.className = `decision-badge ${isDropout ? "pill-high" : "pill-low"}`;
      classBadge.textContent = isDropout ? "Cảnh báo: Bỏ học (1)" : "An toàn: Tốt nghiệp (0)";
    }

    // Advisor Recommendation Box
    const recBox = $("#recommendationBox");
    const recText = $("#recommendationText");
    if (recBox && recText) {
      recBox.className = `recommendation-callout rec-${res.tier_code}`;
      recText.textContent = res.recommendation;
    }

    // Risk Drivers vs Protective Factors
    const explanation = res.explanation || {};
    const riskDrivers = explanation.risk_drivers || [];
    const protectiveFactors = explanation.protective_factors || [];

    const riskList = $("#riskDriversList");
    const safeList = $("#protectiveList");
    if (riskList) {
      riskList.innerHTML = riskDrivers.length
        ? riskDrivers.map((d) => `<li><strong>${escapeHtml(d.title)}:</strong> ${escapeHtml(d.detail)}</li>`).join("")
        : "<li>Không có yếu tố nguy cơ đáng kể.</li>";
    }
    if (safeList) {
      safeList.innerHTML = protectiveFactors.length
        ? protectiveFactors.map((p) => `<li><strong>${escapeHtml(p.title)}:</strong> ${escapeHtml(p.detail)}</li>`).join("")
        : "<li>Cần tiếp tục nỗ lực cải thiện học vụ.</li>";
    }

    // 7 Domain-Engineered Features
    const engGrid = $("#engineeredFeaturesGrid");
    if (engGrid && res.engineered_features) {
      if (Array.isArray(res.engineered_features)) {
        engGrid.innerHTML = res.engineered_features
          .map(
            (feat) => `
            <div class="engineered-chip">
              <span class="eng-label">${escapeHtml(feat.label)}</span>
              <span class="eng-val">${typeof feat.value === "number" ? feat.value.toFixed(2) : feat.value}</span>
            </div>
          `
          )
          .join("");
      } else {
        engGrid.innerHTML = Object.entries(res.engineered_features)
          .map(
            ([k, v]) => `
            <div class="engineered-chip">
              <span class="eng-label">${escapeHtml(k)}</span>
              <span class="eng-val">${typeof v === "number" ? v.toFixed(2) : v}</span>
            </div>
          `
          )
          .join("");
      }
    }
  }

  // =========================================================================
  // View 3: Bulk CSV Screening & Batch Analytics
  // =========================================================================
  function initBatchScreening() {
    const dropzone = $("#csvDropzone");
    const fileInput = $("#csvFileInput");
    const demoBtn = $("#runDemoBatchBtn");
    const thSelect = $("#batchThresholdSelect");
    const exportBtn = $("#exportBatchCsvBtn");
    const tierFilter = $("#batchTierFilter");

    if (dropzone && fileInput) {
      dropzone.addEventListener("click", () => fileInput.click());
      dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("drag-active");
      });
      dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-active"));
      dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("drag-active");
        if (e.dataTransfer.files.length) {
          uploadAndProcessBatchCsv(e.dataTransfer.files[0]);
        }
      });
      fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) {
          uploadAndProcessBatchCsv(e.target.files[0]);
        }
      });
    }

    if (demoBtn) {
      demoBtn.addEventListener("click", async () => {
        try {
          showBatchStatus("Đang tải tệp 25 sinh viên mẫu từ tập kiểm thử...", "info");
          const resp = await fetch("/api/batch-sample?limit=25");
          const blob = await resp.blob();
          const file = new File([blob], "mau_ho_so_sinh_vien.csv", { type: "text/csv" });
          uploadAndProcessBatchCsv(file);
        } catch (err) {
          showBatchStatus("Lỗi tải mẫu CSV: " + err.message, "danger");
        }
      });
    }

    if (thSelect) {
      thSelect.addEventListener("change", () => {
        if (state.lastBatchFile) {
          uploadAndProcessBatchCsv(state.lastBatchFile);
        }
      });
    }

    if (tierFilter) {
      tierFilter.addEventListener("change", () => {
        renderBatchTableRows();
      });
    }

    if (exportBtn) {
      exportBtn.addEventListener("click", () => {
        exportBatchResultsCsv();
      });
    }
  }

  function showBatchStatus(msg, type = "info") {
    const elem = $("#batchStatusMsg");
    if (!elem) return;
    elem.textContent = msg;
    elem.className = `batch-notification-msg text-${type === "danger" ? "crimson" : type === "success" ? "emerald" : "primary"}`;
  }

  async function uploadAndProcessBatchCsv(file) {
    state.lastBatchFile = file;
    const th = parseFloat($("#batchThresholdSelect")?.value || "0.50");
    showBatchStatus(`Đang xử lý ${escapeHtml(file.name)}...`, "info");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const resp = await fetch(`/api/predict-batch?threshold=${th}`, {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Lỗi HTTP ${resp.status}`);
      }

      const res = await resp.json();
      state.batchData = res;
      showBatchStatus(`Đã hoàn tất sàng lọc ${res.total} hồ sơ thành công!`, "success");
      renderBatchSummary(res);
      renderBatchTableRows();
      $("#batchResultsContainer")?.classList.remove("hidden");
    } catch (err) {
      console.error("Lỗi tải file CSV:", err);
      showBatchStatus("Lỗi xử lý file: " + err.message, "danger");
    }
  }

  function renderBatchSummary(data) {
    const total = data.total || 0;
    const tc = data.tier_counts || { high: 0, medium: 0, low: 0 };
    const avgProb = data.avg_probability_pct || 0;

    if ($("#batchTotalCount")) $("#batchTotalCount").textContent = total.toLocaleString("vi-VN");
    if ($("#batchHighCount")) $("#batchHighCount").textContent = (tc.high || 0).toLocaleString("vi-VN");
    if ($("#batchMedCount")) $("#batchMedCount").textContent = (tc.medium || 0).toLocaleString("vi-VN");
    if ($("#batchLowCount")) $("#batchLowCount").textContent = (tc.low || 0).toLocaleString("vi-VN");
    if ($("#batchAvgProb")) $("#batchAvgProb").textContent = `${avgProb}%`;
  }

  function renderBatchTableRows() {
    if (!state.batchData || !state.batchData.rows) return;
    const filter = $("#batchTierFilter")?.value || "all";
    const tbody = $("#batchTableBody");
    if (!tbody) return;

    let items = state.batchData.rows;
    if (filter !== "all") {
      items = items.filter((r) => r.tier_code === filter);
    }

    if (!items.length) {
      tbody.innerHTML = '<tr><td colspan="10" style="text-align:center; padding: 2rem;">Không tìm thấy hồ sơ phù hợp với bộ lọc.</td></tr>';
      return;
    }

    tbody.innerHTML = items
      .map(
        (r) => `
        <tr>
          <td><strong>#${r.row_no}</strong></td>
          <td>${escapeHtml(r.course)}</td>
          <td>${r.age}</td>
          <td>${r.approved_1st}</td>
          <td>${r.approved_2nd}</td>
          <td class="numeric-val">${r.avg_grade !== undefined ? r.avg_grade.toFixed(1) : "—"}</td>
          <td>${r.tuition_ok ? '<span class="text-emerald">Đã đóng</span>' : '<span class="text-crimson">Chưa đóng</span>'}</td>
          <td class="numeric-val"><strong>${(r.probability * 100).toFixed(1)}%</strong></td>
          <td><span class="table-risk-pill pill-${r.tier_code}">${escapeHtml(r.risk_tier)}</span></td>
          <td style="font-size: 0.78rem;">${escapeHtml(r.recommendation)}</td>
        </tr>
      `
      )
      .join("");
  }

  function exportBatchResultsCsv() {
    if (!state.batchData || !state.batchData.rows) return;
    const rows = state.batchData.rows;
    const headers = [
      "Row_No",
      "Course",
      "Age",
      "Approved_Sem1",
      "Approved_Sem2",
      "Avg_Grade",
      "Tuition_Paid",
      "Dropout_Prob",
      "Risk_Tier",
      "Recommendation",
    ];

    const csvContent = [
      headers.join(","),
      ...rows.map((r) =>
        [
          r.row_no,
          `"${(r.course || "").replace(/"/g, '""')}"`,
          r.age,
          r.approved_1st,
          r.approved_2nd,
          r.avg_grade,
          r.tuition_ok,
          r.probability,
          `"${r.risk_tier}"`,
          `"${(r.recommendation || "").replace(/"/g, '""')}"`,
        ].join(",")
      ),
    ].join("\n");

    const blob = new Blob(["\uFEFF" + csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `EduGuard_KetQua_SangLoc_${Date.now()}.csv`;
    link.click();
    showToast("Đã tải xuống file kết quả sàng lọc CSV!", "success");
  }

  // =========================================================================
  // View 4: Experimental Benchmarks & Scientific Figures
  // =========================================================================
  async function loadExperimentsData() {
    try {
      const data = await apiGet("/api/experiments");
      state.experiments = data;
      renderFlowCompareChart(data.results, state.expMetric);
      renderTuningChart(data.tuning, state.tuningAlgo);
      renderResultsTable(data.results);
      renderScientificFigures(data.figures);
      bindExperimentEvents();
    } catch (err) {
      console.error("Lỗi nạp thực nghiệm:", err);
      showToast("Không thể tải kết quả thực nghiệm: " + err.message, "danger");
    }
  }

  function bindExperimentEvents() {
    const metricSel = $("#expMetricSelect");
    if (metricSel) {
      metricSel.addEventListener("change", (e) => {
        state.expMetric = e.target.value;
        if (state.experiments) {
          renderFlowCompareChart(state.experiments.results, state.expMetric);
        }
      });
    }

    const algoToggles = $$("#tuningAlgoToggle .seg-btn");
    algoToggles.forEach((btn) => {
      btn.addEventListener("click", () => {
        algoToggles.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.tuningAlgo = btn.dataset.algo;
        if (state.experiments) {
          renderTuningChart(state.experiments.tuning, state.tuningAlgo);
        }
      });
    });

    const galleryFilters = $$("#galleryFilter .seg-btn");
    galleryFilters.forEach((btn) => {
      btn.addEventListener("click", () => {
        galleryFilters.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.galleryCategory = btn.dataset.cat;
        if (state.experiments) {
          renderScientificFigures(state.experiments.figures);
        }
      });
    });
  }

  function renderFlowCompareChart(compData, metricName) {
    const ctx = $("#chartFlowCompare");
    if (!ctx || !compData) return;
    const colors = getThemeColors();

    if (state.charts.flowCompare) state.charts.flowCompare.destroy();

    const algos = ["K-NN", "Naive Bayes", "Decision Tree", "Random Forest"];
    const flowAValues = [];
    const flowBValues = [];

    algos.forEach((algo) => {
      const rowA = compData.find((r) => r.Flow.includes("Luồng A") && r.Model.includes(algo));
      const rowB = compData.find((r) => r.Flow.includes("Luồng B") && r.Model.includes(algo));
      flowAValues.push(rowA ? (rowA[metricName] * 100).toFixed(2) : 0);
      flowBValues.push(rowB ? (rowB[metricName] * 100).toFixed(2) : 0);
    });

    state.charts.flowCompare = new Chart(ctx, {
      type: "bar",
      data: {
        labels: algos,
        datasets: [
          {
            label: "Luồng A (36 đặc trưng gốc)",
            data: flowAValues,
            backgroundColor: colors.slate,
            borderRadius: 6,
            maxBarThickness: 28,
          },
          {
            label: "Luồng B (43 đặc trưng cải tiến)",
            data: flowBValues,
            backgroundColor: colors.blue,
            borderRadius: 6,
            maxBarThickness: 28,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 50,
            max: 100,
            ticks: {
              callback: (val) => `${val}%`,
              color: colors.text,
              font: { family: "JetBrains Mono", size: 11 },
            },
            grid: { color: colors.grid },
          },
          x: {
            ticks: {
              color: colors.text,
              font: { family: "Plus Jakarta Sans", size: 12, weight: 600 },
            },
            grid: { display: false },
          },
        },
        plugins: {
          legend: {
            position: "top",
            labels: {
              boxWidth: 12,
              font: { family: "Plus Jakarta Sans", size: 12, weight: 600 },
              color: colors.text,
            },
          },
          tooltip: {
            backgroundColor: colors.tooltipBg,
            titleColor: colors.tooltipText,
            bodyColor: colors.tooltipText,
            borderColor: colors.tooltipBorder,
            borderWidth: 1,
            callbacks: {
              label: (item) => ` ${item.dataset.label}: ${item.raw}%`,
            },
          },
        },
      },
    });
  }

  function renderTuningChart(tuningData, algoName) {
    const ctx = $("#chartTuning");
    if (!ctx || !tuningData) return;
    const colors = getThemeColors();

    if (state.charts.tuning) state.charts.tuning.destroy();

    const filtered = tuningData.filter((r) => r.Model.includes(algoName));
    const labels = filtered.map((r) => r.Parameter_Value);
    const f1Scores = filtered.map((r) => (r["F1-score"] * 100).toFixed(2));
    const aucScores = filtered.map((r) => (r["ROC-AUC"] * 100).toFixed(2));

    state.charts.tuning = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "F1-Score (%)",
            data: f1Scores,
            borderColor: colors.blue,
            backgroundColor: "rgba(37, 99, 235, 0.1)",
            tension: 0.3,
            fill: true,
            pointRadius: 4,
          },
          {
            label: "ROC-AUC (%)",
            data: aucScores,
            borderColor: colors.emerald,
            backgroundColor: "transparent",
            tension: 0.3,
            borderDash: [5, 5],
            pointRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            ticks: {
              callback: (val) => `${val}%`,
              color: colors.text,
              font: { family: "JetBrains Mono", size: 11 },
            },
            grid: { color: colors.grid },
          },
          x: {
            title: {
              display: true,
              text: algoName === "Decision Tree" ? "Độ sâu tối đa (max_depth)" : "Số láng giềng k (n_neighbors)",
              color: colors.textMuted,
              font: { size: 11, weight: 600 },
            },
            ticks: {
              color: colors.text,
              font: { family: "JetBrains Mono", size: 11 },
            },
            grid: { display: false },
          },
        },
        plugins: {
          legend: {
            position: "top",
            labels: {
              boxWidth: 12,
              font: { family: "Plus Jakarta Sans", size: 12, weight: 600 },
              color: colors.text,
            },
          },
        },
      },
    });
  }

  function renderResultsTable(compData) {
    const tbody = $("#resultsTableBody");
    if (!tbody || !compData) return;

    tbody.innerHTML = compData
      .map((row) => {
        const isBest = row.Flow.includes("Luồng B") && row.Model.includes("Random Forest");
        const highlight = isBest ? 'style="background-color: var(--brand-light); font-weight: 700;"' : "";
        return `
          <tr ${highlight}>
            <td>${escapeHtml(row.Flow)}</td>
            <td><strong>${escapeHtml(row.Model)}</strong> ${isBest ? "★" : ""}</td>
            <td class="numeric-val">${(row.Accuracy * 100).toFixed(2)}%</td>
            <td class="numeric-val">${(row.Precision * 100).toFixed(2)}%</td>
            <td class="numeric-val">${(row.Recall * 100).toFixed(2)}%</td>
            <td class="numeric-val text-primary"><strong>${row["F1-score"].toFixed(4)}</strong></td>
            <td class="numeric-val">${row["ROC-AUC"].toFixed(4)}</td>
            <td class="numeric-val">${row["PR-AUC"].toFixed(4)}</td>
          </tr>
        `;
      })
      .join("");
  }

  function renderScientificFigures(figures) {
    const grid = $("#figuresGallery");
    if (!grid || !figures) return;

    const cat = state.galleryCategory;
    const filtered = cat === "ALL" ? figures : figures.filter((f) => f.category === cat);

    grid.innerHTML = filtered
      .map(
        (fig) => {
          const imgUrl = `/figures/${fig.filename}`;
          return `
            <div class="figure-gallery-card" data-img="${imgUrl}" data-title="${escapeHtml(fig.title)}" data-desc="${escapeHtml(fig.description)}">
              <div class="figure-img-frame">
                <img src="${imgUrl}" alt="${escapeHtml(fig.title)}" loading="lazy" />
              </div>
              <div class="figure-card-meta">
                <span class="figure-card-cat">${escapeHtml(fig.category)}</span>
                <span class="figure-card-title">${escapeHtml(fig.title)}</span>
              </div>
            </div>
          `;
        }
      )
      .join("");

    grid.querySelectorAll(".figure-gallery-card").forEach((card) => {
      card.addEventListener("click", () => {
        openImageLightbox(card.dataset.img, card.dataset.title, card.dataset.desc);
      });
    });
  }

  function openImageLightbox(url, title, desc) {
    const modal = $("#imageModal");
    const img = $("#lightboxImg");
    const titleEl = $("#lightboxTitle");
    const descEl = $("#lightboxDesc");
    if (!modal || !img) return;

    img.src = url;
    if (titleEl) titleEl.textContent = title;
    if (descEl) descEl.textContent = desc;
    modal.classList.remove("hidden");
  }

  // =========================================================================
  // View 5: Error Analysis & Test Cohort Explorer
  // =========================================================================
  async function fetchErrorSamples() {
    const q = state.errorQuery;
    const url = `/api/test-samples?page=${q.page}&page_size=${q.pageSize}&status=${q.status}&tier=${encodeURIComponent(q.tier)}&search=${encodeURIComponent(q.search)}&sort=${q.sort}`;

    try {
      const data = await apiGet(url);
      renderStatusFilterCards(data.status_counts, q.status);
      renderErrorTableRows(data.items);
      renderPagination(data);
    } catch (err) {
      console.error("Lỗi nạp error samples:", err);
      showToast("Lỗi tải danh sách kiểm thử: " + err.message, "danger");
    }
  }

  function renderStatusFilterCards(counts, activeStatus) {
    const grid = $("#errorStatusCards");
    if (!grid || !counts) return;

    const cards = [
      { key: "ALL", label: "Toàn bộ Tập Test", count: counts.ALL || 885, sub: "N = 885 hồ sơ" },
      { key: "TP", label: "True Positive", count: counts.TP || 0, sub: "Bắt đúng bỏ học" },
      { key: "TN", label: "True Negative", count: counts.TN || 0, sub: "Đoán đúng tốt nghiệp" },
      { key: "FN", label: "False Negative", count: counts.FN || 0, sub: "Bỏ sót rủi ro" },
      { key: "FP", label: "False Positive", count: counts.FP || 0, sub: "Cảnh báo nhầm" },
    ];

    grid.innerHTML = cards
      .map((c) => {
        const isActive = c.key === activeStatus;
        return `
          <button type="button" class="status-card-filter-btn ${isActive ? "active" : ""}" data-status="${c.key}">
            <span class="status-card-label">${escapeHtml(c.label)}</span>
            <span class="status-card-count">${c.count}</span>
            <span class="status-card-sub">${escapeHtml(c.sub)}</span>
          </button>
        `;
      })
      .join("");

    grid.querySelectorAll(".status-card-filter-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.errorQuery.status = btn.dataset.status;
        state.errorQuery.page = 1;
        fetchErrorSamples();
      });
    });
  }

  function renderErrorTableRows(items) {
    const tbody = $("#errorTableBody");
    if (!tbody) return;

    if (!items || !items.length) {
      tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; padding: 2rem;">Không tìm thấy hồ sơ kiểm thử thỏa mãn điều kiện lọc.</td></tr>';
      return;
    }

    tbody.innerHTML = items
      .map((item) => {
        const isActualDrop = item.actual_dropout === 1;
        const isPredDrop = item.predicted_dropout === 1;
        const isError = isActualDrop !== isPredDrop;
        const statusClass = item.prediction_status === "TP" ? "pill-high" : item.prediction_status === "TN" ? "pill-low" : "pill-med";

        return `
          <tr>
            <td><strong>#${item.student_index}</strong></td>
            <td>${escapeHtml(item.course_name)}</td>
            <td>${item.approved_1st} / ${item.approved_2nd}</td>
            <td class="numeric-val">${item.grade_2nd !== undefined ? item.grade_2nd.toFixed(1) : "0.0"}</td>
            <td>
              ${item.tuition_ok ? '<span class="text-emerald">Đủ</span>' : '<span class="text-crimson">Chưa</span>'} / 
              ${item.debtor ? '<span class="text-crimson">Nợ</span>' : '<span class="text-emerald">Không</span>'}
            </td>
            <td><span class="table-risk-pill ${isActualDrop ? "pill-high" : "pill-low"}">${isActualDrop ? "Bỏ học" : "Tốt nghiệp"}</span></td>
            <td class="numeric-val"><strong>${(item.probability * 100).toFixed(1)}%</strong></td>
            <td><span class="table-risk-pill ${statusClass}">${item.prediction_status} (${isError ? "Sai" : "Đúng"})</span></td>
            <td>
              <button class="btn btn-outline btn-sm btn-view-student" data-id="${item.student_index}">
                Chi tiết
              </button>
            </td>
          </tr>
        `;
      })
      .join("");

    tbody.querySelectorAll(".btn-view-student").forEach((btn) => {
      btn.addEventListener("click", () => {
        openStudentDetailModal(btn.dataset.id);
      });
    });
  }

  function renderPagination(data) {
    const pageInfo = $("#errPageInfo");
    if (pageInfo) {
      const start = (data.page - 1) * data.page_size + 1;
      const end = Math.min(data.page * data.page_size, data.total);
      pageInfo.textContent = `Hiển thị ${start} – ${end} / ${data.total} hồ sơ (Trang ${data.page}/${data.total_pages})`;
    }

    const prevBtn = $("#errPrevBtn");
    const nextBtn = $("#errNextBtn");
    if (prevBtn) {
      prevBtn.disabled = data.page <= 1;
      prevBtn.onclick = () => {
        if (state.errorQuery.page > 1) {
          state.errorQuery.page--;
          fetchErrorSamples();
        }
      };
    }
    if (nextBtn) {
      nextBtn.disabled = data.page >= data.total_pages;
      nextBtn.onclick = () => {
        if (state.errorQuery.page < data.total_pages) {
          state.errorQuery.page++;
          fetchErrorSamples();
        }
      };
    }
  }

  function initErrorFilterEvents() {
    const searchInput = $("#errSearchInput");
    if (searchInput) {
      searchInput.addEventListener(
        "input",
        debounce((e) => {
          state.errorQuery.search = e.target.value.trim();
          state.errorQuery.page = 1;
          fetchErrorSamples();
        }, 300)
      );
    }

    const tierFilter = $("#errTierFilter");
    if (tierFilter) {
      tierFilter.addEventListener("change", (e) => {
        state.errorQuery.tier = e.target.value;
        state.errorQuery.page = 1;
        fetchErrorSamples();
      });
    }

    const sortSelect = $("#errSortSelect");
    if (sortSelect) {
      sortSelect.addEventListener("change", (e) => {
        state.errorQuery.sort = e.target.value;
        state.errorQuery.page = 1;
        fetchErrorSamples();
      });
    }
  }

  // =========================================================================
  // Modals & Details
  // =========================================================================
  async function openStudentDetailModal(studentIndex) {
    try {
      const detail = await apiGet(`/api/test-samples/${studentIndex}`);
      state.currentModalStudent = detail;

      const title = $("#modalStudentTitle");
      const sub = $("#modalStudentSub");
      const body = $("#modalStudentBody");
      const modal = $("#studentModal");

      if (title) title.textContent = `Hồ sơ Sinh viên #${detail.student_index}`;
      if (sub) {
        sub.textContent = `${detail.course_name} • ${detail.actual_text}`;
      }

      const pred = detail.prediction || {};
      const exp = pred.explanation || {};
      const isDrop = detail.actual_dropout === 1;

      if (body) {
        body.innerHTML = `
          <div class="actual-truth-banner">
            <div>
              <strong>Kết quả Đánh giá:</strong> Xác suất bỏ học <strong>${((pred.probability || 0) * 100).toFixed(1)}%</strong> 
              (${pred.risk_tier || ""}) • Trạng thái: <strong>${detail.prediction_status}</strong>
            </div>
            <span class="table-risk-pill ${isDrop ? "pill-high" : "pill-low"}">${isDrop ? "Thực tế: Bỏ học" : "Thực tế: An toàn"}</span>
          </div>

          <div class="recommendation-callout rec-${pred.tier_code || "low"}">
            <div class="rec-header">Khuyến nghị cho Cố vấn học tập</div>
            <p class="rec-body">${escapeHtml(pred.recommendation || "")}</p>
          </div>

          <div class="drivers-duo-container">
            <div class="driver-column">
              <h4 class="driver-group-title text-crimson">Yếu tố tăng nguy cơ</h4>
              <ul class="drivers-items-list list-danger">
                ${(exp.risk_drivers || []).map((d) => `<li><strong>${escapeHtml(d.title)}:</strong> ${escapeHtml(d.detail)}</li>`).join("") || "<li>Không có yếu tố nguy cơ lớn.</li>"}
              </ul>
            </div>
            <div class="driver-column">
              <h4 class="driver-group-title text-emerald">Yếu tố bảo vệ</h4>
              <ul class="drivers-items-list list-safe">
                ${(exp.protective_factors || []).map((p) => `<li><strong>${escapeHtml(p.title)}:</strong> ${escapeHtml(p.detail)}</li>`).join("") || "<li>Chưa có yếu tố bảo vệ nổi bật.</li>"}
              </ul>
            </div>
          </div>

          <h4 class="engineered-section-title">7 Đặc trưng Miền Giáo dục (Luồng B)</h4>
          <div class="engineered-metrics-grid">
            ${(pred.engineered_features || [])
              .map(
                (feat) => `
                <div class="engineered-chip">
                  <span class="eng-label">${escapeHtml(feat.label)}</span>
                  <span class="eng-val">${typeof feat.value === "number" ? feat.value.toFixed(2) : feat.value}</span>
                </div>
              `
              )
              .join("")}
          </div>
        `;
      }

      modal?.classList.remove("hidden");
    } catch (err) {
      showToast("Không thể tải chi tiết sinh viên: " + err.message, "danger");
    }
  }

  function populateTeamModal(projectMeta) {
    const list = $("#teamMembersList");
    if (!list || !projectMeta || !projectMeta.members) return;

    list.innerHTML = projectMeta.members
      .map(
        (m) => `
        <div class="member-item-row">
          <div>
            <div class="member-name">${escapeHtml(m.name)}</div>
            <div class="member-role">${escapeHtml(m.role)}</div>
          </div>
          <div class="member-id">${escapeHtml(m.id)} (${escapeHtml(m.class)})</div>
        </div>
      `
      )
      .join("");
  }

  function initModalDismissHandlers() {
    // Dismiss on close button or backdrop
    $$("[data-close-modal]").forEach((elem) => {
      elem.addEventListener("click", () => {
        const modalId = elem.dataset.closeModal;
        $(`#${modalId}`)?.classList.add("hidden");
      });
    });

    // Dismiss on Esc key
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        $$(".modal-backdrop-layer").forEach((m) => m.classList.add("hidden"));
        closeMobileSidebar();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "j") {
        e.preventDefault();
        toggleTheme();
      }
    });

    // Modal action: Load into What-If Simulator
    $("#modalLoadWhatIfBtn")?.addEventListener("click", () => {
      if (state.currentModalStudent && state.currentModalStudent.raw_features) {
        applyFeaturesToForm(state.currentModalStudent.raw_features);
        $("#studentModal")?.classList.add("hidden");
        switchView("predictor");
        runSinglePrediction();
        showToast(`Đã đưa hồ sơ #${state.currentModalStudent.student_index} vào Mô phỏng What-If!`, "success");
      }
    });

    // Open Team Modal Button
    $("#openTeamBtn")?.addEventListener("click", () => {
      $("#teamModal")?.classList.remove("hidden");
    });
  }

  // =========================================================================
  // Application Bootstrap
  // =========================================================================
  async function initApp() {
    // Setup Theme
    applyTheme(state.theme);
    $("#themeToggleBtn")?.addEventListener("click", toggleTheme);

    // Setup Navigation Tabs
    $$(".nav-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        switchView(tab.dataset.view);
      });
    });

    // Jump buttons from hero banner
    $$("[data-jump]").forEach((btn) => {
      btn.addEventListener("click", () => {
        switchView(btn.dataset.jump);
      });
    });

    // Mobile menu toggle
    $("#mobileMenuBtn")?.addEventListener("click", openMobileSidebar);
    $("#sidebarBackdrop")?.addEventListener("click", closeMobileSidebar);

    // Confusion Matrix threshold toggles in Dashboard
    $$("#cmThresholdToggle .seg-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        $$("#cmThresholdToggle .seg-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.cmThreshold = btn.dataset.th;
        updateConfusionMatrixView(state.cmThreshold);
      });
    });

    // Initialize modules
    initModalDismissHandlers();
    initBatchScreening();
    initErrorFilterEvents();

    // Load initial data
    await loadOverviewData();
    await loadPredictorMetadata();
  }

  // Run on DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initApp);
  } else {
    initApp();
  }
})();
