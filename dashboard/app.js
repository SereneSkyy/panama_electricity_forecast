/* ============================================================
   Panama Energy Forecast dashboard
   Talks to the FastAPI backend defined in main.py:
     GET  /predict_sample/{model_name}   -> random 24h window + prediction
     POST /predict_specific              -> re-predict the SAME window with
                                             a different model (fair comparison)

   The chart is drawn as plain inline SVG -- no charting library, no CDN,
   so nothing here depends on an external script loading successfully.
   ============================================================ */

const API_BASE = "http://127.0.0.1:8000";

// Measured on the held-out test set (src/evaluate.py) after fixing the
// train/evaluate path bug and giving LSTM more training epochs (40 vs
// RNN's 20) -- see the project report for the full comparison.
const MODEL_STATS = {
  LSTM: { MAE: "16.29", RMSE: "24.73", MAPE: "1.46", R2: "0.983", color: "#2563eb" },
  RNN:  { MAE: "18.03", RMSE: "25.49", MAPE: "1.52", R2: "0.982", color: "#ea580c" },
};

const state = {
  selectedModel: "LSTM",
  currentRawFeatures: null, // JSON window returned by the backend, reused when switching models
  lastHistory: null,
};

const el = {
  fetchBtn: document.getElementById("fetchBtn"),
  timestampLabel: document.getElementById("timestampLabel"),
  predictionValue: document.getElementById("predictionValue"),
  metricsModelName: document.getElementById("metricsModelName"),
  legendModelName: document.getElementById("legendModelName"),
  legendPredSwatch: document.getElementById("legendPredSwatch"),
  metricMAE: document.getElementById("metricMAE"),
  metricRMSE: document.getElementById("metricRMSE"),
  metricMAPE: document.getElementById("metricMAPE"),
  metricR2: document.getElementById("metricR2"),
  chartWrap: document.getElementById("chartWrap"),
  statusBar: document.getElementById("statusBar"),
  segButtons: document.querySelectorAll(".segmented__btn"),
};

function setLoading(isLoading) {
  el.fetchBtn.disabled = isLoading;
  el.fetchBtn.textContent = isLoading ? "Loading…" : "Fetch new data window";
}

function showError(msg) {
  el.statusBar.textContent = msg;
  setTimeout(() => { el.statusBar.textContent = ""; }, 5000);
}

function updateMetricsPanel() {
  const stats = MODEL_STATS[state.selectedModel];
  el.metricsModelName.textContent = state.selectedModel;
  el.legendModelName.textContent = state.selectedModel;
  el.legendPredSwatch.style.background = stats.color;
  el.metricMAE.textContent = stats.MAE;
  el.metricRMSE.textContent = stats.RMSE;
  el.metricMAPE.textContent = stats.MAPE;
  el.metricR2.textContent = stats.R2;
}

// Builds a simple line chart as an SVG string: no dependencies, no network
// requests, so it can't silently fail the way a CDN-hosted library can.
function buildChartSVG(history, prediction) {
  const stats = MODEL_STATS[state.selectedModel];
  const points = history.concat([prediction]);

  const W = 700, H = 250;
  const marginLeft = 46, marginRight = 12, marginTop = 12, marginBottom = 26;
  const plotW = W - marginLeft - marginRight;
  const plotH = H - marginTop - marginBottom;

  let minV = Math.min(...points);
  let maxV = Math.max(...points);
  const pad = (maxV - minV || 1) * 0.15;
  minV -= pad;
  maxV += pad;

  const x = (i) => marginLeft + (i / (points.length - 1)) * plotW;
  const y = (v) => marginTop + plotH - ((v - minV) / (maxV - minV)) * plotH;

  const actualPts = history.map((v, i) => `${x(i)},${y(v)}`).join(" ");
  const lastIdx = history.length - 1;
  const predLineX1 = x(lastIdx), predLineY1 = y(history[lastIdx]);
  const predLineX2 = x(lastIdx + 1), predLineY2 = y(prediction);

  const gridLines = [0, 0.33, 0.66, 1].map((f) => {
    const gy = marginTop + f * plotH;
    const value = maxV - f * (maxV - minV);
    return `
      <line x1="${marginLeft}" y1="${gy}" x2="${W - marginRight}" y2="${gy}" stroke="#e3e6eb" stroke-width="1" />
      <text x="${marginLeft - 8}" y="${gy + 4}" text-anchor="end" font-size="10" fill="#9aa1ac" font-family="sans-serif">${value.toFixed(0)}</text>
    `;
  }).join("");

  // X-axis ticks: hour offsets relative to the most recent actual reading
  // (index lastIdx = "0h"), ending at the prediction point ("+1h").
  const xAxisY = marginTop + plotH;
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const idx = Math.round(f * (points.length - 1));
    const hourOffset = idx - lastIdx;
    const label = hourOffset === 0 ? "now" : `${hourOffset > 0 ? "+" : ""}${hourOffset}h`;
    return `<text x="${x(idx)}" y="${xAxisY + 16}" text-anchor="middle" font-size="10" fill="#9aa1ac" font-family="sans-serif">${label}</text>`;
  }).join("");

  return `
    <svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Demand chart">
      ${gridLines}
      <polyline points="${actualPts}" fill="none" stroke="#14181f" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />
      <line x1="${predLineX1}" y1="${predLineY1}" x2="${predLineX2}" y2="${predLineY2}"
            stroke="${stats.color}" stroke-width="3" stroke-dasharray="6,4" stroke-linecap="round" />
      <circle cx="${predLineX2}" cy="${predLineY2}" r="5" fill="${stats.color}" />
      ${xTicks}
    </svg>
  `;
}

function renderChart(history, prediction) {
  state.lastHistory = history;
  el.chartWrap.innerHTML = buildChartSVG(history, prediction);
}

async function fetchNewDataAndPredict() {
  setLoading(true);
  el.predictionValue.textContent = "—";
  try {
    const res = await fetch(`${API_BASE}/predict_sample/${state.selectedModel}`);
    if (!res.ok) throw new Error(`Server responded ${res.status}`);
    const data = await res.json();

    state.currentRawFeatures = data.raw_features;
    el.timestampLabel.textContent = data.timestamp;
    el.predictionValue.textContent = data.prediction.toFixed(1);
    renderChart(data.history, data.prediction);
  } catch (err) {
    showError("Connection error — is the FastAPI backend running on port 8000?");
  } finally {
    setLoading(false);
  }
}

async function rePredictForCurrentData(modelName) {
  if (!state.currentRawFeatures) {
    // Nothing fetched yet -- just switch the active model and let the user fetch.
    state.selectedModel = modelName;
    updateMetricsPanel();
    return;
  }

  setLoading(true);
  state.selectedModel = modelName;
  updateMetricsPanel();

  try {
    const res = await fetch(`${API_BASE}/predict_specific`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ raw_features: state.currentRawFeatures, model_name: modelName }),
    });
    if (!res.ok) throw new Error(`Server responded ${res.status}`);
    const data = await res.json();

    el.predictionValue.textContent = data.prediction.toFixed(1);
    if (state.lastHistory) renderChart(state.lastHistory, data.prediction);
  } catch (err) {
    showError(`Could not switch model: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

el.segButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    el.segButtons.forEach((b) => {
      b.classList.remove("is-active");
      b.setAttribute("aria-selected", "false");
    });
    btn.classList.add("is-active");
    btn.setAttribute("aria-selected", "true");
    rePredictForCurrentData(btn.dataset.model);
  });
});

el.fetchBtn.addEventListener("click", fetchNewDataAndPredict);

// Initial load
updateMetricsPanel();
fetchNewDataAndPredict();
