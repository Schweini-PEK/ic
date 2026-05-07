const routeSelect = document.getElementById("routeSelect");
const precisionSelect = document.getElementById("precisionSelect");
const fixedOrdering = document.getElementById("fixedOrdering");
const fixedScaling = document.getElementById("fixedScaling");
const routeMetric = document.getElementById("routeMetric");
const precisionMetric = document.getElementById("precisionMetric");
const runStatus = document.getElementById("runStatus");
const serverStatus = document.getElementById("serverStatus");
const runButton = document.getElementById("runButton");
const matrixTable = document.getElementById("matrixTable");
const resultTable = document.getElementById("resultTable");
const logOutput = document.getElementById("logOutput");
const runIdLabel = document.getElementById("runIdLabel");
const compareBaseSelect = document.getElementById("compareBaseSelect");
const compareTargetSelect = document.getElementById("compareTargetSelect");
const compareButton = document.getElementById("compareButton");
const sptrsvChart = document.getElementById("sptrsvChart");
const pcgChart = document.getElementById("pcgChart");
const sptrsvChartLabel = document.getElementById("sptrsvChartLabel");
const pcgChartLabel = document.getElementById("pcgChartLabel");
const navItems = document.querySelectorAll(".nav-item");
const pages = document.querySelectorAll(".page");

let options = null;
let runGroups = [];

function formatNumber(value) {
  return Number(value).toLocaleString("en-US");
}

function formatMetric(value, digits = 4) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return String(value);
  }
  if (Math.abs(number) >= 100) {
    return number.toFixed(2);
  }
  if (Math.abs(number) >= 1) {
    return number.toFixed(4);
  }
  return number.toExponential(digits);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatStatus(status) {
  const labels = {
    pending: "等待",
    running: "运行中",
    success: "成功",
    failed: "失败",
    partial_failed: "部分失败",
    created: "已创建",
    not_run: "未运行",
  };
  return labels[status] ?? status ?? "等待";
}

function setLog(lines) {
  logOutput.textContent = Array.isArray(lines) ? lines.join("\n") : lines;
}

function fillSelect(select, items) {
  select.innerHTML = "";
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label;
    select.appendChild(option);
  }
}

function fillRunGroupSelect(select, runs) {
  const previous = select.value;
  select.innerHTML = "";
  for (const item of runs) {
    const option = document.createElement("option");
    option.value = item.group_id;
    option.textContent = `${item.group_label}（${formatStatus(item.status)}）`;
    select.appendChild(option);
  }
  if (previous && [...select.options].some((option) => option.value === previous)) {
    select.value = previous;
  }
}

function renderMatrices(matrices) {
  matrixTable.innerHTML = matrices
    .map(
      (item) => `
        <tr>
          <td>${item.index}</td>
          <td>${item.name}</td>
          <td>${formatNumber(item.order)}</td>
          <td>${formatNumber(item.nnz)}</td>
          <td>${item.domain ?? "-"}</td>
        </tr>
      `,
    )
    .join("");
}

function renderResults(matrices) {
  resultTable.innerHTML = matrices
    .map(
      (item) => `
        <tr>
          <td>${item.index}</td>
          <td>${item.name}</td>
          <td>${formatStatus(item.status)}</td>
          <td>${item.pcg_iterations ?? "-"}</td>
          <td>${formatMetric(item.sptrsv_avg_time_ms)}</td>
          <td>${formatMetric(item.pcg_time_ms)}</td>
          <td>${formatMetric(item.relative_residual)}</td>
        </tr>
      `,
    )
    .join("");
}

function refreshMetrics() {
  const routeLabel = routeSelect.selectedOptions[0]?.textContent ?? "-";
  const precisionLabel = precisionSelect.selectedOptions[0]?.textContent ?? "-";
  routeMetric.textContent = routeLabel;
  precisionMetric.textContent = precisionLabel;
}

async function loadOptions() {
  try {
    const response = await fetch("/api/options");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    options = await response.json();
    fillSelect(routeSelect, options.routes);
    fillSelect(precisionSelect, options.precisions);
    fixedOrdering.textContent = options.fixed.ordering;
    fixedScaling.textContent = options.fixed.scaling;
    renderMatrices(options.matrices);
    renderResults(options.matrices.map((item) => ({ ...item, status: "pending" })));
    refreshMetrics();
    await loadRunGroups();
    serverStatus.textContent = "后端已连接";
    serverStatus.className = "status-pill ok";
  } catch (error) {
    serverStatus.textContent = "后端未连接";
    serverStatus.className = "status-pill bad";
    setLog(`无法连接后端：${error.message}`);
  }
}

function showPage(pageId) {
  for (const page of pages) {
    page.classList.toggle("active", page.id === pageId);
  }
  for (const item of navItems) {
    item.classList.toggle("active", item.dataset.page === pageId);
  }
}

async function loadRunGroups() {
  const response = await fetch("/api/compare-runs");
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const payload = await response.json();
  runGroups = payload.runs ?? [];
  fillRunGroupSelect(compareBaseSelect, runGroups);
  fillRunGroupSelect(compareTargetSelect, runGroups);
  if (!compareBaseSelect.value) {
    compareBaseSelect.value = "level_double";
  }
  if (!compareTargetSelect.value || compareTargetSelect.value === compareBaseSelect.value) {
    compareTargetSelect.value = "level_float";
  }
}

async function createRun() {
  runButton.disabled = true;
  runStatus.textContent = "运行中";
  setLog([
    "开始运行实验...",
    `路线：${routeSelect.value}`,
    `精度：${precisionSelect.value}`,
    "固定参数：ordering=RCM, scaling=UnitSqrtDiag",
    "系统将依次运行全部 7 个矩阵，过程可能需要几分钟。",
  ]);

  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        route: routeSelect.value,
        precision: precisionSelect.value,
      }),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const summary = await response.json();
    runIdLabel.textContent = summary.run_id;
    runStatus.textContent = formatStatus(summary.status);
    renderResults(summary.matrices);
    setLog(summary.log ?? [`run_id: ${summary.run_id}`, summary.message]);
    await loadRunGroups();
  } catch (error) {
    runStatus.textContent = "运行失败";
    setLog(`实验运行失败：${error.message}`);
  } finally {
    runButton.disabled = false;
  }
}

function renderBarChart(container, items, metricKey, color) {
  const width = 640;
  const height = 300;
  const margin = { top: 24, right: 22, bottom: 46, left: 48 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const values = items.map((item) => Number(item[metricKey])).filter((value) => Number.isFinite(value));
  const maxValue = Math.max(1, ...values) * 1.18;
  const barGap = 16;
  const barWidth = Math.max(18, (plotWidth - barGap * (items.length - 1)) / items.length);

  const bars = items
    .map((item, index) => {
      const value = Number(item[metricKey]);
      const valid = Number.isFinite(value);
      const x = margin.left + index * (barWidth + barGap);
      const h = valid ? (value / maxValue) * plotHeight : 8;
      const y = margin.top + plotHeight - h;
      const label = valid ? `${value.toFixed(2)}x` : "-";
      const fill = valid ? color : "#b8b8b8";
      const tooltip = `编号 ${escapeHtml(item.index)}：${escapeHtml(item.name ?? "-")}`;
      return `
        <g class="chart-item">
          <title>${tooltip}</title>
          <rect x="${x}" y="${y}" width="${barWidth}" height="${h}" rx="3" fill="${fill}"></rect>
          <text x="${x + barWidth / 2}" y="${y - 8}" text-anchor="middle" font-size="13" fill="#1d2630">${label}</text>
          <text x="${x + barWidth / 2}" y="${height - 16}" text-anchor="middle" font-size="13" fill="#1d2630">${item.index}</text>
        </g>
      `;
    })
    .join("");

  const gridLines = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    .filter((tick) => tick <= maxValue)
    .map((tick) => {
      const y = margin.top + plotHeight - (tick / maxValue) * plotHeight;
      return `
        <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="#d8dde6" stroke-dasharray="5 6"></line>
        <text x="${margin.left - 10}" y="${y + 4}" text-anchor="end" font-size="12" fill="#637083">${tick.toFixed(1)}</text>
      `;
    })
    .join("");

  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img">
      ${gridLines}
      <line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${width - margin.right}" y2="${margin.top + plotHeight}" stroke="#1d2630"></line>
      <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}" stroke="#1d2630"></line>
      ${bars}
      <text x="${width / 2}" y="${height - 2}" text-anchor="middle" font-size="13" fill="#637083">矩阵编号</text>
      <text x="16" y="${height / 2}" text-anchor="middle" transform="rotate(-90 16 ${height / 2})" font-size="13" fill="#637083">Speedup</text>
    </svg>
  `;
}

async function compareRuns() {
  const baseGroup = compareBaseSelect.value;
  const targetGroup = compareTargetSelect.value;
  if (!baseGroup || !targetGroup) {
    setLog("请先选择两组实验结果。");
    return;
  }
  if (baseGroup === targetGroup) {
    setLog("对比对象 A 和 B 不能相同。");
    return;
  }

  compareButton.disabled = true;
  try {
    const response = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        base_group: baseGroup,
        target_group: targetGroup,
      }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail ?? `HTTP ${response.status}`);
    }
    const comparison = await response.json();
    const title = `${comparison.target_label} 相对 ${comparison.base_label}`;
    sptrsvChartLabel.textContent = title;
    pcgChartLabel.textContent = title;
    renderBarChart(sptrsvChart, comparison.matrices, "sptrsv_speedup", "#4E78BE");
    renderBarChart(pcgChart, comparison.matrices, "pcg_speedup", "#EA793C");
    setLog([
      `生成对比图表：${title}`,
      "加速比 speedup 的计算公式已在页面上方显示。",
      "单次SpTRSV平均时间/ms 与 PCG总时间/ms 的 speedup 图表已更新。",
    ]);
  } catch (error) {
    setLog(`生成对比图表失败：${error.message}`);
  } finally {
    compareButton.disabled = false;
  }
}

routeSelect.addEventListener("change", refreshMetrics);
precisionSelect.addEventListener("change", refreshMetrics);
runButton.addEventListener("click", createRun);
compareButton.addEventListener("click", compareRuns);
for (const item of navItems) {
  item.addEventListener("click", () => showPage(item.dataset.page));
}

loadOptions();
