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

let options = null;

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

function formatStatus(status) {
  const labels = {
    pending: "等待",
    running: "运行中",
    success: "成功",
    failed: "失败",
    partial_failed: "部分失败",
    created: "已创建",
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

function renderMatrices(matrices) {
  matrixTable.innerHTML = matrices
    .map(
      (item) => `
        <tr>
          <td>${item.index}</td>
          <td>${item.name}</td>
          <td>${formatNumber(item.order)}</td>
          <td>${formatNumber(item.nnz)}</td>
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
    serverStatus.textContent = "后端已连接";
    serverStatus.className = "status-pill ok";
  } catch (error) {
    serverStatus.textContent = "后端未连接";
    serverStatus.className = "status-pill bad";
    setLog(`无法连接后端：${error.message}`);
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
  } catch (error) {
    runStatus.textContent = "运行失败";
    setLog(`实验运行失败：${error.message}`);
  } finally {
    runButton.disabled = false;
  }
}

routeSelect.addEventListener("change", refreshMetrics);
precisionSelect.addEventListener("change", refreshMetrics);
runButton.addEventListener("click", createRun);

loadOptions();
