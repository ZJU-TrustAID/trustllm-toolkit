/* 运行页：日志轮询、状态刷新、结果渲染 */
"use strict";

const session = qs("session");
const logPane = document.getElementById("log-pane");
let logOffset = 0;
let lastState = "running";
let timer = null;

async function refresh() {
  let status;
  try {
    status = await HUB.get(`/api/runs/${session}`);
  } catch (err) {
    logPane.textContent = "会话加载失败：" + err.message;
    stopPolling();
    return;
  }
  document.title = `运行 ${session} · 第${status.chapter}章 ${status.title || status.toolkit || ""}`;
  document.getElementById("session-id").textContent = `${session} · 第${status.chapter}章 ${status.title || status.toolkit || ""}`;
  document.getElementById("state-chip").innerHTML = stateChip(status.state);
  document.getElementById("elapsed").textContent =
    fmtElapsed(status.started_at, status.ended_at ? status.ended_at : (status.state === "running" ? null : status.started_at));
  document.getElementById("btn-stop").disabled = status.state !== "running";

  if (status.state !== lastState && lastState === "running") {
    if (status.state === "succeeded") toast("执行完成");
    else if (status.state === "failed") toast("执行失败，请查看日志", true);
  }
  lastState = status.state;

  await pullLog(status.state);
  if (status.config) {
    document.getElementById("yaml-view").value = yamlStringify(status.config);
  }
  if (status.state !== "running") {
    stopPolling();
    renderResult(status);
  }
}

async function pullLog(state) {
  try {
    const data = await HUB.get(`/api/runs/${session}/logs?offset=${logOffset}&wait=2`);
    if (data.text) {
      const stick = logPane.scrollTop + logPane.clientHeight >= logPane.scrollHeight - 30;
      logPane.textContent += data.text;
      if (stick) logPane.scrollTop = logPane.scrollHeight;
      logOffset = data.offset;
    }
    document.getElementById("log-hint").textContent =
      state === "running" ? "日志实时拉取中…" : "日志已结束";
  } catch (err) { /* transient */ }
}

function stopPolling() {
  if (timer) { clearInterval(timer); timer = null; }
}

function yamlStringify(config) {
  // 后端返回的 config 是已校验的 dict；这里仅作展示（后端已提供原始 YAML 下载）
  return JSON.stringify(config, null, 2);
}

function renderResult(status) {
  const result = status.result;
  const card = document.getElementById("result-card");
  const body = document.getElementById("result-body");
  const artifacts = document.getElementById("artifacts");
  if (!result) {
    if (status.state === "succeeded") {
      card.style.display = "";
      body.innerHTML = "<p class='muted'>适配器执行成功，但未提供 result.json。</p>";
    }
    return;
  }
  card.style.display = "";
  body.innerHTML = "";
  if (result.summary) {
    body.append(el("p", { style: "font-weight:600;margin:0 0 12px" }, result.summary));
  }
  renderBlocks(body, result.blocks);
  if (status.state === "failed" && result.error) {
    body.append(el("p", { class: "muted", style: "color:var(--bad)" }, result.error));
  }
  artifacts.innerHTML = "";
  const names = new Set(["result.json", ...(result.artifacts || []).map((a) => a.name || a)]);
  for (const name of names) {
    artifacts.append(el("a", { class: "btn small", href: `/api/runs/${session}/artifacts/${encodeURIComponent(name)}` }, "下载 " + name));
  }
}

document.getElementById("btn-stop").addEventListener("click", async () => {
  try {
    await HUB.post(`/api/runs/${session}/stop`);
    toast("已发送停止信号");
  } catch (err) { toast(err.message, true); }
  refresh();
});
document.getElementById("btn-yaml-view").addEventListener("click", () => {
  document.getElementById("yaml-modal").style.display = "flex";
});
document.getElementById("btn-yaml-close").addEventListener("click", () => {
  document.getElementById("yaml-modal").style.display = "none";
});

/* 渲染共用骨架（左侧章节目录 + 顶栏），并加返回按钮 */
Shell.init(null);
document.getElementById("btn-back").addEventListener("click", () => {
  if (document.referrer && document.referrer !== location.href) history.back();
  else location.href = "run.html";
});

refresh();
timer = setInterval(refresh, 1500);
