/* 选择模式框（同越狱攻击工具包）：预设回放 / 实时模型。
   当前仅做样式与状态指示：切换只更新右侧说明文字，页面功能暂不随模式切换；
   「实时模型」按钮的小字轮询 /api/real/status，显示模型服务状态。 */
"use strict";

const $mode = (id) => document.getElementById(id);

const modeState = {
  mode: "demo",   // 默认进入预设回放（同越狱攻击工具包）
  liveReady: false,
  model: "",      // 模型名（仅末级目录/仓库名，避免长路径撑破按钮）
};

/* 右侧说明文字：与越狱攻击工具包保持同一套文案语义 */
function renderModeDescription() {
  const node = $mode("mode-description");
  if (modeState.mode === "demo") {
    node.textContent = "浏览已保存的教学记录；切换到实时模型后，才会提交新的实验。";
  } else if (modeState.liveReady) {
    node.textContent = `实时连接到${modeState.model}；提交后才会生成新的回答。`;
  } else {
    node.textContent = "实时模型服务当前未启动；可先继续浏览离线记录。";
  }
}

function setMode(next) {
  if (!["demo", "live"].includes(next)) return;
  modeState.mode = next;
  $mode("mode-demo").classList.toggle("active", next === "demo");
  $mode("mode-live").classList.toggle("active", next === "live");
  $mode("mode-demo").setAttribute("aria-pressed", String(next === "demo"));
  $mode("mode-live").setAttribute("aria-pressed", String(next === "live"));
  renderModeDescription();
}

/* 「实时模型」按钮小字：服务可达且已加载 → 模型名；已配置未加载 → 模型待加载；
   可达但未配置模型 → 未配置模型；服务不可达 → 模型服务离线。 */
function renderLiveBadge(reachable, status) {
  let text;
  if (!reachable) text = "模型服务离线";
  else if (status.loaded) text = shortModelName(status.model) || "模型已就绪";
  else text = status.model ? "模型待加载" : "未配置模型";
  $mode("live-mode-status").textContent = text;
}

function shortModelName(full) {
  const parts = String(full || "").split("/").filter(Boolean);
  return parts[parts.length - 1] || "";
}

async function checkLiveStatus() {
  let reachable = false;
  let status = {};
  try {
    const response = await fetch("api/real/status");
    if (response.ok) {
      status = await response.json();
      reachable = true;
    }
  } catch { /* 服务未启动时保持离线展示 */ }
  modeState.liveReady = Boolean(reachable && status.loaded);
  modeState.model = shortModelName(status.model);
  renderLiveBadge(reachable, status);
  if (modeState.mode === "live") renderModeDescription();
}

$mode("mode-demo").addEventListener("click", () => setMode("demo"));
$mode("mode-live").addEventListener("click", () => setMode("live"));
checkLiveStatus();
setInterval(checkLiveStatus, 5000);
