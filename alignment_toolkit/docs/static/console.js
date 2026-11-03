'use strict';

const $ = id => document.getElementById(id);
let workflow = 'rlhf';
let gpus = [];
let pollTimer;
let logOffset = 0;

const defaults = {
  sft: {id: 'HuggingFaceH4/ultrachat_200k', split: 'train_sft', type: 'huggingface'},
  preference: {id: 'HuggingFaceH4/ultrafeedback_binarized', split: 'train_prefs', type: 'huggingface'},
  prompts: {id: 'HuggingFaceH4/ultrachat_200k', split: 'train_gen', type: 'huggingface'},
};

function datasetRoles() {
  if (workflow === 'sft') return [['sft', 'SFT / instruction']];
  if (workflow === 'dpo') return [['preference', 'Preference / chosen · rejected']];
  return [['sft', 'SFT / instruction'], ['preference', 'Reward Model / preference'], ['prompts', 'PPO / prompts']];
}

function renderDatasets() {
  $('dataset-fields').innerHTML = datasetRoles().map(([key, label]) => {
    const item = defaults[key];
    return `<div class="dataset-card"><h3>${label}</h3><div class="dataset-fields"><input data-dataset="${key}" data-field="id" value="${item.id}" placeholder="Hugging Face ID 或本地 JSON 路径"><select data-dataset="${key}" data-field="type"><option value="huggingface">Hugging Face</option><option value="json">本地 JSON</option></select><input data-dataset="${key}" data-field="split" value="${item.split}" placeholder="split"><span></span></div></div>`;
  }).join('');
  document.querySelectorAll('[data-dataset][data-field="type"]').forEach(select => {
    select.value = defaults[select.dataset.dataset].type;
    select.addEventListener('change', () => {
      const card = select.closest('.dataset-card');
      const input = card.querySelector('[data-field="id"]');
      input.placeholder = select.value === 'json' ? '/path/to/data.json' : 'Hugging Face dataset ID';
    });
  });
}

function renderPlan() {
  const stages = workflow === 'sft' ? [['sft', 'SFT', 'instruction']] : workflow === 'dpo' ? [['dpo', 'DPO', 'preference']] : [['sft', 'SFT', 'instruction'], ['reward_model', 'Reward Model', 'preference'], ['ppo', 'PPO', 'prompts']];
  $('plan-name').textContent = `console_${workflow}`;
  $('stage-plan').innerHTML = stages.map((stage, i) => `<div class="stage-line"><span class="stage-index">0${i + 1}</span><b>${stage[1]}</b><span>${stage[2]}</span></div>`).join('');
}

function readDatasets() {
  const result = {};
  document.querySelectorAll('[data-dataset][data-field="id"]').forEach(input => {
    const key = input.dataset.dataset;
    const card = input.closest('.dataset-card');
    const type = card.querySelector('[data-field="type"]').value;
    const split = card.querySelector('[data-field="split"]').value;
    result[key] = type === 'json' ? {type, path: input.value, split} : {type, id: input.value, split};
  });
  return result;
}

function payload() {
  return {
    workflow,
    model_source: $('model-source').value,
    model: $('model').value.trim(),
    revision: $('revision').value.trim(),
    gpus: [...document.querySelectorAll('#gpu-list input:checked')].map(input => Number(input.value)),
    output_root: $('output-root').value.trim(),
    epochs: Number($('epochs').value),
    batch_size: Number($('batch-size').value),
    learning_rate: Number($('learning-rate').value),
    max_length: Number($('max-length').value),
    lora: $('lora').checked,
    datasets: readDatasets(),
  };
  if (workflow === 'dpo') {
    data.beta = Number($('beta').value);
    data.max_prompt_length = Number($('max-prompt-length').value);
  }
  if (workflow === 'rlhf') {
    data.ppo_episodes = Number($('ppo-episodes').value);
    data.response_length = Number($('response-length').value);
    data.ppo_epochs = Number($('ppo-epochs').value);
    data.ppo_minibatches = Number($('ppo-minibatches').value);
    data.ppo_kl = Number($('ppo-kl').value);
    data.ppo_gamma = Number($('ppo-gamma').value);
    data.ppo_lam = Number($('ppo-lam').value);
    data.ppo_vf = Number($('ppo-vf').value);
    data.ppo_clip = Number($('ppo-clip').value);
  }
  return data;
}

function showPreview() {
  $('config-preview').textContent = JSON.stringify(payload(), null, 2);
}

function setWorkflow(next) {
  workflow = next;
  document.querySelectorAll('[data-workflow]').forEach(button => button.classList.toggle('active', button.dataset.workflow === workflow));
  document.querySelectorAll('[data-method]').forEach(group => group.classList.toggle('hidden', group.dataset.method !== workflow));
  renderDatasets();
  renderPlan();
  showPreview();
}

async function loadGpus() {
  try {
    const response = await fetch('api/gpus');
    gpus = await response.json();
    $('gpu-list').innerHTML = gpus.length ? gpus.map(gpu => `<label class="gpu-card selected"><input type="checkbox" value="${gpu.index}" checked><span><b>GPU ${gpu.index}</b><small>${gpu.name} · ${gpu.memory_free_mb} MiB free · ${gpu.utilization_gpu}% util</small></span></label>`).join('') : '<div class="empty-state">未检测到 NVIDIA GPU</div>';
    document.querySelectorAll('#gpu-list input').forEach(input => input.addEventListener('change', () => { input.closest('.gpu-card').classList.toggle('selected', input.checked); showPreview(); }));
    showPreview();
  } catch (error) {
    $('gpu-list').innerHTML = '<div class="empty-state">GPU 查询失败，请确认 nvidia-smi 可用</div>';
  }
}

function setStatus(status, detail = '') {
  const label = status.toUpperCase();
  $('run-state').querySelector('strong').textContent = label;
  $('run-state').querySelector('small').textContent = detail || ({idle: '等待一次运行', running: '训练进程正在运行', succeeded: '训练已完成', failed: '训练进程失败', stopped: '训练已停止'}[status] || '');
  $('header-status').textContent = `本地控制台 · ${label}`;
  $('log-kicker').textContent = label;
  $('result-strip').className = `result-strip ${status}`;
  $('result-text').textContent = detail || ({idle: '尚未运行', running: '正在接收后端日志', succeeded: '训练完成，checkpoint 已保存', failed: '训练失败，请查看日志', stopped: '训练已停止'}[status] || '');
  $('start').disabled = status === 'running';
  $('stop').disabled = status !== 'running';
}

async function refreshStatus() {
  const status = await fetch('api/train/status').then(response => response.json());
  setStatus(status.status, status.return_code === null || status.return_code === undefined ? '' : `return code ${status.return_code}`);
  $('job-id').textContent = status.id ? `${status.id} · GPU ${status.gpus?.join(', ') || 'default'}` : '暂无任务';
  const logResponse = await fetch(`api/train/logs?offset=${logOffset}`).then(response => response.json());
  if (logResponse.lines?.length) {
    const output = $('log-output');
    if (logOffset === 0) output.textContent = '';
    output.textContent += `${logResponse.lines.join('\n')}\n`;
    output.scrollTop = output.scrollHeight;
    logOffset = logResponse.next_offset;
  }
}

async function validate() {
  const response = await fetch('api/train/validate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload())});
  const body = await response.json();
  if (!response.ok) throw Error(body.error || '配置校验失败');
  $('result-text').textContent = '配置通过解析和阶段引用校验';
}

async function start() {
  logOffset = 0;
  $('log-output').textContent = '';
  try {
    const response = await fetch('api/train/start', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload())});
    const body = await response.json();
    if (!response.ok) throw Error(body.error || '启动失败');
    setStatus('running');
    $('job-id').textContent = `${body.id} · GPU ${body.gpus?.join(', ') || 'default'}`;
  } catch (error) {
    setStatus('failed', error.message);
  }
}

async function stop() { await fetch('api/train/stop', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'}); await refreshStatus(); }

document.querySelectorAll('[data-workflow]').forEach(button => button.addEventListener('click', () => setWorkflow(button.dataset.workflow)));
document.querySelectorAll('input,select').forEach(element => element.addEventListener('input', showPreview));
document.querySelectorAll('input,select').forEach(element => element.addEventListener('change', showPreview));
$('refresh-gpus').addEventListener('click', loadGpus);
$('validate').addEventListener('click', async () => { try { await validate(); } catch (error) { setStatus('failed', error.message); } });
$('start').addEventListener('click', start);
$('stop').addEventListener('click', stop);
$('clear-log').addEventListener('click', () => { $('log-output').textContent = ''; logOffset = 0; });
$('copy-config').addEventListener('click', () => navigator.clipboard?.writeText($('config-preview').textContent));
setWorkflow('rlhf');
loadGpus();
refreshStatus();
pollTimer = setInterval(refreshStatus, 1200);

// hub 嵌入时（/proxy/alignment/）预填抽屉里选的基座模型，独立使用时保持本地默认。
(async () => {
  try {
    const health = await (await fetch('api/health')).json();
    if (health.hub_model) {
      $('model').value = health.hub_model;
      showPreview();
    }
  } catch (error) { /* 独立运行时无变化 */ }
})();
