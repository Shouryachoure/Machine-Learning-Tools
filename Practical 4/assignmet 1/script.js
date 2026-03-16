let model = null;
let imgEl = null;

function log(msg, type = '') {
  const box = document.getElementById('log');
  const d = document.createElement('div');
  d.className = `log-line ${type}`;
  d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  box.appendChild(d); box.scrollTop = box.scrollHeight;
}

async function loadModel() {
  log('Loading MobileNet v2...', 'inf');
  model = await mobilenet.load({ version: 2, alpha: 1.0 });
  log('Model ready!', 'ok');
}

function loadImage(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    const img = document.getElementById('staticImg');
    img.src = ev.target.result;
    img.style.display = 'block';
    document.getElementById('uploadPrompt').style.display = 'none';
    imgEl = img;
    document.getElementById('classBtn').disabled = false;
    log(`Loaded: ${file.name}`, 'inf');
  };
  reader.readAsDataURL(file);
}

async function classify() {
  if (!imgEl) return;
  if (!model) await loadModel();
  log('Classifying...', 'inf');
  const t0 = performance.now();
  const preds = await model.classify(imgEl, 3);
  const ms = Math.round(performance.now() - t0);
  log(`Done in ${ms}ms`, 'ok');
  document.getElementById('s_time').textContent = ms;
  document.getElementById('topResult').textContent = preds[0].className.split(',')[0];
  renderCards(preds);
}

function renderCards(preds) {
  const area = document.getElementById('predCards');
  area.innerHTML = '';
  preds.forEach((p, i) => {
    const name = p.className.split(',')[0];
    const pct  = (p.probability * 100).toFixed(1);
    const d = document.createElement('div');
    d.className = `pred-card ${i === 0 ? 'first' : ''}`;
    d.innerHTML = `
      <div class="pc-row">
        <span class="pc-name">${i+1}. ${name}</span>
        <span class="pc-pct">${pct}%</span>
      </div>
      <div class="bar-wrap"><div class="bar" style="width:${pct}%"></div></div>`;
    area.appendChild(d);
  });
}

function reset() {
  imgEl = null;
  document.getElementById('staticImg').style.display = 'none';
  document.getElementById('uploadPrompt').style.display = 'flex';
  document.getElementById('classBtn').disabled = true;
  document.getElementById('imgInput').value = '';
  document.getElementById('predCards').innerHTML = '';
  document.getElementById('topResult').textContent = 'Upload & classify an image';
  document.getElementById('s_time').textContent = '--';
}

window.addEventListener('load', loadModel);