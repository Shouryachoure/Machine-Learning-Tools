let modelA = null; // MobileNet v2
let modelB = null; // MobileNet v1 (acts as comparison)
let imgEl  = null;

function log(msg, type = '') {
  const box = document.getElementById('log');
  const d = document.createElement('div');
  d.className = `log-line ${type}`;
  d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  box.appendChild(d); box.scrollTop = box.scrollHeight;
}

async function loadModels() {
  log('Loading MobileNet v2...', 'inf');
  modelA = await mobilenet.load({ version: 2, alpha: 1.0 });
  log('MobileNet v2 ready!', 'ok');

  log('Loading MobileNet v1...', 'inf');
  modelB = await mobilenet.load({ version: 1, alpha: 1.0 });
  log('MobileNet v1 ready!', 'ok');
}

function loadImg(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    const img = document.getElementById('cmpImg');
    img.src = ev.target.result;
    img.style.display = 'block';
    document.getElementById('upPrompt').style.display = 'none';
    imgEl = img;
    document.getElementById('cmpBtn').disabled = false;
    log(`Image loaded: ${file.name}`, 'inf');
  };
  reader.readAsDataURL(file);
}

async function compareModels() {
  if (!imgEl) return;
  if (!modelA || !modelB) await loadModels();

  document.getElementById('cmpBtn').disabled = true;
  log('Running MobileNet v2...', 'inf');

  // Model A — MobileNet v2
  const t0 = performance.now();
  const predsA = await modelA.classify(imgEl, 3);
  const msA = Math.round(performance.now() - t0);

  renderResults('resA', predsA);
  document.getElementById('timeA').textContent = msA;
  document.getElementById('confA').textContent = (predsA[0].probability * 100).toFixed(1);
  log(`v2: ${predsA[0].className.split(',')[0]} — ${(predsA[0].probability*100).toFixed(1)}% (${msA}ms)`, 'ok');

  // Model B — MobileNet v1
  log('Running MobileNet v1...', 'inf');
  const t1 = performance.now();
  const predsB = await modelB.classify(imgEl, 3);
  const msB = Math.round(performance.now() - t1);

  renderResults('resB', predsB);
  document.getElementById('timeB').textContent = msB;
  document.getElementById('confB').textContent = (predsB[0].probability * 100).toFixed(1);
  log(`v1: ${predsB[0].className.split(',')[0]} — ${(predsB[0].probability*100).toFixed(1)}% (${msB}ms)`, 'ok');

  showVerdict(predsA, predsB, msA, msB);
  document.getElementById('cmpBtn').disabled = false;
}

function renderResults(containerId, preds) {
  const el = document.getElementById(containerId);
  el.innerHTML = '';
  preds.forEach((p, i) => {
    const name = p.className.split(',')[0];
    const pct  = (p.probability * 100).toFixed(1);
    const row  = document.createElement('div');
    row.className = `res-row ${i === 0 ? 'top' : ''}`;
    row.innerHTML = `<span class="res-name">${i+1}. ${name}</span>
                     <span class="res-pct">${pct}%</span>`;
    el.appendChild(row);
  });
}

function showVerdict(predsA, predsB, msA, msB) {
  const topA = predsA[0].className.split(',')[0];
  const topB = predsB[0].className.split(',')[0];
  const confA = (predsA[0].probability * 100).toFixed(1);
  const confB = (predsB[0].probability * 100).toFixed(1);

  const agree = topA.toLowerCase() === topB.toLowerCase();
  const fasterModel = msA <= msB ? 'MobileNet v2' : 'MobileNet v1';
  const higherConf  = parseFloat(confA) >= parseFloat(confB) ? `MobileNet v2 (${confA}%)` : `MobileNet v1 (${confB}%)`;

  const text = `
${agree ? '✅ Both models AGREE' : '⚠ Models DISAGREE'} on the top prediction.
MobileNet v2 → "${topA}" at ${confA}% in ${msA}ms
MobileNet v1 → "${topB}" at ${confB}% in ${msB}ms
Faster model: ${fasterModel}
Higher confidence: ${higherConf}
  `.trim();

  document.getElementById('verdictText').textContent = text;
  document.getElementById('verdictCard').style.display = 'block';
}

function resetCmp() {
  imgEl = null;
  document.getElementById('cmpImg').style.display = 'none';
  document.getElementById('upPrompt').style.display = 'flex';
  document.getElementById('cmpBtn').disabled = true;
  document.getElementById('cmpInput').value = '';
  document.getElementById('resA').innerHTML = '<div class="model-waiting">Run comparison to see results</div>';
  document.getElementById('resB').innerHTML = '<div class="model-waiting">Run comparison to see results</div>';
  document.getElementById('verdictCard').style.display = 'none';
  ['timeA','confA','timeB','confB'].forEach(id => document.getElementById(id).textContent = '--');
  log('Reset.', 'inf');
}

window.addEventListener('load', loadModels);