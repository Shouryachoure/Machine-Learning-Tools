let model = null;
let stream = null;
const CONFIDENCE_THRESHOLD = 0.10; // ignore predictions below 10%

const video     = document.getElementById('video');
const snapHidden = document.getElementById('snap');
const snapShow   = document.getElementById('snapshot');
const snapCtx    = snapShow.getContext('2d');

function addLog(msg, type = '') {
  const box = document.getElementById('log');
  const d = document.createElement('div');
  d.className = `log-line ${type}`;
  d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  box.appendChild(d);
  box.scrollTop = box.scrollHeight;
}

async function startCam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width:640, height:480 } });
    video.srcObject = stream;
    document.getElementById('classifyBtn').disabled = false;
    document.getElementById('stopBtn').disabled = false;
    addLog('Webcam started.', 'inf');

    if (!model) {
      addLog('Loading MobileNet v2...', 'inf');
      model = await mobilenet.load({ version: 2, alpha: 1.0 });
      addLog('Model ready!', 'ok');
    }
  } catch (e) { addLog(`Error: ${e.message}`, 'err'); }
}

async function captureAndClassify() {
  if (!model || !stream) return;

  snapHidden.width  = video.videoWidth;
  snapHidden.height = video.videoHeight;
  const hCtx = snapHidden.getContext('2d');
  hCtx.drawImage(video, 0, 0);

  snapCtx.save();
  snapCtx.scale(-1, 1);
  snapCtx.translate(-snapShow.width, 0);
  snapCtx.drawImage(snapHidden, 0, 0, snapShow.width, snapShow.height);
  snapCtx.restore();

  addLog('Classifying...', 'inf');
  const t0 = performance.now();
  const raw = await model.classify(snapHidden, 5);
  const elapsed = (performance.now() - t0).toFixed(0);

  // Filter noise
  const predictions = raw.filter(p => p.probability > CONFIDENCE_THRESHOLD);

  if (!predictions.length) {
    addLog('No confident prediction. Try better lighting.', 'err');
    return;
  }

  addLog(`Done in ${elapsed}ms → ${predictions[0].className.split(',')[0]}`, 'ok');
  document.getElementById('snapLabel').textContent =
    `Captured ${new Date().toLocaleTimeString()} · ${elapsed}ms inference`;

  renderResults(predictions);
}

function renderResults(preds) {
  const area = document.getElementById('results');
  area.innerHTML = '';
  preds.forEach((p, i) => {
    const name = p.className.split(',')[0];
    const pct  = (p.probability * 100).toFixed(1);
    const row  = document.createElement('div');
    row.className = `result-row ${i === 0 ? 'first' : ''}`;
    row.innerHTML = `<span class="result-name">${i+1}. ${name}</span>
                     <span class="result-pct">${pct}%</span>`;
    area.appendChild(row);
  });
}

function stopCam() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  document.getElementById('classifyBtn').disabled = true;
  document.getElementById('stopBtn').disabled = true;
  addLog('Webcam stopped.', 'inf');
}