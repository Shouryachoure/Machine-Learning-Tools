let model = null;
let stream = null;
let isRunning = false;
let animFrame = null;

// Stability: throttle inference to every N ms
const INFERENCE_INTERVAL = 300; // ms between classifications
let lastInferTime = 0;

// Smoothing: keep last predictions to avoid flicker
let lastPredictions = [];

// FPS tracking
let lastFrameTime = 0;
let totalFrames = 0;
let fpsSamples = [];

const video  = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx    = canvas.getContext('2d');

function log(msg, type = '') {
  const el = document.getElementById('statusLog');
  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  entry.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  el.appendChild(entry);
  el.scrollTop = el.scrollHeight;
}

async function loadModel() {
  log('Loading MobileNet v2 (alpha 1.0)...', 'info');
  // version 2, alpha 1.0 = most accurate MobileNet available in tfjs
  model = await mobilenet.load({ version: 2, alpha: 1.0 });
  log('✅ Model loaded!', 'success');
}

async function startDetection() {
  try {
    log('Requesting webcam...', 'info');
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }
    });
    video.srcObject = stream;
    await new Promise(res => video.onloadedmetadata = res);

    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;

    if (!model) await loadModel();

    isRunning = true;
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled  = false;

    lastFrameTime = performance.now();
    lastInferTime = 0;
    totalFrames   = 0;
    fpsSamples    = [];
    lastPredictions = [];

    log('🎥 Detection running!', 'success');
    renderLoop();

  } catch (err) {
    log(`❌ ${err.message}`, 'error');
  }
}

async function renderLoop() {
  if (!isRunning) return;

  const now     = performance.now();
  const elapsed = now - lastFrameTime;
  const fps     = Math.round(1000 / elapsed);
  lastFrameTime = now;

  totalFrames++;
  fpsSamples.push(fps);
  if (fpsSamples.length > 60) fpsSamples.shift();
  const avgFps = Math.round(fpsSamples.reduce((a,b) => a+b, 0) / fpsSamples.length);

  // Only run inference every INFERENCE_INTERVAL ms for stability
  if (now - lastInferTime >= INFERENCE_INTERVAL) {
    lastInferTime = now;
    const t0 = performance.now();

    try {
      const raw = await model.classify(video, 5);

      // Filter low-confidence noise (below 5%)
      const filtered = raw.filter(p => p.probability > 0.05);

      // Only update if top prediction changed significantly (avoid flicker)
      if (filtered.length > 0) {
        const topNew = filtered[0].className;
        const topOld = lastPredictions[0]?.className ?? '';
        // Accept new prediction only if confidence > 15% or same class
        if (filtered[0].probability > 0.15 || topNew === topOld) {
          lastPredictions = filtered;
        }
      }

      const inferTime = Math.round(performance.now() - t0);
      updateStats(fps, inferTime, avgFps, totalFrames);
      updatePredictions(lastPredictions);
    } catch (e) {
      log(`Inference error: ${e.message}`, 'error');
    }
  }

  drawOverlay(lastPredictions);
  document.getElementById('fps-badge').textContent = `FPS: ${fps}`;

  animFrame = requestAnimationFrame(renderLoop);
}

function updatePredictions(predictions) {
  if (!predictions.length) return;

  const top = predictions[0];
  document.getElementById('topLabel').textContent = top.className.split(',')[0];
  document.getElementById('topConf').textContent  =
    `Confidence: ${(top.probability * 100).toFixed(1)}%`;

  const list = document.getElementById('predictionList');
  list.innerHTML = '';

  predictions.slice(0, 3).forEach((p, i) => {
    const name = p.className.split(',')[0];
    const conf = (p.probability * 100).toFixed(1);

    const item = document.createElement('div');
    item.className = `pred-item ${i === 0 ? 'top' : ''}`;
    item.innerHTML = `
      <div class="pred-row">
        <span class="pred-name">${name}</span>
        <span class="pred-conf">${conf}%</span>
      </div>
      <div class="pred-bar-wrap">
        <div class="pred-bar" style="width:${conf}%"></div>
      </div>`;
    list.appendChild(item);
  });
}

function drawOverlay(predictions) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!predictions.length) return;

  const top  = predictions[0];
  const name = top.className.split(',')[0];
  const conf = (top.probability * 100).toFixed(1);

  ctx.save();
  ctx.scale(-1, 1);
  ctx.translate(-canvas.width, 0);

  // Corner brackets
  const bx = 20, by = 20;
  const bw = canvas.width  - 40;
  const bh = canvas.height - 40;
  const bl = 28;

  ctx.strokeStyle = '#ff2d2d';
  ctx.lineWidth   = 3;

  [
    [bx, by, bx+bl, by, bx, by+bl],
    [bx+bw-bl, by, bx+bw, by, bx+bw, by+bl],
    [bx, by+bh-bl, bx, by+bh, bx+bl, by+bh],
    [bx+bw-bl, by+bh, bx+bw, by+bh, bx+bw, by+bh-bl]
  ].forEach(([x1,y1,x2,y2,x3,y3]) => {
    ctx.beginPath();
    ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.lineTo(x3,y3);
    ctx.stroke();
  });

  // Label background
  ctx.fillStyle   = 'rgba(255,45,45,0.18)';
  ctx.fillRect(bx, by + bh - 54, bw, 54);
  ctx.strokeStyle = '#ff2d2d';
  ctx.lineWidth   = 1;
  ctx.strokeRect(bx, by + bh - 54, bw, 54);

  ctx.fillStyle = '#ff6b35';
  ctx.font      = 'bold 18px Space Mono, monospace';
  ctx.fillText(name, bx + 12, by + bh - 28);

  ctx.fillStyle = '#ff2d2d';
  ctx.font      = '13px Space Mono, monospace';
  ctx.fillText(`${conf}% confidence`, bx + 12, by + bh - 10);

  ctx.restore();
}

function updateStats(fps, inferTime, avgFps, frames) {
  document.getElementById('fpsVal').textContent       = fps;
  document.getElementById('inferenceVal').textContent = inferTime;
  document.getElementById('avgFpsVal').textContent    = avgFps;
  document.getElementById('frameCount').textContent   = frames;
}

function stopDetection() {
  isRunning = false;
  if (animFrame) cancelAnimationFrame(animFrame);
  if (stream) stream.getTracks().forEach(t => t.stop());
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById('startBtn').disabled = false;
  document.getElementById('stopBtn').disabled  = true;
  log('⏹ Stopped.', 'info');
}

window.addEventListener('load', async () => {
  try { await loadModel(); }
  catch(e) { log(`⚠ Preload failed: ${e.message}`, 'error'); }
});