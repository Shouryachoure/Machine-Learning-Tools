let model = null, stream = null, isRunning = false, rafId = null;
let lastInferTime = 0;
let lastPredictions = [];
const INFER_INTERVAL = 250; // ms — stable, not too fast

const video  = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');

const fontSizeRange   = document.getElementById('fontSize');
const labelColorPick  = document.getElementById('labelColor');
const boxColorPick    = document.getElementById('boxColor');
const boxOpacityRange = document.getElementById('boxOpacity');

fontSizeRange.addEventListener('input', () => {
  document.getElementById('fontSizeVal').textContent = fontSizeRange.value + 'px';
});
boxOpacityRange.addEventListener('input', () => {
  document.getElementById('boxOpacityVal').textContent = boxOpacityRange.value + '%';
});

function log(msg, type='') {
  const box = document.getElementById('log');
  const d = document.createElement('div');
  d.className = `log-line ${type}`;
  d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  box.appendChild(d);
  box.scrollTop = box.scrollHeight;
}

function setStatus(on) {
  document.getElementById('statusDot').className = `status-dot ${on?'on':'off'}`;
}

async function start() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video:{width:640,height:480} });
    video.srcObject = stream;
    await new Promise(r => video.onloadedmetadata = r);
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;

    if (!model) {
      log('Loading MobileNet v2...', 'inf');
      model = await mobilenet.load({ version:2, alpha:1.0 });
      log('Model ready!', 'ok');
    }

    isRunning = true;
    setStatus(true);
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled  = false;
    log('Live overlay started.', 'ok');
    loop();
  } catch(e) { log(`Error: ${e.message}`, 'err'); }
}

async function loop() {
  if (!isRunning) return;

  const now = performance.now();

  if (now - lastInferTime >= INFER_INTERVAL) {
    lastInferTime = now;
    try {
      const raw = await model.classify(video, 3);
      // Keep only confident predictions
      const filtered = raw.filter(p => p.probability > 0.08);
      if (filtered.length) lastPredictions = filtered;
    } catch(e) {}
  }

  drawOverlay(lastPredictions);
  updateLiveList(lastPredictions);
  rafId = requestAnimationFrame(loop);
}

function drawOverlay(preds) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!preds.length) return;

  const showAll  = document.getElementById('showAll').checked;
  const showBox  = document.getElementById('showBox').checked;
  const fontSize = parseInt(fontSizeRange.value);
  const labelColor = labelColorPick.value;
  const boxColor   = boxColorPick.value;
  const boxAlpha   = parseInt(boxOpacityRange.value) / 100;

  ctx.save();
  ctx.scale(-1,1);
  ctx.translate(-canvas.width, 0);

  const pad=16, bx=pad, by=pad;
  const bw=canvas.width-pad*2, bh=canvas.height-pad*2;

  if (showBox) {
    const r=parseInt(boxColor.slice(1,3),16);
    const g=parseInt(boxColor.slice(3,5),16);
    const b=parseInt(boxColor.slice(5,7),16);
    ctx.fillStyle   = `rgba(${r},${g},${b},${boxAlpha})`;
    ctx.fillRect(bx,by,bw,bh);
    ctx.strokeStyle = boxColor;
    ctx.lineWidth   = 2;
    ctx.strokeRect(bx,by,bw,bh);

    const cl=20;
    ctx.lineWidth=3;
    [[bx,by,1,1],[bx+bw,by,-1,1],[bx,by+bh,1,-1],[bx+bw,by+bh,-1,-1]].forEach(([cx,cy,dx,dy])=>{
      ctx.beginPath();
      ctx.moveTo(cx+dx*cl,cy); ctx.lineTo(cx,cy); ctx.lineTo(cx,cy+dy*cl);
      ctx.stroke();
    });
  }

  const list = showAll ? preds : [preds[0]];
  list.forEach((p, i) => {
    const name = p.className.split(',')[0];
    const conf = (p.probability*100).toFixed(1);
    const lx   = bx+10;
    const ly   = by+bh-12-i*(fontSize+14);

    ctx.font = `bold ${fontSize}px Space Mono, monospace`;
    const tw = ctx.measureText(`${name} ${conf}%`).width+24;
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    ctx.fillRect(lx-4, ly-fontSize, tw, fontSize+8);

    ctx.fillStyle = i===0 ? labelColor : '#ffffff88';
    ctx.fillText(`${name}  ${conf}%`, lx, ly);
  });

  ctx.restore();
}

function updateLiveList(preds) {
  const list = document.getElementById('liveList');
  list.innerHTML = '';
  if (!preds.length) {
    list.innerHTML = '<div class="live-item empty">No detection</div>';
    return;
  }
  preds.forEach((p,i) => {
    const el = document.createElement('div');
    el.className = `live-item ${i===0?'top':''}`;
    el.innerHTML = `<span class="li-name">${p.className.split(',')[0]}</span>
                    <span class="li-pct">${(p.probability*100).toFixed(1)}%</span>`;
    list.appendChild(el);
  });
}

function stop() {
  isRunning = false;
  if (rafId) cancelAnimationFrame(rafId);
  if (stream) stream.getTracks().forEach(t=>t.stop());
  ctx.clearRect(0,0,canvas.width,canvas.height);
  setStatus(false);
  document.getElementById('startBtn').disabled = false;
  document.getElementById('stopBtn').disabled  = true;
  list.innerHTML = '<div class="live-item empty">Stopped</div>';
  log('Stopped.','inf');
}