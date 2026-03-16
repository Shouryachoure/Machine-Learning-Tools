let model = null;
let currentImage = null;

function log(msg, type = '') {
  const box = document.getElementById('log');
  const d = document.createElement('div');
  d.className = `log-entry ${type}`;
  d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  box.appendChild(d);
  box.scrollTop = box.scrollHeight;
}

async function loadModel() {
  log('Loading MobileNet v2...', 'info');
  model = await mobilenet.load({ version: 2, alpha: 1.0 });
  log('✅ MobileNet ready!', 'ok');
}

function handleFile(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    const img = document.getElementById('previewImg');
    img.src = e.target.result;
    img.onload = () => {
      currentImage = img;
      document.getElementById('previewWrap').style.display = 'block';
      document.getElementById('dropZone').style.display = 'none';
      document.getElementById('classifyBtn').disabled = false;
      log(`Image loaded: ${file.name}`, 'info');
    };
  };
  reader.readAsDataURL(file);
}

// Drag & drop support
const dropZone = document.getElementById('dropZone');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor = '#ff2d2d'; });
dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = ''; });
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.style.borderColor = '';
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    const dt = new DataTransfer();
    dt.items.add(file);
    document.getElementById('fileInput').files = dt.files;
    handleFile({ target: { files: [file] } });
  }
});

async function classifyImage() {
  if (!currentImage) return;

  if (!model) await loadModel();

  log('Classifying...', 'info');
  const t0 = performance.now();
  const predictions = await model.classify(currentImage, 3);
  const elapsed = Math.round(performance.now() - t0);

  log(`Done in ${elapsed}ms → ${predictions[0].className.split(',')[0]}`, 'ok');

  document.getElementById('inferTime').textContent = elapsed;
  document.getElementById('topLabel').textContent  = predictions[0].className.split(',')[0];
  document.getElementById('topConf').textContent   = `Confidence: ${(predictions[0].probability * 100).toFixed(1)}%`;

  renderPredictions(predictions);
}

function renderPredictions(preds) {
  const list = document.getElementById('predList');
  list.innerHTML = '';
  preds.forEach((p, i) => {
    const name = p.className.split(',')[0];
    const pct  = (p.probability * 100).toFixed(1);
    const item = document.createElement('div');
    item.className = `pred-item ${i === 0 ? 'top' : ''}`;
    item.innerHTML = `
      <div class="pred-row">
        <span class="pred-name">${i + 1}. ${name}</span>
        <span class="pred-pct">${pct}%</span>
      </div>
      <div class="pred-bar-wrap">
        <div class="pred-bar" style="width:${pct}%"></div>
      </div>`;
    list.appendChild(item);
  });
}

function resetAll() {
  currentImage = null;
  document.getElementById('previewWrap').style.display = 'none';
  document.getElementById('dropZone').style.display = 'block';
  document.getElementById('classifyBtn').disabled = true;
  document.getElementById('fileInput').value = '';
  document.getElementById('predList').innerHTML = '<div class="pred-empty">No predictions yet</div>';
  document.getElementById('topLabel').textContent = 'Upload an image to begin';
  document.getElementById('topConf').textContent  = '--';
  document.getElementById('inferTime').textContent = '--';
  log('Reset.', 'info');
}

window.addEventListener('load', loadModel);