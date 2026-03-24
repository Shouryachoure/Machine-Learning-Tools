let net = null;
let stream = null;
let isRunning = false;
let rafId = null;
let lastTime = 0;
let fpsSamples = [];

const video  = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx    = canvas.getContext('2d');

const KEYPOINT_NAMES = [
  'nose','leftEye','rightEye','leftEar','rightEar',
  'leftShoulder','rightShoulder','leftElbow','rightElbow',
  'leftWrist','rightWrist','leftHip','rightHip',
  'leftKnee','rightKnee','leftAnkle','rightAnkle'
];

const SKELETON_PAIRS = [
  ['leftShoulder','rightShoulder'],['leftShoulder','leftElbow'],
  ['leftElbow','leftWrist'],['rightShoulder','rightElbow'],
  ['rightElbow','rightWrist'],['leftShoulder','leftHip'],
  ['rightShoulder','rightHip'],['leftHip','rightHip'],
  ['leftHip','leftKnee'],['leftKnee','leftAnkle'],
  ['rightHip','rightKnee'],['rightKnee','rightAnkle'],
  ['nose','leftEye'],['nose','rightEye'],
  ['leftEye','leftEar'],['rightEye','rightEar']
];

function log(msg, type = '') {
  const box = document.getElementById('log');
  const d = document.createElement('div');
  d.className = `log-entry ${type}`;
  d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  box.appendChild(d); box.scrollTop = box.scrollHeight;
}

async function loadModel() {
  log('Loading PoseNet...', 'info');
  net = await posenet.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    inputResolution: { width: 640, height: 480 },
    multiplier: 0.75
  });
  log('✅ PoseNet ready!', 'ok');
}

async function startDetection() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width:640, height:480 } });
    video.srcObject = stream;
    await new Promise(r => video.onloadedmetadata = r);
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;

    if (!net) await loadModel();

    isRunning = true;
    lastTime  = performance.now();
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled  = false;
    document.getElementById('liveBadge').textContent = '● LIVE';
    document.getElementById('liveBadge').className   = 'live-badge on';
    log('🎥 Detection started!', 'ok');
    loop();
  } catch(e) { log(`❌ ${e.message}`, 'err'); }
}

async function loop() {
  if (!isRunning) return;

  const now = performance.now();
  const fps = Math.round(1000 / (now - lastTime));
  lastTime  = now;
  fpsSamples.push(fps);
  if (fpsSamples.length > 30) fpsSamples.shift();

  const t0   = performance.now();
  const pose = await net.estimateSinglePose(video, { flipHorizontal: true });
  const ms   = Math.round(performance.now() - t0);

  drawPose(pose);
  updateUI(pose, fps, ms);

  rafId = requestAnimationFrame(loop);
}

function drawPose(pose) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const kps = {};
  pose.keypoints.forEach(kp => { kps[kp.part] = kp; });

  // Draw skeleton lines
  SKELETON_PAIRS.forEach(([a, b]) => {
    const kpA = kps[a], kpB = kps[b];
    if (!kpA || !kpB) return;
    if (kpA.score < 0.3 || kpB.score < 0.3) return;
    ctx.beginPath();
    ctx.moveTo(kpA.position.x, kpA.position.y);
    ctx.lineTo(kpB.position.x, kpB.position.y);
    ctx.strokeStyle = 'rgba(255,107,53,0.8)';
    ctx.lineWidth   = 3;
    ctx.stroke();
  });

  // Draw keypoint dots
  pose.keypoints.forEach(kp => {
    if (kp.score < 0.3) return;
    ctx.beginPath();
    ctx.arc(kp.position.x, kp.position.y, 6, 0, 2 * Math.PI);
    ctx.fillStyle   = kp.score > 0.7 ? '#ff2d2d' : '#ff6b35';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth   = 2;
    ctx.fill();
    ctx.stroke();
  });
}

function updateUI(pose, fps, ms) {
  const confident = pose.keypoints.filter(k => k.score > 0.3);
  document.getElementById('keypointCount').textContent = confident.length;
  document.getElementById('poseScore').textContent     = (pose.score * 100).toFixed(1) + '%';
  document.getElementById('fps').textContent           = fps;
  document.getElementById('inferMs').textContent       = ms;

  // Top 5 keypoints by confidence
  const sorted = [...pose.keypoints].sort((a,b) => b.score - a.score).slice(0,5);
  const list   = document.getElementById('keypointList');
  list.innerHTML = '';
  sorted.forEach(kp => {
    const item = document.createElement('div');
    item.className = 'kp-item';
    item.innerHTML = `<span class="kp-name">${kp.part}</span>
                      <span class="kp-score">${(kp.score*100).toFixed(1)}%</span>`;
    list.appendChild(item);
  });
}

function stopDetection() {
  isRunning = false;
  if (rafId) cancelAnimationFrame(rafId);
  if (stream) stream.getTracks().forEach(t => t.stop());
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById('startBtn').disabled = false;
  document.getElementById('stopBtn').disabled  = true;
  document.getElementById('liveBadge').textContent = '● IDLE';
  document.getElementById('liveBadge').className   = 'live-badge off';
  log('⏹ Stopped.', 'info');
}

window.addEventListener('load', loadModel);