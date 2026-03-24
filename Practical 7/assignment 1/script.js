let net = null, stream = null, isRunning = false, rafId = null;
let lastTime = 0;

const video  = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');

const SKELETON = [
  ['leftShoulder','rightShoulder'],['leftShoulder','leftElbow'],
  ['leftElbow','leftWrist'],['rightShoulder','rightElbow'],
  ['rightElbow','rightWrist'],['leftShoulder','leftHip'],
  ['rightShoulder','rightHip'],['leftHip','rightHip'],
  ['leftHip','leftKnee'],['leftKnee','leftAnkle'],
  ['rightHip','rightKnee'],['rightKnee','rightAnkle'],
  ['nose','leftEye'],['nose','rightEye'],
  ['leftEye','leftEar'],['rightEye','rightEar']
];

function log(msg, type=''){
  const b=document.getElementById('log');
  const d=document.createElement('div');
  d.className=`ll ${type}`;
  d.textContent=`[${new Date().toLocaleTimeString()}] ${msg}`;
  b.appendChild(d); b.scrollTop=b.scrollHeight;
}

async function start(){
  try {
    stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}});
    video.srcObject=stream;
    await new Promise(r=>video.onloadedmetadata=r);
    canvas.width=video.videoWidth; canvas.height=video.videoHeight;

    if(!net){
      log('Loading PoseNet...','inf');
      net=await posenet.load({architecture:'MobileNetV1',outputStride:16,inputResolution:{width:640,height:480},multiplier:0.75});
      log('PoseNet ready!','ok');
    }

    isRunning=true; lastTime=performance.now();
    document.getElementById('stopBtn').disabled=false;
    document.getElementById('liveDot').className='ldot on';
    log('Detection started.','ok');
    loop();
  } catch(e){ log(`Error: ${e.message}`,'err'); }
}

async function loop(){
  if(!isRunning) return;
  const now=performance.now();
  const fps=Math.round(1000/(now-lastTime)); lastTime=now;

  const pose=await net.estimateSinglePose(video,{flipHorizontal:true});
  draw(pose);
  updateKPs(pose, fps);
  rafId=requestAnimationFrame(loop);
}

function draw(pose){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const kps={};
  pose.keypoints.forEach(k=>kps[k.part]=k);

  // Skeleton
  SKELETON.forEach(([a,b])=>{
    const ka=kps[a], kb=kps[b];
    if(!ka||!kb||ka.score<0.3||kb.score<0.3) return;
    ctx.beginPath();
    ctx.moveTo(ka.position.x,ka.position.y);
    ctx.lineTo(kb.position.x,kb.position.y);
    ctx.strokeStyle='rgba(255,107,53,0.85)';
    ctx.lineWidth=3; ctx.stroke();
  });

  // Dots
  pose.keypoints.forEach(kp=>{
    if(kp.score<0.3) return;
    ctx.beginPath();
    ctx.arc(kp.position.x,kp.position.y,6,0,2*Math.PI);
    ctx.fillStyle=kp.score>0.7?'#ff2d2d':'#ff6b35';
    ctx.strokeStyle='#fff'; ctx.lineWidth=2;
    ctx.fill(); ctx.stroke();
  });
}

function updateKPs(pose, fps){
  const active=pose.keypoints.filter(k=>k.score>0.3);
  document.getElementById('poseScore').textContent=(pose.score*100).toFixed(1)+'%';
  document.getElementById('kpCount').textContent=active.length;
  document.getElementById('fpsVal').textContent=fps;

  const grid=document.getElementById('kpGrid');
  grid.innerHTML='';
  pose.keypoints.forEach(kp=>{
    const cell=document.createElement('div');
    cell.className=`kp-cell ${kp.score>0.3?'active':''}`;
    cell.innerHTML=`<span class="kc-name">${kp.part}</span>
                    <span class="kc-val">${(kp.score*100).toFixed(0)}%</span>`;
    grid.appendChild(cell);
  });
}

function stop(){
  isRunning=false;
  if(rafId) cancelAnimationFrame(rafId);
  if(stream) stream.getTracks().forEach(t=>t.stop());
  ctx.clearRect(0,0,canvas.width,canvas.height);
  document.getElementById('stopBtn').disabled=true;
  document.getElementById('liveDot').className='ldot off';
  log('Stopped.','inf');
}

window.addEventListener('load',async()=>{
  log('Loading PoseNet...','inf');
  net=await posenet.load({architecture:'MobileNetV1',outputStride:16,inputResolution:{width:640,height:480},multiplier:0.75});
  log('Ready!','ok');
});