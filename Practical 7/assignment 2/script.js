let net=null, stream=null, isRunning=false, rafId=null;
let repCount=0, phase='up'; // phase: 'up' or 'down'

const video  = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');

const SKELETON=[
  ['leftShoulder','rightShoulder'],['leftShoulder','leftElbow'],
  ['leftElbow','leftWrist'],['rightShoulder','rightElbow'],
  ['rightElbow','rightWrist'],['leftShoulder','leftHip'],
  ['rightShoulder','rightHip'],['leftHip','rightHip'],
  ['leftHip','leftKnee'],['leftKnee','leftAnkle'],
  ['rightHip','rightKnee'],['rightKnee','rightAnkle']
];

function log(msg,type=''){
  const b=document.getElementById('log');
  const d=document.createElement('div');
  d.className=`ll ${type}`;
  d.textContent=`[${new Date().toLocaleTimeString()}] ${msg}`;
  b.appendChild(d); b.scrollTop=b.scrollHeight;
}

// Calculate angle between 3 points (in degrees)
function calcAngle(A, B, C) {
  const ab = { x: A.x-B.x, y: A.y-B.y };
  const cb = { x: C.x-B.x, y: C.y-B.y };
  const dot  = ab.x*cb.x + ab.y*cb.y;
  const magA = Math.sqrt(ab.x**2 + ab.y**2);
  const magC = Math.sqrt(cb.x**2 + cb.y**2);
  const cos  = Math.min(1, Math.max(-1, dot / (magA * magC)));
  return Math.round(Math.acos(cos) * (180 / Math.PI));
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
      log('Ready!','ok');
    }

    isRunning=true;
    document.getElementById('stopBtn').disabled=false;
    document.getElementById('liveDot').className='ldot on';
    log('Squat counter started. Stand in view!','ok');
    loop();
  } catch(e){ log(`Error: ${e.message}`,'err'); }
}

async function loop(){
  if(!isRunning) return;
  const pose=await net.estimateSinglePose(video,{flipHorizontal:true});
  processPose(pose);
  draw(pose);
  rafId=requestAnimationFrame(loop);
}

function processPose(pose){
  const kps={};
  pose.keypoints.forEach(k=>kps[k.part]=k);

  // Left knee angle: leftHip → leftKnee → leftAnkle
  let leftAngle=null, rightAngle=null;

  if(kps.leftHip?.score>0.3 && kps.leftKnee?.score>0.3 && kps.leftAnkle?.score>0.3){
    leftAngle=calcAngle(kps.leftHip.position, kps.leftKnee.position, kps.leftAnkle.position);
    document.getElementById('leftAngle').textContent=leftAngle+'°';
  }

  if(kps.rightHip?.score>0.3 && kps.rightKnee?.score>0.3 && kps.rightAnkle?.score>0.3){
    rightAngle=calcAngle(kps.rightHip.position, kps.rightKnee.position, kps.rightAnkle.position);
    document.getElementById('rightAngle').textContent=rightAngle+'°';
  }

  const avgAngle = leftAngle !== null && rightAngle !== null
    ? (leftAngle + rightAngle) / 2
    : leftAngle ?? rightAngle;

  if(avgAngle === null) return;

  // Squat logic
  if(avgAngle < 100 && phase === 'up'){
    phase = 'down';
    document.getElementById('phaseText').textContent = '⬇ DOWN';
    document.getElementById('phaseBox').style.borderColor = '#ff2d2d';
  } else if(avgAngle > 160 && phase === 'down'){
    phase = 'up';
    repCount++;
    document.getElementById('repCount').textContent = repCount;
    document.getElementById('phaseText').textContent = '⬆ UP';
    document.getElementById('phaseBox').style.borderColor = '#ff6b35';
    log(`Rep #${repCount} counted! Angle: ${Math.round(avgAngle)}°`, 'ok');
  }
}

function draw(pose){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const kps={};
  pose.keypoints.forEach(k=>kps[k.part]=k);

  SKELETON.forEach(([a,b])=>{
    const ka=kps[a], kb=kps[b];
    if(!ka||!kb||ka.score<0.3||kb.score<0.3) return;
    ctx.beginPath();
    ctx.moveTo(ka.position.x,ka.position.y);
    ctx.lineTo(kb.position.x,kb.position.y);
    ctx.strokeStyle='rgba(255,107,53,0.8)';
    ctx.lineWidth=3; ctx.stroke();
  });

  pose.keypoints.forEach(kp=>{
    if(kp.score<0.3) return;
    // Highlight knee joints
    const isKnee=kp.part.includes('Knee')||kp.part.includes('Hip')||kp.part.includes('Ankle');
    ctx.beginPath();
    ctx.arc(kp.position.x,kp.position.y,isKnee?9:5,0,2*Math.PI);
    ctx.fillStyle=isKnee?'#ff2d2d':'#ff6b35';
    ctx.strokeStyle='#fff'; ctx.lineWidth=2;
    ctx.fill(); ctx.stroke();
  });

  // Draw rep count on canvas
  ctx.fillStyle='rgba(0,0,0,0.6)';
  ctx.fillRect(10,10,140,50);
  ctx.strokeStyle='#ff2d2d'; ctx.lineWidth=1;
  ctx.strokeRect(10,10,140,50);
  ctx.fillStyle='#ff2d2d';
  ctx.font='bold 28px Space Mono, monospace';
  ctx.fillText(`REPS: ${repCount}`,18,44);
}

function resetCount(){
  repCount=0; phase='up';
  document.getElementById('repCount').textContent='0';
  document.getElementById('phaseText').textContent='--';
  document.getElementById('leftAngle').textContent='--°';
  document.getElementById('rightAngle').textContent='--°';
  log('Counter reset.','inf');
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