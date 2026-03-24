let net=null, stream=null, isRunning=false, rafId=null;
let currentMode='single';
let lastTime=0;
let singleSamples=[], multiSamples=[];

const COLORS=['#ff2d2d','#ff6b35','#ffaa00','#ff0066','#cc00ff'];

const video  = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');

const SKELETON=[
  ['leftShoulder','rightShoulder'],['leftShoulder','leftElbow'],
  ['leftElbow','leftWrist'],['rightShoulder','rightElbow'],
  ['rightElbow','rightWrist'],['leftShoulder','leftHip'],
  ['rightShoulder','rightHip'],['leftHip','rightHip'],
  ['leftHip','leftKnee'],['leftKnee','leftAnkle'],
  ['rightHip','rightKnee'],['rightKnee','rightAnkle'],
  ['nose','leftEye'],['nose','rightEye'],
  ['leftEye','leftEar'],['rightEye','rightEar']
];

function log(msg,type=''){
  const b=document.getElementById('log');
  const d=document.createElement('div');
  d.className=`ll ${type}`;
  d.textContent=`[${new Date().toLocaleTimeString()}] ${msg}`;
  b.appendChild(d); b.scrollTop=b.scrollHeight;
}

function setMode(mode){
  currentMode=mode;
  document.getElementById('btnSingle').className=`mode-btn ${mode==='single'?'active':''}`;
  document.getElementById('btnMulti').className =`mode-btn ${mode==='multi' ?'active':''}`;
  document.getElementById('modeTitle').textContent = mode==='single'?'single_pose.js':'multi_pose.js';

  if(mode==='single'){
    document.getElementById('miTitle').textContent='Single Pose Mode';
    document.getElementById('miDesc').textContent='Detects one person. Faster inference, higher per-person accuracy.';
  } else {
    document.getElementById('miTitle').textContent='Multi Pose Mode';
    document.getElementById('miDesc').textContent='Detects multiple people simultaneously. Slower, lower per-person score.';
  }

  log(`Switched to ${mode} pose mode.`,'inf');
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

    isRunning=true; lastTime=performance.now();
    document.getElementById('stopBtn').disabled=false;
    document.getElementById('liveDot').className='ldot on';
    log(`${currentMode} pose detection started.`,'ok');
    loop();
  } catch(e){ log(`Error: ${e.message}`,'err'); }
}

async function loop(){
  if(!isRunning) return;

  const now=performance.now();
  const fps=Math.round(1000/(now-lastTime)); lastTime=now;

  const t0=performance.now();
  let poses=[];

  if(currentMode==='single'){
    const p=await net.estimateSinglePose(video,{flipHorizontal:true});
    poses=[p];
    const ms=Math.round(performance.now()-t0);
    singleSamples.push(ms);
    updateStats(poses,fps,ms);
  } else {
    poses=await net.estimateMultiplePoses(video,{
      flipHorizontal:true, maxDetections:5,
      scoreThreshold:0.5, nmsRadius:20
    });
    const ms=Math.round(performance.now()-t0);
    multiSamples.push(ms);
    updateStats(poses,fps,ms);
  }

  drawPoses(poses);
  updateAverages();
  rafId=requestAnimationFrame(loop);
}

function drawPoses(poses){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  poses.forEach((pose,pi)=>{
    const color=COLORS[pi%COLORS.length];
    const kps={};
    pose.keypoints.forEach(k=>kps[k.part]=k);

    SKELETON.forEach(([a,b])=>{
      const ka=kps[a], kb=kps[b];
      if(!ka||!kb||ka.score<0.3||kb.score<0.3) return;
      ctx.beginPath();
      ctx.moveTo(ka.position.x,ka.position.y);
      ctx.lineTo(kb.position.x,kb.position.y);
      ctx.strokeStyle=color+'cc'; ctx.lineWidth=3; ctx.stroke();
    });

    pose.keypoints.forEach(kp=>{
      if(kp.score<0.3) return;
      ctx.beginPath();
      ctx.arc(kp.position.x,kp.position.y,6,0,2*Math.PI);
      ctx.fillStyle=color; ctx.strokeStyle='#fff'; ctx.lineWidth=2;
      ctx.fill(); ctx.stroke();
    });
  });
}

function updateStats(poses,fps,ms){
  const avgScore=poses.reduce((s,p)=>s+p.score,0)/poses.length;
  document.getElementById('posesDetected').textContent=poses.length;
  document.getElementById('avgScore').textContent=(avgScore*100).toFixed(1)+'%';
  document.getElementById('inferTime').textContent=ms;
  document.getElementById('fpsVal').textContent=fps;
}

function updateAverages(){
  const avg=arr=>arr.length?Math.round(arr.reduce((a,b)=>a+b,0)/arr.length):null;
  const as=avg(singleSamples), am=avg(multiSamples);
  document.getElementById('avgSingle').textContent=as!==null?as+'ms':'--';
  document.getElementById('avgMulti').textContent =am!==null?am+'ms':'--';

  if(as!==null && am!==null){
    const diff=am-as;
    document.getElementById('verdictText').textContent=
      `Single Pose avg: ${as}ms\nMulti Pose avg: ${am}ms\n`+
      `Multi Pose is ${diff>0?diff+'ms slower':'faster'} than Single.\n`+
      `Single Pose is better for accuracy per person.\nMulti Pose enables detecting several people at once.`;
    document.getElementById('verdict').style.display='block';
  }
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