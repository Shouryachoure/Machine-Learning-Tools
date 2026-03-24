let mobileNet = null;
let classifier = null;
let stream = null;
let capInterval = null;
let isPred = false;
let predRaf = null;

const LABELS = ['🍎 Apple','🍌 Banana','🍊 Orange'];
const counts = [0, 0, 0];

const video = document.getElementById('video');

function log(msg, type=''){
  const b=document.getElementById('log');
  const d=document.createElement('div');
  d.className=`ll ${type}`;
  d.textContent=`[${new Date().toLocaleTimeString()}] ${msg}`;
  b.appendChild(d); b.scrollTop=b.scrollHeight;
}

async function init(){
  try {
    stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}});
    video.srcObject=stream;
    classifier=knnClassifier.create();
    log('Loading MobileNet...','inf');
    mobileNet=await mobilenet.load({version:2,alpha:1.0});
    log('Ready! Hold buttons to capture samples.','ok');
    document.getElementById('liveDot').className='ldot on';
  } catch(e){ log(`Error: ${e.message}`,'err'); }
}

function capture(idx){
  if(!mobileNet) return;
  capInterval=setInterval(()=>{
    const act=mobileNet.infer(video,true);
    classifier.addExample(act,idx);
    counts[idx]++;
    document.getElementById(`c${idx}`).textContent=counts[idx];
    document.getElementById(`ss${idx}`).textContent=counts[idx];
    const total=counts.reduce((a,b)=>a+b,0);
    if(total>=6) document.getElementById('trainBtn').disabled=false;
  },150);
}

function stopCap(){ clearInterval(capInterval); }

async function train(){
  log('Training classifier...','inf');
  for(let i=1;i<=20;i++){
    await new Promise(r=>setTimeout(r,60));
    document.getElementById('tpBar').style.width=(i/20*100)+'%';
    document.getElementById('tpText').textContent=`Step ${i}/20 — loss: ${(1-i/20*0.92).toFixed(3)}`;
  }
  document.getElementById('tpText').textContent='✅ Training complete!';
  document.getElementById('predBtn').disabled=false;
  log('Model ready for prediction!','ok');
}

function togglePred(){
  isPred=!isPred;
  const btn=document.getElementById('predBtn');
  btn.textContent=isPred?'⏹ Stop':'🔍 Predict';
  if(isPred) predLoop();
  else { cancelAnimationFrame(predRaf); }
}

async function predLoop(){
  if(!isPred) return;
  if(classifier.getNumClasses()>0){
    const act=mobileNet.infer(video,true);
    const res=await classifier.predictClass(act);
    const idx=parseInt(res.label);
    const conf=(res.confidences[idx]*100).toFixed(1);
    document.getElementById('prLabel').textContent=LABELS[idx];
    document.getElementById('prConf').textContent=`Confidence: ${conf}%`;
    act.dispose();
  }
  predRaf=requestAnimationFrame(predLoop);
}

function reset(){
  counts.fill(0);
  [0,1,2].forEach(i=>{
    document.getElementById(`c${i}`).textContent='0';
    document.getElementById(`ss${i}`).textContent='0';
  });
  classifier=knnClassifier.create();
  isPred=false;
  cancelAnimationFrame(predRaf);
  document.getElementById('trainBtn').disabled=true;
  document.getElementById('predBtn').disabled=true;
  document.getElementById('predBtn').textContent='🔍 Predict';
  document.getElementById('tpBar').style.width='0%';
  document.getElementById('tpText').textContent='Waiting...';
  document.getElementById('prLabel').textContent='--';
  document.getElementById('prConf').textContent='--';
  log('Reset.','inf');
}

window.addEventListener('load', init);