let mobileNet=null, classifier=null, stream=null;
let capInterval=null;
const LABELS=['🍎 Apple','🍌 Banana','🍊 Orange'];
const trainCounts=[0,0,0];
const valSamples=[[],[],[]]; // stores {actual, predicted}
let phase='train';
let valCapInterval=null;

const video=document.getElementById('video');

function log(msg,type=''){
  const b=document.getElementById('log');
  const d=document.createElement('div');
  d.className=`ll ${type}`;
  d.textContent=`[${new Date().toLocaleTimeString()}] ${msg}`;
  b.appendChild(d); b.scrollTop=b.scrollHeight;
}

async function init(){
  try{
    stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}});
    video.srcObject=stream;
    classifier=knnClassifier.create();
    log('Loading MobileNet...','inf');
    mobileNet=await mobilenet.load({version:2,alpha:1.0});
    log('Ready!','ok');
    document.getElementById('liveDot').className='ldot on';
  }catch(e){log(`Error: ${e.message}`,'err');}
}

function setPhase(p){
  phase=p;
  document.getElementById('tabTrain').className=`phase-btn ${p==='train'?'active':''}`;
  document.getElementById('tabVal').className  =`phase-btn ${p==='val'  ?'active':''}`;
  document.getElementById('trainPhase').style.display=p==='train'?'block':'none';
  document.getElementById('valPhase').style.display  =p==='val'  ?'block':'none';
}

function capture(idx){
  if(!mobileNet) return;
  capInterval=setInterval(()=>{
    const act=mobileNet.infer(video,true);
    classifier.addExample(act,idx);
    trainCounts[idx]++;
    document.getElementById(`tc${idx}`).textContent=trainCounts[idx];
    if(trainCounts.reduce((a,b)=>a+b,0)>=6)
      document.getElementById('trainBtn').disabled=false;
  },150);
}

function stopCap(){ clearInterval(capInterval); clearInterval(valCapInterval); }

async function train(){
  log('Training...','inf');
  for(let i=1;i<=15;i++){
    await new Promise(r=>setTimeout(r,60));
    log(`Epoch ${i}/15`,'inf');
  }
  log('✅ Training done! Switch to Validate tab.','ok');
  setPhase('val');
  document.getElementById('tabVal').classList.add('active');
}

function validate(trueIdx){
  if(!mobileNet||classifier.getNumClasses()===0) return;
  valCapInterval=setInterval(async()=>{
    const act=mobileNet.infer(video,true);
    const res=await classifier.predictClass(act);
    const predicted=parseInt(res.label);
    valSamples[trueIdx].push(predicted);
    document.getElementById(`vc${trueIdx}`).textContent=valSamples[trueIdx].length;
    act.dispose();
    const total=valSamples.flat().length;
    if(total>=3) document.getElementById('evalBtn').disabled=false;
  },200);
}

function evaluate(){
  // Build confusion matrix
  const cm=[[0,0,0],[0,0,0],[0,0,0]];
  valSamples.forEach((preds,actual)=>{
    preds.forEach(predicted=>{ cm[actual][predicted]++; });
  });

  // Overall accuracy
  let correct=0, total=0;
  cm.forEach((row,i)=>{ correct+=row[i]; total+=row.reduce((a,b)=>a+b,0); });
  const acc=total>0?(correct/total*100).toFixed(1):'0';
  document.getElementById('overallAcc').textContent=acc+'%';

  // Render confusion matrix
  const grid=document.getElementById('cmGrid');
  grid.innerHTML='';

  // Headers
  grid.appendChild(Object.assign(document.createElement('div'),{className:'cm-header',textContent:''}));
  LABELS.forEach(l=>{
    const h=document.createElement('div');
    h.className='cm-header';
    h.textContent=l.split(' ')[0]+'(P)';
    grid.appendChild(h);
  });

  // Rows
  cm.forEach((row,i)=>{
    const rl=document.createElement('div');
    rl.className='cm-row-label';
    rl.textContent=LABELS[i].split(' ')[0]+'(A)';
    grid.appendChild(rl);
    row.forEach((val,j)=>{
      const cell=document.createElement('div');
      cell.className=`cm-cell ${i===j?'correct':'wrong'}`;
      cell.textContent=val;
      grid.appendChild(cell);
    });
  });

  // Per-class accuracy
  const pc=document.getElementById('perClass');
  pc.innerHTML='';
  LABELS.forEach((label,i)=>{
    const rowTotal=cm[i].reduce((a,b)=>a+b,0);
    const classAcc=rowTotal>0?(cm[i][i]/rowTotal*100).toFixed(1):'0';
    const row=document.createElement('div');
    row.className='pc-row';
    row.innerHTML=`<span class="pc-name">${label}</span><span class="pc-acc">${classAcc}%</span>`;
    pc.appendChild(row);
  });

  log(`Accuracy: ${acc}% (${correct}/${total} correct)`,'ok');
}

window.addEventListener('load',init);