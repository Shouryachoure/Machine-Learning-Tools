let mobileNet=null, classifier=null, stream=null;
let isPred=false, predRaf=null, capInterval=null;

// Starting 3 categories always present
const ALL_CATS=[
  {id:0,emoji:'🍎',name:'Apple'},
  {id:1,emoji:'🍌',name:'Banana'},
  {id:2,emoji:'🍊',name:'Orange'},
  {id:3,emoji:'🍇',name:'Grapes'},
  {id:4,emoji:'🍓',name:'Strawberry'},
  {id:5,emoji:'🥭',name:'Mango'},
];

let activeIds=[0,1,2]; // start with 3
const sampleCounts={};
ALL_CATS.forEach(c=>sampleCounts[c.id]=0);

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
    log('Ready! Capture samples for each category.','ok');
    document.getElementById('liveDot').className='ldot on';
    renderActiveClasses();
    renderTimeline();
  }catch(e){log(`Error: ${e.message}`,'err');}
}

function renderActiveClasses(){
  const area=document.getElementById('activeClasses');
  area.innerHTML='';
  activeIds.forEach(id=>{
    const cat=ALL_CATS[id];
    const row=document.createElement('div');
    row.className='cat-row';
    row.innerHTML=`
      <span class="cat-emoji">${cat.emoji}</span>
      <span class="cat-name">${cat.name}</span>
      <span class="cat-count" id="cnt-${id}">${sampleCounts[id]} samples</span>
      <button class="cat-capture"
        onmousedown="startCapture(${id})"
        onmouseup="stopCap()"
        ontouchstart="startCapture(${id})"
        ontouchend="stopCap()">
        Hold
      </button>`;
    area.appendChild(row);
  });
}

function renderTimeline(){
  const tl=document.getElementById('timeline');
  const items=activeIds.map(id=>{
    const cat=ALL_CATS[id];
    return `<div class="tl-item">
      <div class="tl-dot"></div>
      <span class="tl-name">${cat.emoji} ${cat.name}</span>
      <span class="tl-samples">${sampleCounts[id]} samples</span>
    </div>`;
  }).join('');
  tl.innerHTML=`<div class="tl-label">CATEGORY TIMELINE</div>${items||'<div class="tl-empty">No categories yet</div>'}`;
}

function addCategory(){
  const sel=document.getElementById('newCatSelect');
  const id=parseInt(sel.value);
  if(activeIds.includes(id)){ log('Already added!','err'); return; }
  activeIds.push(id);
  const cat=ALL_CATS[id];
  log(`Added ${cat.emoji} ${cat.name} (${activeIds.length} categories now)`, 'ok');
  renderActiveClasses();
  renderTimeline();

  // Remove from dropdown
  const opt=sel.querySelector(`option[value="${id}"]`);
  if(opt) opt.remove();

  updateAnalysis();
}

function startCapture(id){
  if(!mobileNet) return;
  capInterval=setInterval(()=>{
    const act=mobileNet.infer(video,true);
    classifier.addExample(act,id);
    sampleCounts[id]++;
    const el=document.getElementById(`cnt-${id}`);
    if(el) el.textContent=sampleCounts[id]+' samples';
    renderTimeline();

    const totalSamples=Object.values(sampleCounts).reduce((a,b)=>a+b,0);
    if(totalSamples>=activeIds.length*2)
      document.getElementById('predBtn').disabled=false;
  },150);
}

function stopCap(){ clearInterval(capInterval); }

function togglePred(){
  isPred=!isPred;
  const btn=document.getElementById('predBtn');
  btn.textContent=isPred?'⏹ Stop Predicting':'🔍 Start Predicting';
  if(isPred) predLoop();
  else cancelAnimationFrame(predRaf);
}

async function predLoop(){
  if(!isPred) return;
  if(classifier.getNumClasses()>0){
    const act=mobileNet.infer(video,true);
    const res=await classifier.predictClass(act);
    const predId=parseInt(res.label);
    const cat=ALL_CATS[predId];
    const conf=(res.confidences[predId]*100).toFixed(1);

    document.getElementById('cpLabel').textContent=`${cat.emoji} ${cat.name}`;
    document.getElementById('cpConf').textContent=`Confidence: ${conf}%`;

    // Confidence bars
    const bars=document.getElementById('confBars');
    bars.innerHTML='';
    activeIds.forEach(id=>{
      const c=((res.confidences[id]||0)*100).toFixed(1);
      const row=document.createElement('div');
      row.className='cb-row';
      row.innerHTML=`
        <span class="cb-name">${ALL_CATS[id].emoji} ${ALL_CATS[id].name}</span>
        <div class="cb-wrap"><div class="cb-fill" style="width:${c}%"></div></div>
        <span class="cb-pct">${c}%</span>`;
      bars.appendChild(row);
    });

    act.dispose();
    updateAnalysis();
  }
  predRaf=requestAnimationFrame(predLoop);
}

function updateAnalysis(){
  const n=activeIds.length;
  const txt=
    n<=3 ? `${n} categories loaded. Model has clear decision boundaries with few classes.`
    : n<=4 ? `${n} categories: slight increase in confusion between similar-looking items expected.`
    : `${n} categories: with more classes, per-class confidence may drop. Capture more samples per class to maintain accuracy.`;
  document.getElementById('analysisText').textContent=txt;
}

function resetAll(){
  ALL_CATS.forEach(c=>sampleCounts[c.id]=0);
  activeIds=[0,1,2];
  classifier=knnClassifier.create();
  isPred=false;
  cancelAnimationFrame(predRaf);
  document.getElementById('predBtn').disabled=true;
  document.getElementById('predBtn').textContent='🔍 Start Predicting';
  document.getElementById('cpLabel').textContent='--';
  document.getElementById('cpConf').textContent='--';
  document.getElementById('confBars').innerHTML='';
  document.getElementById('analysisText').textContent='Add categories and start predicting to see analysis.';

  // Restore dropdown
  const sel=document.getElementById('newCatSelect');
  sel.innerHTML=`
    <option value="3">🍇 Grapes</option>
    <option value="4">🍓 Strawberry</option>
    <option value="5">🥭 Mango</option>`;

  renderActiveClasses();
  renderTimeline();
  log('Reset complete.','inf');
}

window.addEventListener('load',init);