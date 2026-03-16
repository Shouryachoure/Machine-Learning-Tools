let model=null, stream=null, isRunning=false, rafId=null;
let lastInferTime=0, lastPredictions=[];
const INFER_INTERVAL = 350; // stable inference rate

const video  = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');

let fpsSamples=[], infSamples=[];
let totalFrames=0, dropFrames=0, lastTime=0;
const DROP_THRESHOLD=15;

const chartOpts=(label,color)=>({
  type:'line',
  data:{ labels:[], datasets:[{ label, data:[], borderColor:color,
    backgroundColor:color+'20', borderWidth:2, pointRadius:0, tension:0.4, fill:true }] },
  options:{
    animation:false,
    plugins:{ legend:{ display:false } },
    scales:{
      x:{ ticks:{color:'#8a5555',font:{family:'Space Mono',size:9}}, grid:{color:'#3d1515'} },
      y:{ ticks:{color:'#8a5555',font:{family:'Space Mono',size:9}}, grid:{color:'#3d1515'} }
    }
  }
});

const fpsChart = new Chart(document.getElementById('fpsChart'), chartOpts('FPS','#ff2d2d'));
const infChart = new Chart(document.getElementById('infChart'), chartOpts('Inference ms','#ff6b35'));
const MAX_POINTS=60;

function addLog(msg,type=''){
  const box=document.getElementById('log');
  const d=document.createElement('div');
  d.className=`log-line ${type}`;
  d.textContent=`[${new Date().toLocaleTimeString()}] ${msg}`;
  box.appendChild(d); box.scrollTop=box.scrollHeight;
}

async function startSession(){
  try {
    stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}});
    video.srcObject=stream;
    await new Promise(r=>video.onloadedmetadata=r);
    canvas.width=video.videoWidth; canvas.height=video.videoHeight;

    if(!model){
      addLog('Loading MobileNet v2...','inf');
      model=await mobilenet.load({version:2,alpha:1.0});
      addLog('Model ready.','ok');
    }

    isRunning=true; lastTime=performance.now();
    document.getElementById('stopBtn').disabled=false;
    document.getElementById('liveIndicator').textContent='● LIVE';
    document.getElementById('liveIndicator').classList.add('on');
    addLog('Session started.','ok');
    loop();
  } catch(e){ addLog(`Error: ${e.message}`,'err'); }
}

async function loop(){
  if(!isRunning) return;

  const now=performance.now();
  const elapsed=now-lastTime; lastTime=now;
  const fps=Math.round(1000/elapsed);

  totalFrames++;
  fpsSamples.push(fps);
  if(fps<DROP_THRESHOLD) dropFrames++;

  if(now-lastInferTime>=INFER_INTERVAL){
    lastInferTime=now;
    const t0=performance.now();
    try {
      const raw=await model.classify(video,3);
      const filtered=raw.filter(p=>p.probability>0.08);
      if(filtered.length) lastPredictions=filtered;
    } catch(e){}
    const inf=Math.round(performance.now()-t0);
    infSamples.push(inf);

    const avgFps=avg(fpsSamples), avgInf=avg(infSamples);
    const minFps=Math.min(...fpsSamples), maxFps=Math.max(...fpsSamples);

    set('m_fps',fps); set('m_avg',avgFps.toFixed(1));
    set('m_inf',inf);  set('m_minf',avgInf.toFixed(1));
    set('m_min',minFps); set('m_max',maxFps);
    set('m_drop',dropFrames); set('m_total',totalFrames);

    pushChart(fpsChart, totalFrames.toString(), fps);
    pushChart(infChart, totalFrames.toString(), inf);

    if(lastPredictions.length){
      document.getElementById('topPred').textContent=lastPredictions[0].className.split(',')[0];
      document.getElementById('topConf').textContent=`${(lastPredictions[0].probability*100).toFixed(1)}% confidence`;
    }
  }

  drawHUD(fps, lastPredictions[0]);
  rafId=requestAnimationFrame(loop);
}

function drawHUD(fps, top){
  ctx.clearRect(0,0,canvas.width,canvas.height);

  ctx.save();
  ctx.scale(-1,1); ctx.translate(-canvas.width,0);

  const fpsColor=fps>=20?'#ff2d2d':fps>=10?'#ffaa00':'#ff0000';

  ctx.fillStyle='rgba(0,0,0,0.65)';
  ctx.fillRect(12,12,190,58);
  ctx.strokeStyle=fpsColor; ctx.lineWidth=1.5;
  ctx.strokeRect(12,12,190,58);

  ctx.fillStyle=fpsColor;
  ctx.font='bold 22px Space Mono, monospace';
  ctx.fillText(`${fps} FPS`,22,38);
  ctx.fillStyle='#ff6b35';
  ctx.font='12px Space Mono, monospace';
  ctx.fillText(`Frames: ${totalFrames}`,22,58);

  if(top){
    const name=top.className.split(',')[0];
    const conf=(top.probability*100).toFixed(1);
    ctx.fillStyle='rgba(0,0,0,0.65)';
    ctx.fillRect(12,canvas.height-52,280,40);
    ctx.strokeStyle='#ff2d2d'; ctx.lineWidth=1;
    ctx.strokeRect(12,canvas.height-52,280,40);
    ctx.fillStyle='#ff2d2d';
    ctx.font='bold 14px Space Mono, monospace';
    ctx.fillText(`${name}  ${conf}%`,22,canvas.height-28);
  }

  ctx.restore();
}

function pushChart(chart,label,value){
  chart.data.labels.push(label);
  chart.data.datasets[0].data.push(value);
  if(chart.data.labels.length>MAX_POINTS){
    chart.data.labels.shift(); chart.data.datasets[0].data.shift();
  }
  chart.update('none');
}

function stopSession(){
  isRunning=false;
  if(rafId) cancelAnimationFrame(rafId);
  if(stream) stream.getTracks().forEach(t=>t.stop());
  ctx.clearRect(0,0,canvas.width,canvas.height);
  document.getElementById('stopBtn').disabled=true;
  document.getElementById('liveIndicator').textContent='● IDLE';
  document.getElementById('liveIndicator').classList.remove('on');
  addLog('Session ended.','inf');
  generateReport();
}

function generateReport(){
  if(!fpsSamples.length) return;
  const avgFps=avg(fpsSamples).toFixed(2), avgInf=avg(infSamples).toFixed(2);
  const minFps=Math.min(...fpsSamples), maxFps=Math.max(...fpsSamples);
  const dropPct=((dropFrames/totalFrames)*100).toFixed(1);

  const rows=[
    ['Total Frames',totalFrames],['Avg FPS',avgFps],['Min FPS',minFps],['Max FPS',maxFps],
    ['Avg Inference',`${avgInf} ms`],['Min Inference',`${Math.min(...infSamples)} ms`],
    ['Max Inference',`${Math.max(...infSamples)} ms`],['Frame Drops',`${dropFrames} (${dropPct}%)`]
  ];

  let note = avgFps>=20
    ? `✅ Good performance. ${avgFps} FPS — smooth real-time inference.`
    : avgFps>=10
    ? `⚠ Moderate (${avgFps} FPS). Consider reducing resolution.`
    : `❌ Poor performance (${avgFps} FPS). Use lighter model or lower resolution.`;

  document.getElementById('reportContent').innerHTML =
    rows.map(([l,v])=>`<div class="report-row"><span class="r-label">${l}</span><span class="r-val">${v}</span></div>`).join('')
    + `<div class="report-note">${note}</div>`;
}

function resetData(){
  fpsSamples=[]; infSamples=[];
  totalFrames=0; dropFrames=0; lastPredictions=[];
  fpsChart.data.labels=[]; fpsChart.data.datasets[0].data=[]; fpsChart.update();
  infChart.data.labels=[]; infChart.data.datasets[0].data=[]; infChart.update();
  ['m_fps','m_avg','m_inf','m_minf','m_min','m_max','m_drop'].forEach(id=>set(id,'--'));
  set('m_total',0);
  document.getElementById('topPred').textContent='--';
  document.getElementById('topConf').textContent='--';
  document.getElementById('reportContent').innerHTML='<p class="hint">Run a session to generate a report.</p>';
  addLog('Data reset.','inf');
}

function set(id,val){ document.getElementById(id).textContent=val; }
function avg(arr){ return arr.length?arr.reduce((a,b)=>a+b,0)/arr.length:0; }