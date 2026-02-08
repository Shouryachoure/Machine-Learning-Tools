let net;
let webcam;

async function setupWebcam() {
  webcam = document.getElementById("webcam");

  const stream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: false
  });

  webcam.srcObject = stream;

  return new Promise(resolve => {
    webcam.onloadedmetadata = () => resolve();
  });
}

async function start() {
  document.getElementById("status").innerText = "Loading model...";
  net = await mobilenet.load();
  document.getElementById("status").innerText = "Model Loaded ✔";

  await setupWebcam();

  classifyFrame();
}

async function classifyFrame() {
  const predictions = await net.classify(webcam);

  if (predictions && predictions.length > 0) {
    const p = predictions[0];
    document.getElementById("label").innerHTML =
      `Detected: <b>${p.className}</b> (${(p.probability*100).toFixed(2)}%)`;
  }

  setTimeout(() => classifyFrame(), 600);
}

start();
