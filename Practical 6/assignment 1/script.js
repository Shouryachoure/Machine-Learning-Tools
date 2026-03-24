let video = document.getElementById("video");
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

let net, running = false;

async function loadModel() {
  net = await posenet.load();
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;

  video.onloadedmetadata = () => {
    video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    running = true;
    detect();
  };
}

function stopCamera() {
  running = false;
}

async function detect() {
  if (!running) return;

  const pose = await net.estimateSinglePose(video);

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  pose.keypoints.forEach(p => {
    if (p.score > 0.5) {
      ctx.beginPath();
      ctx.arc(p.position.x, p.position.y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "red";
      ctx.fill();
    }
  });

  const pairs = posenet.getAdjacentKeyPoints(pose.keypoints, 0.5);

  pairs.forEach(pair => {
    ctx.beginPath();
    ctx.moveTo(pair[0].position.x, pair[0].position.y);
    ctx.lineTo(pair[1].position.x, pair[1].position.y);
    ctx.strokeStyle = "lime";
    ctx.stroke();
  });

  requestAnimationFrame(detect);
}

loadModel();