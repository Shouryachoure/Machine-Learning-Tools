let net;
const video = document.getElementById("webcam");
const canvas = document.getElementById("output");
const ctx = canvas.getContext("2d");

async function loadPoseNet() {
    document.getElementById("status").innerText = "Loading PoseNet…";
    net = await posenet.load();
    document.getElementById("status").innerText = "PoseNet Loaded ✔";
    startCamera();
}

async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    video.onloadedmetadata = () => detectPose();
}

async function detectPose() {
    const pose = await net.estimateSinglePose(video);

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    pose.keypoints.forEach(k => {
        if (k.score > 0.5) {
            ctx.beginPath();
            ctx.arc(k.position.x, k.position.y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = "#00ffcc";
            ctx.fill();
        }
    });

    requestAnimationFrame(detectPose);
}

loadPoseNet();