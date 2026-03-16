let net;
let count = 0;
let isDown = false;

const video = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

function angle(A, B, C) {
    const AB = Math.hypot(B.x - A.x, B.y - A.y);
    const BC = Math.hypot(B.x - C.x, B.y - C.y);
    const AC = Math.hypot(C.x - A.x, C.y - A.y);
    return Math.acos((AB * AB + BC * BC - AC * AC) / (2 * AB * BC)) * (180 / Math.PI);
}

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

    const leftHip = pose.keypoints[11].position;
    const leftKnee = pose.keypoints[13].position;
    const leftAnkle = pose.keypoints[15].position;

    const kneeAngle = angle(leftHip, leftKnee, leftAnkle);

    if (kneeAngle < 70) {
        isDown = true;
    }
    if (kneeAngle > 150 && isDown) {
        count++;
        isDown = false;
    }

    document.getElementById("count").innerText = "Squats: " + count;

    requestAnimationFrame(detectPose);
}

loadPoseNet();