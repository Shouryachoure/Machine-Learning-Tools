let net;
let mode = "single";

document.getElementById("modeText").innerText = "Single Pose";

async function loadPoseNet() {
    document.getElementById("status").innerHTML = "Loading PoseNet…";
    net = await posenet.load();
    document.getElementById("status").innerHTML = "PoseNet Loaded ✔";
    startCamera();
}

async function startCamera() {
    const video = document.getElementById("webcam");
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });

    video.onloadedmetadata = () => detect();
}

async function detect() {
    const video = document.getElementById("webcam");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    let poses;

    if (mode === "single") {
        poses = [await net.estimateSinglePose(video)];
    } else {
        poses = await net.estimateMultiplePoses(video);
    }

    poses.forEach(pose => {
        pose.keypoints.forEach(k => {
            if (k.score > 0.5) {
                ctx.beginPath();
                ctx.arc(k.position.x, k.position.y, 4, 0, 2 * Math.PI);
                ctx.fillStyle = mode === "single" ? "cyan" : "yellow";
                ctx.fill();
            }
        });
    });

    requestAnimationFrame(detect);
}

loadPoseNet();