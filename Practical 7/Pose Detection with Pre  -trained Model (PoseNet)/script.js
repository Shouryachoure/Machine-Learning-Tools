let net;
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;

        await new Promise(resolve => {
            video.onloadedmetadata = () => {
                video.play();

                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                resolve();
            };
        });

        console.log("Webcam ready:", video.videoWidth, video.videoHeight);

    } catch (err) {
        alert("Webcam permission denied!");
    }
}

async function loadModel() {
    net = await posenet.load();
    document.getElementById("status").innerHTML = "PoseNet Loaded ✔";
    detectPose();
}

async function detectPose() {
    if (video.videoWidth === 0) {
        console.log("Webcam not ready yet...");
        return requestAnimationFrame(detectPose);
    }

    const pose = await net.estimateSinglePose(video, { flipHorizontal: true });

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    drawKeypoints(pose.keypoints);
    drawSkeleton(pose.keypoints);

    requestAnimationFrame(detectPose);
}

function drawKeypoints(keypoints) {
    keypoints.forEach(kp => {
        if (kp.score > 0.5) {
            ctx.beginPath();
            ctx.arc(kp.position.x, kp.position.y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = "#00ffea";
            ctx.fill();
        }
    });
}

function drawSkeleton(keypoints) {
    const adjacent = posenet.getAdjacentKeyPoints(keypoints, 0.5);

    adjacent.forEach(pair => {
        ctx.beginPath();
        ctx.moveTo(pair[0].position.x, pair[0].position.y);
        ctx.lineTo(pair[1].position.x, pair[1].position.y);
        ctx.lineWidth = 3;
        ctx.strokeStyle = "#00ffea";
        ctx.stroke();
    });
}

startWebcam();
loadModel();