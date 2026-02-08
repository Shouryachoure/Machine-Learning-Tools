let model;

// Smooth output buffer
let lastPrediction = "";
let lastConfidence = 0;
let frameCount = 0;

async function loadModel() {
    model = await mobilenet.load();
    document.getElementById("status").innerHTML = "Model Loaded ✔";
    startWebcam();
}

function startWebcam() {
    const video = document.getElementById("cam");

    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream;
            video.onloadedmetadata = () => {
                video.play();
                runDetection();
            };
        })
        .catch((err) => {
            alert("Camera Access Error: " + err);
        });
}

async function runDetection() {
    const video = document.getElementById("cam");

    setInterval(async () => {
        frameCount++;

        // Skip 2 out of every 3 frames (FOR SUPER STABILITY)
        if (frameCount % 3 !== 0) return;

        const predictions = await model.classify(video);

        // Extract top-1
        const top = predictions[0];

        // EXPONENTIAL SMOOTHING —— stabilizes prediction  
        const smoothFactor = 0.65;  
        lastConfidence = lastConfidence * smoothFactor + top.probability * (1 - smoothFactor);

        // Update prediction ONLY if label changed significantly
        if (top.className !== lastPrediction) {
            lastPrediction = top.className;
            document.getElementById("label").innerHTML =
                `<b>Detected:</b> ${top.className} (${(lastConfidence * 100).toFixed(2)}%)`;
        }

    }, 300); // 300ms = ultra stable, smooth
}

loadModel();
