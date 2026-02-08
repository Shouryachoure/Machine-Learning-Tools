let model;

async function loadModel() {
    model = await mobilenet.load();
    document.getElementById("status").innerHTML = "Model Loaded ✔";
    startWebcam();
}

function startWebcam() {
    const video = document.getElementById("cam");

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            video.onloadedmetadata = () => {
                video.play();
                detect();
            };
        });
}

async function detect() {
    const video = document.getElementById("cam");

    const predictions = await model.classify(video);

    document.getElementById("label").innerHTML =
        `${predictions[0].className} (${(predictions[0].probability * 100).toFixed(2)}%)`;

    setTimeout(detect, 800); // Slow + stable
}

loadModel();
