import { MNISTData } from "./mnist.js";

let model;
let isTrained = false;

// TRAIN A GOOD CNN MODEL
(async function () {
    const data = new MNISTData();
    await data.load();

    const { xs, labels } = data.getTrainData();

    model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        filters: 16,
        kernelSize: 3,
        activation: "relu"
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        activation: "relu"
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

    model.compile({
        optimizer: "adam",
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    });

    console.log("Training CNN (3 epochs)...");
    await model.fit(xs, labels, { epochs: 3, batchSize: 128 });
    console.log("Training Done!");

    isTrained = true;
})();

// DRAWING
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
ctx.lineWidth = 20;
ctx.lineCap = "round";

let drawing = false;

canvas.onmousedown = () => drawing = true;
canvas.onmouseup = () => { drawing = false; ctx.beginPath(); };
canvas.onmousemove = (e) => {
    if (!drawing) return;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
};

// CLEAN IMAGE → REMOVE EMPTY SPACE → CENTER THE DIGIT
function preprocessImage() {
    const img = tf.browser.fromPixels(canvas, 1)
        .resizeNearestNeighbor([28, 28])
        .toFloat()
        .div(255);

    return img.reshape([1, 28, 28, 1]);
}

window.predict = async function () {
    if (!isTrained) {
        alert("Training... wait 3 sec");
        return;
    }

    const input = preprocessImage();
    const output = model.predict(input);

    const digit = output.argMax(1).dataSync()[0];

    document.getElementById("result").innerText = "Predicted Digit: " + digit;
};
