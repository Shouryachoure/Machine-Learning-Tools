let featureModel;
let classifier;
let trainingData = [];
let classes = [];

// Log utility
function log(msg) {
    const logBox = document.getElementById("log");
    logBox.innerHTML += msg + "<br>";
    logBox.scrollTop = logBox.scrollHeight;
}

// Load MobileNet (1024-D)
async function loadModel() {
    log("Loading MobileNet...");
    featureModel = await mobilenet.load({ version: 1, alpha: 1.0 });
    log("MobileNet Loaded ✔ (1024D)");
}
loadModel();

// Manual image upload
document.getElementById("fileInput").addEventListener("change", async (evt) => {
    const files = evt.target.files;

    for (const f of files) {
        const label = prompt(`Enter label for image: ${f.name}\n(Shinchan / Harry / Mitsy)`);

        if (!label) continue;
        if (!classes.includes(label)) classes.push(label);

        const img = new Image();
        img.src = URL.createObjectURL(f);

        await new Promise(res => img.onload = res);

        trainingData.push({ img, label });

        document.getElementById("preview").innerHTML += 
            `<img src="${img.src}">`;
    }

    log(`Loaded ${trainingData.length} images`);
    log("Classes: " + classes.join(", "));
});

// Train the classifier
async function trainModel() {
    if (trainingData.length === 0) {
        alert("Upload and label images first.");
        return;
    }

    log("Extracting embeddings...");

    const xs = [];
    const ys = [];

    for (const item of trainingData) {
        const emb = featureModel.infer(item.img, true);
        xs.push(emb);

        const oh = tf.oneHot(classes.indexOf(item.label), classes.length)
                     .reshape([1, classes.length]);
        ys.push(oh);

        await tf.nextFrame();
    }

    const X = tf.concat(xs);
    const Y = tf.concat(ys);

    classifier = tf.sequential({
        layers: [
            tf.layers.dense({ units: 64, activation: "relu", inputShape: [1024] }),
            tf.layers.dropout({ rate: 0.3 }),
            tf.layers.dense({ units: classes.length, activation: "softmax" })
        ]
    });

    classifier.compile({
        optimizer: "adam",
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    });

    log("Training...");

    await classifier.fit(X, Y, {
        epochs: 10,
        batchSize: 8,
        callbacks: {
            onEpochEnd: (epoch, logs) =>
                log(`Epoch ${epoch+1} → Acc: ${(logs.acc*100).toFixed(2)}%`)
        }
    });

    log("✔ Training Completed Successfully");
}

// Test a random image
async function testRandom() {
    if (!classifier) {
        alert("Train the model first.");
        return;
    }

    const sample = trainingData[Math.floor(Math.random() * trainingData.length)];
    const emb = featureModel.infer(sample.img, true);
    const pred = classifier.predict(emb).dataSync();

    const best = pred.indexOf(Math.max(...pred));

    log(`<b>Prediction:</b> ${classes[best]} (${(pred[best]*100).toFixed(2)}%)`);
}