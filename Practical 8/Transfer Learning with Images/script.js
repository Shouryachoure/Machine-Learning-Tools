let featureModel;
let classifier;
let classes = [];
let trainingData = [];

const log = (msg) => {
    const box = document.getElementById("log");
    box.innerHTML += msg + "<br>";
    box.scrollTop = box.scrollHeight;
};

async function loadMobileNet() {
    log("Loading MobileNet (1024-D)...");

    featureModel = await mobilenet.load({
        version: 1,
        alpha: 1.0
    });

    log("MobileNet Loaded ✓ (1024-D embeddings)");
}

loadMobileNet();

document.getElementById("files").addEventListener("change", async (evt) => {
    const files = evt.target.files;

    for (const f of files) {
        const folder = f.webkitRelativePath.split("/")[1];

        if (!classes.includes(folder)) classes.push(folder);

        const img = new Image();
        img.src = URL.createObjectURL(f);
        await new Promise(res => img.onload = res);

        trainingData.push({ img, label: folder });

        document.getElementById("preview").innerHTML += 
            `<img src="${img.src}" />`;
    }

    log(`Loaded ${trainingData.length} images`);
    log(`Classes: ${classes.join(", ")}`);
});

async function trainModel() {
    if (trainingData.length === 0) {
        alert("Upload files first");
        return;
    }

    log("Extracting embeddings...");

    const xs = [];
    const ys = [];

    for (let item of trainingData) {
        const emb = featureModel.infer(item.img, true);
        xs.push(emb);

        const oneHot = tf.oneHot(classes.indexOf(item.label), classes.length)
                       .reshape([1, classes.length]);
        ys.push(oneHot);

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

    log("Training classifier…");

    await classifier.fit(X, Y, {
        epochs: 10,
        batchSize: 8,
        callbacks: {
            onEpochEnd: (epoch, logs) =>
                log(`Epoch ${epoch + 1}: Accuracy = ${(logs.acc * 100).toFixed(2)}%`)
        }
    });

    log("Training Completed ✓");
}

async function testRandom() {
    if (!classifier) {
        alert("Train the model first");
        return;
    }

    const sample = trainingData[Math.floor(Math.random() * trainingData.length)];
    const emb = featureModel.infer(sample.img, true);
    const pred = classifier.predict(emb).dataSync();

    const idx = pred.indexOf(Math.max(...pred));

    log(`Prediction → ${classes[idx]} (${(pred[idx] * 100).toFixed(2)}%)`);
}