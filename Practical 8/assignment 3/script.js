let featureModel;
let classifier;

let dataset = [];
let classes = [];
let baselineAccuracy = 0;

// Log helper
function log(msg) {
    const box = document.getElementById("log");
    box.innerHTML += msg + "<br>";
    box.scrollTop = box.scrollHeight;
}

// Load MobileNet V1 (100% safe)
async function initMobileNet() {
    log("Loading MobileNet V1 (Safe URL)...");
    featureModel = await mobilenet.load();
    log("MobileNet V1 Loaded ✔ (1024-D Embeddings)");
}
initMobileNet();

// Upload images manually
document.getElementById("fileInput").addEventListener("change", async (evt) => {
    const files = evt.target.files;

    for (let f of files) {
        const label = prompt(`Enter label for: ${f.name}`);

        if (!label) continue;
        if (!classes.includes(label)) classes.push(label);

        const img = new Image();
        img.src = URL.createObjectURL(f);
        await new Promise(res => img.onload = res);

        dataset.push({ img, label });

        document.getElementById("preview").innerHTML += 
            `<img src="${img.src}">`;
    }

    log(`Total images: ${dataset.length}`);
    log(`Classes: ${classes.join(", ")}`);
});

// Shuffle utility
function shuffle(a) {
    return a.sort(() => Math.random() - 0.5);
}

// 80/20 train-test split
function split(data) {
    let s = shuffle([...data]);
    let t = Math.floor(s.length * 0.8);
    return { train: s.slice(0,t), test: s.slice(t) };
}

// -------- INITIAL TRAINING --------
async function trainModel() {
    if (!dataset.length) return alert("Upload images first!");

    const { train, test } = split(dataset);

    let xs = [], ys = [];

    log("Extracting embeddings...");

    for (let item of train) {
        const emb = featureModel.infer(item.img, true); // 1024-D
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

    log("Training started...");

    await classifier.fit(X, Y, {
        epochs: 10,
        batchSize: 8,
        callbacks: {
            onEpochEnd: (ep, logs) =>
                log(`Epoch ${ep + 1}: ${(logs.acc * 100).toFixed(2)}% accuracy`)
        }
    });

    baselineAccuracy = evaluate(test);
    log(`✔ Baseline Accuracy: ${baselineAccuracy.toFixed(2)}%`);
}

// -------- ADD NEW CATEGORY ----------
function addMore() {
    document.getElementById("fileInput").click();
}

// -------- RETRAIN AFTER ADDING NEW CATEGORY ----------
async function retrainModel() {
    if (!dataset.length) return alert("Upload initial dataset!");

    log("Retraining with added category...");

    const { train, test } = split(dataset);

    let xs = [], ys = [];

    for (let item of train) {
        const emb = featureModel.infer(item.img, true);
        xs.push(emb);

        const oh = tf.oneHot(classes.indexOf(item.label), classes.length)
                    .reshape([1, classes.length]);
        ys.push(oh);
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

    await classifier.fit(X, Y, { epochs: 10 });

    let newAcc = evaluate(test);
    log(`🔥 New Accuracy After Retraining: ${newAcc.toFixed(2)}%`);

    let diff = newAcc - baselineAccuracy;
    log(`📊 Accuracy Change: ${diff.toFixed(2)}%`);
}

// -------- EVALUATION & CONFUSION MATRIX --------
function evaluate(testSet) {
    let correct = 0;

    let cm = Array(classes.length)
        .fill(0)
        .map(() => Array(classes.length).fill(0));

    for (let item of testSet) {
        const emb = featureModel.infer(item.img, true);
        const pred = classifier.predict(emb).dataSync();

        const predicted = pred.indexOf(Math.max(...pred));
        const actual = classes.indexOf(item.label);

        if (predicted === actual) correct++;

        cm[actual][predicted]++;
    }

    renderCM(cm);

    return (correct / testSet.length) * 100;
}

function renderCM(cm) {
    let html = `<table><tr><th></th>`;
    for (let c of classes) html += `<th>${c}</th>`;
    html += `</tr>`;

    cm.forEach((row, i) => {
        html += `<tr><th>${classes[i]}</th>`;
        row.forEach(v => html += `<td class="cmCell">${v}</td>`);
        html += `</tr>`;
    });

    html += "</table>";

    document.getElementById("cm").innerHTML = html;
}