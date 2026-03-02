let featureModel;
let classifier;

let dataset = [];
let classes = [];

// Logging
function log(msg) {
    const l = document.getElementById("log");
    l.innerHTML += msg + "<br>";
    l.scrollTop = l.scrollHeight;
}

// Load MobileNet V2 (CORS SAFE)
async function loadMobileNet() {
    log("Loading MobileNet V2...");
    featureModel = await mobilenet.load({ version: 2, alpha: 1.0 });
    log("MobileNet V2 Loaded ✔");
}
loadMobileNet();

// Manual upload
document.getElementById("fileInput").addEventListener("change", async (evt) => {
    const files = evt.target.files;
    
    for (let f of files) {
        const label = prompt(`Enter label for ${f.name} (Shinchan / Harry / Mitsy)`);

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

// Shuffle
function shuffle(arr) {
    return arr.sort(() => Math.random() - 0.5);
}

// Split dataset
function split(data, ratio = 0.8) {
    let s = shuffle([...data]);
    let t = Math.floor(s.length * ratio);
    return { train: s.slice(0, t), test: s.slice(t) };
}

// Train + Validate
async function trainAndValidate() {
    if (!dataset.length) {
        alert("Upload images first!");
        return;
    }

    const { train, test } = split(dataset);

    log(`Training samples: ${train.length}`);
    log(`Validation samples: ${test.length}`);

    let xs = [], ys = [];

    for (let item of train) {
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
            tf.layers.dense({ units: 64, activation: "relu", inputShape: [1280] }),
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
            onEpochEnd: (e, logs) =>
                log(`Epoch ${e + 1} → Accuracy: ${(logs.acc * 100).toFixed(2)}%`)
        }
    });

    log("Training completed ✔");
    log("Evaluating model...");

    evaluateModel(test);
}

// Evaluate
async function evaluateModel(testSet) {
    let correct = 0;

    // Confusion Matrix NxN
    let cm = Array(classes.length)
            .fill(0)
            .map(() => Array(classes.length).fill(0));

    for (let item of testSet) {
        const emb = featureModel.infer(item.img, true);
        const pred = classifier.predict(emb).dataSync();

        const predictedIndex = pred.indexOf(Math.max(...pred));
        const actualIndex = classes.indexOf(item.label);

        if (predictedIndex === actualIndex) correct++;

        cm[actualIndex][predictedIndex]++;
    }

    let acc = (correct / testSet.length) * 100;
    log(`Validation Accuracy: ${acc.toFixed(2)}%`);

    renderCM(cm);
}

// Render Confusion Matrix
function renderCM(cm) {
    let html = `<table><tr><th></th>`;
    for (let c of classes) html += `<th>${c}</th>`;
    html += `</tr>`;

    cm.forEach((row, i) => {
        html += `<tr><th>${classes[i]}</th>`;
        row.forEach(v => html += `<td class='cmCell'>${v}</td>`);
        html += `</tr>`;
    });

    html += "</table>";

    document.getElementById("cm").innerHTML = html;
}