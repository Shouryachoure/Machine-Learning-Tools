const sentences = [
    "I love this", "Amazing experience", "This is wonderful", "Feeling good",
    "Best ever", "I hate this", "This is bad", "Terrible service",
    "I am sad", "Worst day ever"
];

const labels = [1,1,1,1,1, 0,0,0,0,0];

let tokenizer = new Map();
let index = 1;

function encode(text) {
    return text.toLowerCase().split(" ").map(w => {
        if (!tokenizer.has(w)) tokenizer.set(w, index++);
        return tokenizer.get(w);
    });
}

function pad(arr) {
    while (arr.length < 6) arr.push(0);
    return arr.slice(0, 6);
}

let model;

async function trainModel() {
    const xs = tf.tensor2d(sentences.map(s => pad(encode(s))));
    const ys = tf.tensor1d(labels);

    model = tf.sequential();
    model.add(tf.layers.dense({ units: 16, activation: "relu", inputShape: [6] }));
    model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

    model.compile({ optimizer: "adam", loss: "binaryCrossentropy", metrics: ["accuracy"] });

    await model.fit(xs, ys, { epochs: 40 });

    console.log("Dense Model Trained!");
}

trainModel();

async function analyze() {
    const text = document.getElementById("input").value.trim();
    if (!text) return;

    const arr = pad(encode(text));
    const prob = (await model.predict(tf.tensor2d([arr])).data())[0];

    document.getElementById("output").innerHTML =
        `Sentiment: ${prob > 0.5 ? "Positive 😊" : "Negative 😡"}<br>Confidence: ${(prob*100).toFixed(2)}%`;
}