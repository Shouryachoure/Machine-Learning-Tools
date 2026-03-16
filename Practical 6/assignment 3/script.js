const data = [
    "I love this", "This is amazing", "What a great experience", "Happy right now",
    "Totally loved it", "I hate this", "Very bad", "Worst feeling ever",
    "I am upset", "This is terrible"
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

function pad(x) {
    while (x.length < 6) x.push(0);
    return x.slice(0, 6);
}

let model;

async function train() {
    const xs = tf.tensor2d(data.map(s => pad(encode(s))));
    const ys = tf.tensor1d(labels);

    model = tf.sequential();
    model.add(tf.layers.embedding({ inputDim: 300, outputDim: 16, inputLength: 6 }));
    model.add(tf.layers.lstm({ units: 20 }));
    model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

    model.compile({ optimizer: "adam", loss: "binaryCrossentropy", metrics: ["accuracy"] });

    await model.fit(xs, ys, { epochs: 35 });
    console.log("LSTM model trained!");
}

train();

async function analyze() {
    const txt = document.getElementById("inputText").value.trim();
    if (!txt) return;

    const arr = pad(encode(txt));
    const prob = (await model.predict(tf.tensor2d([arr])).data())[0];

    document.getElementById("result").innerHTML =
        `Sentiment: ${prob > 0.5 ? "Positive 😊" : "Negative 😡"}<br>Confidence: ${(prob*100).toFixed(2)}%`;
}