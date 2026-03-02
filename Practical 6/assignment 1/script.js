// Small training dataset
const sentences = [
    "I love this product", "This is amazing", "I feel great today",
    "What a wonderful experience", "Absolutely fantastic work",
    "I hate this", "This is terrible", "Worst day ever",
    "I am sad", "This makes me angry"
];

const labels = [1,1,1,1,1, 0,0,0,0,0]; // 1 = Positive, 0 = Negative

let model;

async function trainModel() {
    // Text vectorizer (Naive)
    const tokenizer = new Map();
    let index = 1;

    function encode(text) {
        return text.toLowerCase().split(" ").map(w => {
            if (!tokenizer.has(w)) tokenizer.set(w, index++);
            return tokenizer.get(w);
        });
    }

    const xs = tf.tensor2d(sentences.map(t => {
        const arr = encode(t);
        while (arr.length < 6) arr.push(0);
        return arr.slice(0, 6);
    }));

    const ys = tf.tensor1d(labels);

    // Dense model
    model = tf.sequential();
    model.add(tf.layers.embedding({ inputDim: 200, outputDim: 16, inputLength: 6 }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 8, activation: "relu" }));
    model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

    model.compile({ optimizer: "adam", loss: "binaryCrossentropy", metrics: ["accuracy"] });

    await model.fit(xs, ys, { epochs: 20 });

    console.log("Model trained!");
    return encode;
}

let encoder;

(async () => {
    encoder = await trainModel();
})();

async function analyze() {
    const text = document.getElementById("inputText").value.trim();
    if (!text) return;

    const arr = encoder(text);
    while (arr.length < 6) arr.push(0);

    const inputTensor = tf.tensor2d([arr]);

    const prediction = model.predict(inputTensor);
    const prob = (await prediction.data())[0];

    const sentiment = prob > 0.5 ? "Positive 😊" : "Negative 😡";

    document.getElementById("output").innerHTML =
        `${sentiment}<br><span style="font-size:18px;">Confidence: ${(prob*100).toFixed(2)}%</span>`;
}