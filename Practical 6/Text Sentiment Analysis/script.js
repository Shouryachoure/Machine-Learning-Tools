// Small training dataset
const trainingData = [
    { text: "I love this product", label: 1 },
    { text: "This is amazing", label: 1 },
    { text: "I feel great today", label: 1 },
    { text: "This is the worst", label: 0 },
    { text: "I hate this", label: 0 },
    { text: "This is terrible", label: 0 }
];

// Simple tokenizer
function tokenize(text) {
    return text.toLowerCase().replace(/[^a-z\s]/g, "").split(" ");
}

const vocab = {};
let index = 1;

trainingData.forEach(item => {
    tokenize(item.text).forEach(word => {
        if (!vocab[word]) vocab[word] = index++;
    });
});

// Vectorize
function vectorize(text) {
    const words = tokenize(text);
    const vec = new Array(Object.keys(vocab).length).fill(0);
    words.forEach(w => {
        if (vocab[w]) vec[vocab[w] - 1] = 1;
    });
    return vec;
}

// Dense model (simple)
const denseModel = tf.sequential();
denseModel.add(tf.layers.dense({ units: 8, activation: "relu", inputShape: [Object.keys(vocab).length] }));
denseModel.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

denseModel.compile({ optimizer: "adam", loss: "binaryCrossentropy", metrics: ["accuracy"] });

// Prepare training data
const xs = tf.tensor2d(trainingData.map(d => vectorize(d.text)));
const ys = tf.tensor2d(trainingData.map(d => [d.label]));

// Train
(async () => {
    await denseModel.fit(xs, ys, { epochs: 20 });
    console.log("Dense Model Trained.");
})();

// Predict
window.analyze = async function () {
    const text = document.getElementById("inputText").value.trim();
    if (!text) {
        alert("Enter text first!");
        return;
    }

    const input = tf.tensor2d([vectorize(text)]);
    const prediction = denseModel.predict(input);
    const score = (await prediction.data())[0];

    let sentiment = score > 0.5 ? "Positive 😊" : "Negative 😡";

    document.getElementById("result").innerHTML = `
        Sentiment: <b>${sentiment}</b><br>
        Confidence: ${(score * 100).toFixed(2)}%
    `;
};