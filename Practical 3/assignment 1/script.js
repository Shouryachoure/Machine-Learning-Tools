// Small dataset
const data = [
  { t: "I love it", y: 1 },
  { t: "This is great", y: 1 },
  { t: "Amazing experience", y: 1 },
  { t: "I hate this", y: 0 },
  { t: "This is bad", y: 0 },
  { t: "Worst feeling ever", y: 0 }
];

let wordIndex = { "<PAD>": 0, "<UNK>": 1 };
let idx = 2;

data.forEach(d => {
  d.t.toLowerCase().split(" ").forEach(w => {
    if (!wordIndex[w]) wordIndex[w] = idx++;
  });
});

function tok(t) {
  return t.toLowerCase().split(" ").map(w => wordIndex[w] || 1);
}

function pad(arr, len = 6) {
  while (arr.length < len) arr.push(0);
  return arr.slice(0, len);
}

const xs = tf.tensor2d(data.map(d => pad(tok(d.t))), [data.length, 6]);
const ys = tf.tensor2d(data.map(d => [d.y]), [data.length, 1]);

const model = tf.sequential();
model.add(tf.layers.embedding({ inputDim: idx, outputDim: 8, inputLength: 6 }));
model.add(tf.layers.simpleRNN({ units: 16 }));
model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

model.compile({ optimizer: "adam", loss: "binaryCrossentropy", metrics: ["accuracy"] });

(async () => {
  document.getElementById("status").innerText = "Status: Training...";
  const history = await model.fit(xs, ys, { epochs: 20 });
  document.getElementById("status").innerText =
    "Training Complete! Accuracy: " + (history.history.acc.pop() * 100).toFixed(2) + "%";
})();
