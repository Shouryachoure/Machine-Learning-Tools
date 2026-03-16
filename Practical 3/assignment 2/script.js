const data = [
  { t: "I love this", y: 1 },
  { t: "Amazing feeling", y: 1 },
  { t: "This is great", y: 1 },
  { t: "I hate this", y: 0 },
  { t: "This is terrible", y: 0 },
  { t: "Worst experience", y: 0 }
];

let wordIndex = { "<PAD>": 0, "<UNK>": 1 };
let ix = 2;

data.forEach(d => {
  d.t.toLowerCase().split(" ").forEach(w => {
    if (!wordIndex[w]) wordIndex[w] = ix++;
  });
});

function tok(s) { return s.toLowerCase().split(" ").map(w => wordIndex[w] || 1); }
function pad(a, l = 6) { while (a.length < l) a.push(0); return a.slice(0, l); }

const xs = tf.tensor2d(data.map(d => pad(tok(d.t))), [data.length, 6]);
const ys = tf.tensor2d(data.map(d => [d.y]), [data.length, 1]);

const model = tf.sequential();
model.add(tf.layers.embedding({ inputDim: ix, outputDim: 8, inputLength: 6 }));
model.add(tf.layers.simpleRNN({ units: 20 }));
model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

model.compile({ optimizer: "adam", loss: "binaryCrossentropy" });

(async () => {
  console.log("Training...");
  await model.fit(xs, ys, { epochs: 25 });
  console.log("Done!");
})();

window.predictSentiment = function () {
  let text = document.getElementById("text").value;
  let seq = pad(tok(text));
  let input = tf.tensor2d([seq], [1, 6]);

  let score = model.predict(input).dataSync()[0];

  document.getElementById("output").innerText =
    score > 0.5 
      ? `😊 Positive (${(score * 100).toFixed(1)}% confidence)` 
      : `😡 Negative (${((1 - score) * 100).toFixed(1)}% confidence)`;
};
