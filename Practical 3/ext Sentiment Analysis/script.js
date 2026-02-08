// ----------------------
// SMALL TRAINING DATASET
// ----------------------
const dataset = [
  { text: "I love this so much", label: 1 },
  { text: "This made me very happy", label: 1 },
  { text: "I am feeling awesome today", label: 1 },
  { text: "This is the best day ever", label: 1 },

  { text: "I hate this so much", label: 0 },
  { text: "This is terrible", label: 0 },
  { text: "I feel really bad", label: 0 },
  { text: "This ruined my mood", label: 0 }
];

// ----------------------
// BUILD VOCAB
// ----------------------
let wordIndex = { "<PAD>": 0, "<UNK>": 1 };
let idx = 2;

dataset.forEach(x => {
  x.text.toLowerCase().split(" ").forEach(word => {
    if (!wordIndex[word]) wordIndex[word] = idx++;
  });
});

// tokenize
function tokenize(text) {
  return text.toLowerCase().split(" ").map(w => wordIndex[w] || 1);
}

// Pad to len=6
function pad(seq, len = 6) {
  while (seq.length < len) seq.push(0);
  return seq.slice(0, len);
}

// ----------------------
// MODEL TRAINING
// ----------------------
const xs = tf.tensor2d(
  dataset.map(d => pad(tokenize(d.text))), 
  [dataset.length, 6]
);

const ys = tf.tensor2d(
  dataset.map(d => [d.label]), 
  [dataset.length, 1]
);

const model = tf.sequential();
model.add(tf.layers.embedding({ inputDim: idx, outputDim: 8, inputLength: 6 }));
model.add(tf.layers.simpleRNN({ units: 16, activation: "tanh" }));
model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

model.compile({
  optimizer: "adam",
  loss: "binaryCrossentropy",
  metrics: ["accuracy"]
});

// train on load
(async () => {
  console.log("Training...");
  await model.fit(xs, ys, { epochs: 30 });
  console.log("Training finished!");
})();

// ----------------------
// PREDICT FUNCTION
// ----------------------
window.predictSentiment = async () => {
  const inputText = document.getElementById("inputText").value;

  if (inputText.trim() === "") return;

  const seq = pad(tokenize(inputText));
  const inp = tf.tensor2d([seq], [1, 6]);

  const prediction = model.predict(inp);
  const score = prediction.dataSync()[0];

  const sentimentText = document.getElementById("sentimentText");
  const fill = document.getElementById("confidenceFill");

  if (score > 0.5) {
    sentimentText.innerHTML = `😊 Positive`;
    fill.style.background = "#00ff9d";
    fill.style.width = (score * 100) + "%";
  } else {
    sentimentText.innerHTML = `😡 Negative`;
    fill.style.background = "#ff5e5e";
    fill.style.width = ((1 - score) * 100) + "%";
  }
};
