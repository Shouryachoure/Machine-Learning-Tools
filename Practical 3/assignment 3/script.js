const data = [
  { t: "I love this", y: 1 },
  { t: "Amazing", y: 1 },
  { t: "Very good", y: 1 },
  { t: "I hate this", y: 0 },
  { t: "Very bad", y: 0 },
  { t: "Terrible", y: 0 }
];

let wordIndex = { "<PAD>":0, "<UNK>":1 }; 
let ix=2;

data.forEach(d => {
  d.t.toLowerCase().split(" ").forEach(w => { if(!wordIndex[w]) wordIndex[w]=ix++; });
});

function tok(t){return t.toLowerCase().split(" ").map(w=>wordIndex[w]||1);}
function pad(a,l=5){while(a.length<l)a.push(0);return a.slice(0,l);}

const xs = tf.tensor2d(data.map(d=>pad(tok(d.t))),[data.length,5]);
const ys = tf.tensor2d(data.map(d=>[d.y]),[data.length,1]);

// Dense model
const dense = tf.sequential();
dense.add(tf.layers.dense({inputShape:[5], units:16, activation:"relu"}));
dense.add(tf.layers.dense({units:1, activation:"sigmoid"}));
dense.compile({optimizer:"adam", loss:"binaryCrossentropy", metrics:["accuracy"]});

// RNN model
const rnn = tf.sequential();
rnn.add(tf.layers.embedding({inputDim: ix, outputDim:8, inputLength:5}));
rnn.add(tf.layers.simpleRNN({units:20}));
rnn.add(tf.layers.dense({units:1, activation:"sigmoid"}));
rnn.compile({optimizer:"adam", loss:"binaryCrossentropy", metrics:["accuracy"]});

(async ()=>{
  const r=await dense.fit(xs,ys,{epochs:20});
  document.getElementById("denseResult").innerText =
    "Dense Accuracy: " + (r.history.acc.pop()*100).toFixed(2)+"%";

  const r2=await rnn.fit(xs,ys,{epochs:20});
  document.getElementById("rnnResult").innerText =
    "RNN Accuracy: " + (r2.history.acc.pop()*100).toFixed(2)+"%";
})();
