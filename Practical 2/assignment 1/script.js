import { MNISTData } from "./mnist.js";

async function train() {
  const data = new MNISTData();
  await data.load();
  const { xs, labels } = data.getTrainData();

  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 8,
    kernelSize: 3,
    activation: "relu"
  }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  console.log("Training...");
  const hist = await model.fit(xs, labels, { epochs: 5 });

  console.log("Accuracy:", hist.history.acc.pop());
}

train();
