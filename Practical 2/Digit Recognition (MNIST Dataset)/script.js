import { MNISTData } from "./mnist.js";

async function trainCNN() {
  const data = new MNISTData();
  await data.load();
  const { xs, labels } = data.getTrainData();

  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 16,
    kernelSize: 3,
    activation: "relu"
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  console.log("Training...");
  const history = await model.fit(xs, labels, {
    epochs: 5,
    batchSize: 128
  });

  console.log("Final Accuracy:", history.history.acc.pop());
}

trainCNN();
