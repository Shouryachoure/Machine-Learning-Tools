// Generate synthetic data: y = 2x + 1
const xs = tf.tensor1d([1, 2, 3, 4, 5]);
const ys = tf.tensor1d([3, 5, 7, 9, 11]);

// Build model
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

model.compile({
    optimizer: tf.train.sgd(0.1),
    loss: 'meanSquaredError'
});

// Train
async function trainModel() {
    console.log("Training Started...");
    await model.fit(xs, ys, {epochs: 100});
    console.log("Training Completed!");
    
    const output = model.predict(tf.tensor1d([6]));
    output.print(); // Expected approx => 13
}

trainModel();
