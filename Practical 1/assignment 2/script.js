const xs = tf.tensor1d([1, 2, 3, 4]);
const ys = tf.tensor1d([3, 5, 7, 9]);

async function trainWithLR(lr) {
    console.log("Training with learning rate:", lr);

    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    model.compile({optimizer: tf.train.sgd(lr), loss: "meanSquaredError"});

    await model.fit(xs, ys, {epochs: 50});
    const prediction = model.predict(tf.tensor1d([5]));
    
    console.log(`LR ${lr} → Prediction:`);
    prediction.print();
}

trainWithLR(0.01);
trainWithLR(0.1);
trainWithLR(1.0);
