const xs = tf.tensor1d([0, 1, 2, 3, 4]);
const ys = tf.tensor1d([1, 3, 5, 7, 9]); // y = 2x + 1

const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({optimizer: tf.train.sgd(0.1), loss: 'meanSquaredError'});

async function predictUnseen() {
    await model.fit(xs, ys, {epochs: 150});

    const testValues = tf.tensor1d([6, 10, 15]);
    const preds = model.predict(testValues);

    console.log("Unseen Inputs:", testValues.arraySync());
    console.log("Predicted:", preds.arraySync());
}

predictUnseen();
