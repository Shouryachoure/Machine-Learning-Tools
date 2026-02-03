const xs = tf.tensor1d([1, 2, 3, 4, 5]);
const ys = tf.tensor1d([3, 5, 7, 9, 11]);

const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

async function run() {
    await model.fit(xs, ys, {epochs: 200});

    const predictions = model.predict(xs);
    predictions.print();

    console.log("Actual:", ys.arraySync());
    console.log("Predicted:", predictions.arraySync());
}

run();
