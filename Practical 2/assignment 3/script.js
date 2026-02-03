async function loadData() {
    const mnist = await mnistData.load();
    const train = mnist.getTrainData();
    return {
        xsCNN: train.images.reshape([train.images.shape[0], 28, 28, 1]),
        xsDense: train.images.reshape([train.images.shape[0], 784]),
        labels: train.labels
    };
}

async function compareModels() {
    const data = await loadData();

    // Dense network
    const dense = tf.sequential();
    dense.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [784] }));
    dense.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
    dense.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    console.log("Training Dense model...");
    const denseHist = await dense.fit(data.xsDense, data.labels, { epochs: 3 });
    console.log("Dense Accuracy:", denseHist.history.acc.pop());

    // CNN network
    const cnn = tf.sequential();
    cnn.add(tf.layers.conv2d({ inputShape: [28, 28, 1], filters: 16, kernelSize: 3, activation: 'relu' }));
    cnn.add(tf.layers.flatten());
    cnn.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
    cnn.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    console.log("Training CNN model...");
    const cnnHist = await cnn.fit(data.xsCNN, data.labels, { epochs: 3 });
    console.log("CNN Accuracy:", cnnHist.history.acc.pop());
}

compareModels();
