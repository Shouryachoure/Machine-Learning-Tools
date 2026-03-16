// mnist.js (100% working MNIST loader for TensorFlow.js)

export class MNISTData {
  constructor() {
    this.IMAGE_SIZE = 28 * 28;
    this.NUM_CLASSES = 10;
    this.NUM_DATASET_ELEMENTS = 65000;

    this.IMAGE_SPRITE_PATH =
      "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";

    this.LABELS_PATH =
      "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";
  }

  async load() {
    const imgRequest = fetch(this.IMAGE_SPRITE_PATH);
    const labelsRequest = fetch(this.LABELS_PATH);

    const [imgResponse, labelsResponse] = await Promise.all([
      imgRequest,
      labelsRequest,
    ]);

    const imgBlob = await imgResponse.blob();
    const bitmap = await createImageBitmap(imgBlob);

    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(bitmap, 0, 0);

    const { data } = ctx.getImageData(
      0,
      0,
      canvas.width,
      canvas.height
    );

    this.datasetImages = new Float32Array(
      this.NUM_DATASET_ELEMENTS * this.IMAGE_SIZE
    );

    for (let i = 0; i < this.NUM_DATASET_ELEMENTS * this.IMAGE_SIZE; i++) {
      this.datasetImages[i] = data[i * 4] / 255;
    }

    this.datasetLabels = new Uint8Array(
      await labelsResponse.arrayBuffer()
    );
  }

  getTrainData() {
    const trainCount = 55000;

    const xs = tf.tensor2d(
      this.datasetImages.slice(0, trainCount * this.IMAGE_SIZE),
      [trainCount, this.IMAGE_SIZE]
    ).reshape([trainCount, 28, 28, 1]);

    const labels = tf.tensor2d(
      this.datasetLabels.slice(0, trainCount * this.NUM_CLASSES),
      [trainCount, this.NUM_CLASSES]
    );

    return { xs, labels };
  }

  getTestData(count = 10000) {
    const start = 55000 * this.IMAGE_SIZE;
    const startLabels = 55000 * this.NUM_CLASSES;

    const xs = tf.tensor2d(
      this.datasetImages.slice(start, start + count * this.IMAGE_SIZE),
      [count, this.IMAGE_SIZE]
    ).reshape([count, 28, 28, 1]);

    const labels = tf.tensor2d(
      this.datasetLabels.slice(startLabels, startLabels + count * this.NUM_CLASSES),
      [count, this.NUM_CLASSES]
    );

    return { xs, labels };
  }
}
