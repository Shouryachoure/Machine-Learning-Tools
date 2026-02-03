export class MNISTData {
  constructor() {
    this.IMAGE_SIZE = 28 * 28;
    this.NUM_CLASSES = 10;
    this.NUM_DATASET_ELEMENTS = 65000;

    this.TRAIN_TEST_RATIO = 5 / 6;
    this.NUM_TRAIN_ELEMENTS =
        Math.floor(this.TRAIN_TEST_RATIO * this.NUM_DATASET_ELEMENTS);
    this.NUM_TEST_ELEMENTS =
        this.NUM_DATASET_ELEMENTS - this.NUM_TRAIN_ELEMENTS;

    this.MNIST_IMAGES_SPRITE_PATH =
        'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
    this.MNIST_LABELS_PATH =
        'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';
  }

  async load() {
    const imgRequest = fetch(this.MNIST_IMAGES_SPRITE_PATH);
    const labelsRequest = fetch(this.MNIST_LABELS_PATH);

    const [imgResponse, labelsResponse] =
        await Promise.all([imgRequest, labelsRequest]);

    const img = await createImageBitmap(await imgResponse.blob());
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    this.datasetImages = new Float32Array(this.NUM_DATASET_ELEMENTS * this.IMAGE_SIZE);

    for (let i = 0; i < imageData.data.length / 4; i++) {
      this.datasetImages[i] = imageData.data[i * 4] / 255;
    }

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    this.trainImages =
        this.datasetImages.slice(0, this.IMAGE_SIZE * this.NUM_TRAIN_ELEMENTS);
    this.testImages =
        this.datasetImages.slice(this.IMAGE_SIZE * this.NUM_TRAIN_ELEMENTS);

    this.trainLabels =
        this.datasetLabels.slice(0, this.NUM_CLASSES * this.NUM_TRAIN_ELEMENTS);
    this.testLabels =
        this.datasetLabels.slice(this.NUM_CLASSES * this.NUM_TRAIN_ELEMENTS);
  }

  getTrainData() {
    const xs = tf.tensor2d(
        this.trainImages, [this.NUM_TRAIN_ELEMENTS, this.IMAGE_SIZE])
        .reshape([this.NUM_TRAIN_ELEMENTS, 28, 28, 1]);
    const labels = tf.tensor2d(
        this.trainLabels, [this.NUM_TRAIN_ELEMENTS, this.NUM_CLASSES]);
    return {xs, labels};
  }
}
