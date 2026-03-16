// Scalar (0D)
const scalar = tf.scalar(10);
scalar.print();

// Vector (1D)
const vector = tf.tensor1d([1, 2, 3]);
vector.print();

// Matrix (2D)
const matrix = tf.tensor2d([[1, 2], [3, 4]]);
matrix.print();
