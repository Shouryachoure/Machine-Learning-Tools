const original = tf.tensor([1, 2, 3, 4, 5, 6]);

// reshape()
const reshaped = original.reshape([2, 3]);
reshaped.print();

// flatten()
const flattened = reshaped.flatten();
flattened.print();
