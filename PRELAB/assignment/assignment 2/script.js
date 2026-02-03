const v1 = tf.tensor1d([1, 2, 3]);
const v2 = tf.tensor1d([4, 5, 6]);

const add = tf.add(v1, v2);
const mul = tf.mul(v1, v2);

add.print();
mul.print();
