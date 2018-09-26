# softmax regression

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('datasets/MNIST_data/', one_hot=True)

# softmax(x) = normalize(exp(x))

# 占位符
x = tf.placeholder('float', [None, 784])
# tf.Variable 可变化的张量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 占位符 用于输入正确值
y_ = tf.placeholder('float', [None, 10])

# 计算所有100幅图片的交叉熵总和
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 评估模型
# tf.argmax 返回最大值所在索引，对于one-hot，最大值1所在位置就是类别标签
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

print(sess.run(accuracy, feed_dict={
      x: mnist.test.images, y_: mnist.test.labels}))
