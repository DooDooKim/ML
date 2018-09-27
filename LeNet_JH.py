import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import matplotlib.pyplot as plt
learning_rate = 0.001
training_epochs = 15
batch_size = 100
mnist = input_data.read_data_sets("MNIST_data/",reshape=False,one_hot=True)
x_train, y_train = mnist.train.images, mnist.train.labels
x_validation, y_validation = mnist.validation.images, mnist.validation.labels
x_test, y_test = mnist.test.images, mnist.test.labels
xy_batch = mnist.train.next_batch(batch_size)
x_batch = np.empty(2, dtype=object)
x_train = np.pad(x_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
x_validation = np.pad(x_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
x_test = np.pad(x_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

x_batch[0] = np.pad(xy_batch[0],((0,0),(2,2),(2,2),(0,0)),'constant')
x_batch[1] = xy_batch[1]

mn_trdata = tf.placeholder(tf.float32, [None,32,32,1])
fcl = tf.placeholder(tf.float32, [None, 10])
in_img = tf.reshape(mn_trdata, [-1,32,32,1])

w1 = tf.Variable(tf.random_normal([5,5,1,6], stddev=0.1))
b1 = tf.Variable(tf.random_normal([6]))
#3x3 짜리 1channel #5 filter
conv2d1 = tf.nn.conv2d(in_img, w1, strides=[1,1,1,1], padding='VALID')+b1
#SAME은 원래 이미지랑 같은 크기로 출력해주는 것 (zero-padding)
conv2d1 = tf.nn.max_pool(conv2d1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
conv2d1 = tf.sigmoid(conv2d1)

w2 = tf.Variable(tf.random_normal([5,5,6,16], stddev=0.1))
b2 = tf.Variable(tf.random_normal([16]))
conv2d2 = tf.nn.conv2d(conv2d1, w2, strides=[1,1,1,1], padding='VALID') + b2
conv2d2 = tf.nn.max_pool(conv2d2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
conv2d2 = tf.sigmoid(conv2d2)

w3 = tf.Variable(tf.random_normal([5,5,16,120], stddev=0.1))
b3 = tf.Variable(tf.random_normal([120]))
conv2d3 = tf.nn.conv2d(conv2d2, w3, strides=[1,1,1,1], padding='VALID') + b3
conv2d3 = tf.reshape(conv2d3, [-1,120])
conv2d3 = tf.sigmoid(conv2d3)

w4 = tf.Variable(tf.random_normal([120, 84]))
b4 = tf.Variable(tf.random_normal([84]))
fcl1 = tf.matmul(conv2d3,w4) + b4
fcl1 = 1.7159*tf.tanh(fcl1)

w5 = tf.Variable(tf.random_normal([84,10]))
b5 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(fcl1, w5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=fcl))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = x_batch[0], x_batch[1]
            feed_dict = {mn_trdata: batch_xs, fcl: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print('Learning Finished!')
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(fcl, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={mn_trdata: x_test, fcl: y_test}))
    # 랜덤하게 픽 1 / 결과확인
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(y_test[r:r + 1], 1)))
    print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={mn_trdata: x_test[r:r + 1]}))
    plt.imshow(x_test[r:r + 1].reshape(32, 32), cmap='Greys', interpolation='nearest')
    plt.show()