import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
import datetime

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")


learning_rate = 0.0002
training_epochs = 1000
batch_size = 100
noise_n = 100
flag = 0

X = tf.placeholder(tf.float32, [None,784])
Z = tf.placeholder(tf.float32, [None, 100])
Y = tf.placeholder(tf.float32, [None, 100])
E = tf.placeholder(tf.string)
J = tf.placeholder(tf.string)


# for Discriminator
W1_d = tf.get_variable("W1_d", shape = [3,3,1,32], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
W2_d = tf.get_variable("W2_d", shape = [3,3,32,64], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
W3_d = tf.get_variable("W3_d", shape = [28*28*64, 625], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
b3_d = tf.Variable(tf.random_normal([625]))
W4_d = tf.get_variable("W4_d", shape = [625, 1], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
b4_d = tf.Variable(tf.random_normal([1]))

D_var_list = [W1_d, W2_d, W3_d, b3_d, W4_d, b4_d]

# for Generator
W1_g = tf.get_variable("W1_g", shape = [noise_n,256], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
b1_g = tf.Variable(tf.random_normal([256]))
W2_g = tf.get_variable("W2_g", shape = [256,512], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
b2_g = tf.Variable(tf.random_normal([512]))
W3_g = tf.get_variable("W3_g", shape = [512,784], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
b3_g = tf.Variable(tf.random_normal([784]))

G_var_list = [W1_g, b1_g, W2_g, b2_g, W3_g, b3_g]


def Generator(Zin):

    # Layer 1 : FC
    L1 = tf.nn.relu(tf.matmul(Zin, W1_g) + b1_g)

    # Layer 2 : FC
    L2 = tf.nn.relu(tf.matmul(L1, W2_g) + b2_g)

    # Layer 3 : FC
    L3 = tf.nn.sigmoid(tf.matmul(L2, W3_g) + b3_g)

    return L3



def Discriminator(X):

    X_img = tf.reshape(X, [-1,28,28,1])
    
    # Layer 1 : conv
      
    L1 = tf.nn.conv2d(X_img, W1_d, strides=[1,1,1,1], padding = 'SAME')
    L1 = tf.nn.relu(L1)

    # Layer 2 : conv
    L2 = tf.nn.conv2d(L1, W2_d, strides=[1,1,1,1], padding = 'SAME')
    L2 = tf.nn.relu(L2)
    L2_flat = tf.reshape(L2, [-1, 28*28*64])

    # Layer 3 : FC
    L3 = tf.nn.relu(tf.matmul(L2_flat,W3_d)+b3_d)
    L3 = tf.nn.dropout(L3, keep_prob=0.7)

    # Layer 4 : FC
    L4 = tf.nn.sigmoid(tf.matmul(L3,W4_d)+b4_d)

    return L4


def make_noise(batch_size, noise_n):
    return np.random.normal(size=[batch_size, noise_n])


G = Generator(Z)

D_fake = Discriminator(G)
D_real = Discriminator(X)

D_loss = tf.reduce_mean(tf.log(D_real) + tf.log(1-D_fake))
G_loss = tf.reduce_mean(tf.log(D_fake))

D_train = tf.train.AdamOptimizer(learning_rate).minimize(-D_loss, var_list = D_var_list)
G_train = tf.train.AdamOptimizer(learning_rate).minimize(-G_loss, var_list = G_var_list)

samples = Generator(Y)
samples_img = tf.reshape(tf.cast(samples*128, tf.uint8), [-1,28,28,1])
img = tf.image.encode_jpeg(tf.reshape(samples_img,[28,28,1]), format='grayscale')
temp_name = tf.constant("./testimages/")+E+tf.constant("_")+J+tf.constant(".jpeg") 
fsave = tf.write_file(temp_name,img)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Learing Started')

total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(training_epochs):
    if True : #(epoch + 1) % 10 == 0 or epoch == 0:
            flag = 1
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = make_noise(batch_size, noise_n)

        _, loss_val_D = sess.run([D_train, D_loss], feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([G_train, G_loss], feed_dict={Z: noise})

        print('Epoch:', '%04d' % epoch,
          'D loss: {:.4}'.format(loss_val_D),
          'G loss: {:.4}'.format(loss_val_G))

        if flag == 1:
            sample_size = 10
            for j in range(sample_size):
                sample_noise = make_noise(1, noise_n)
                # temp_name =  str(epoch) + "_" + str(j) + ".jpeg"
                # fname = tf.constant("./testimages/"+temp_name)
                sess.run(fsave, feed_dict={Y:sample_noise, E:str(epoch), J:str(j)})

            flag = 0


print('Learning finished')

        




        
