import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
import datetime

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

learning_rate = 0.0002
training_epochs = 1000
batch_size = 100
noise_n = 100
flag = 0

X = tf.placeholder(tf.float32, [None, 64,64,1])
Z = tf.placeholder(tf.float32, [None, 1,1,noise_n])
Y = tf.placeholder(tf.float32, [None, 1,1,noise_n])
E = tf.placeholder(tf.string)
J = tf.placeholder(tf.string)


# for Discriminator
W1_d = tf.get_variable("DW1", shape=[4, 4, 1, 128],
                       initializer=tf.random_normal_initializer(stddev=0.02))
W2_d = tf.get_variable("DW2", shape=[4, 4, 128, 256],
                       initializer=tf.random_normal_initializer(stddev=0.02))
W3_d = tf.get_variable("DW3", shape=[4, 4, 256, 512],
                       initializer=tf.random_normal_initializer(stddev=0.02))
W4_d = tf.get_variable("DW4", shape=[4, 4, 512, 1024],
                       initializer=tf.random_normal_initializer(stddev=0.02))
W5_d = tf.get_variable("DW5", shape=[4, 4, 1024, 1],
                       initializer=tf.random_normal_initializer(stddev=0.02))


# for Generator
W1_g = tf.get_variable("GW1", shape=[4, 4, 1024, 100],
                       initializer=tf.random_normal_initializer(stddev=0.02))
W2_g = tf.get_variable("GW2", shape=[4, 4, 512, 1024],
                       initializer=tf.random_normal_initializer(stddev=0.02))
W3_g = tf.get_variable("GW3", shape=[4, 4, 256, 512],
                       initializer=tf.random_normal_initializer(stddev=0.02))
W4_g = tf.get_variable("GW4", shape=[4, 4, 128, 256],
                       initializer=tf.random_normal_initializer(stddev=0.02))
W5_g = tf.get_variable("GW5", shape=[4, 4, 1, 128],
                       initializer=tf.random_normal_initializer(stddev=0.02))



def Generator(Zin, batch_size = batch_size, train = True, reuse=False,name='g'):

    with tf.variable_scope(name, reuse=reuse):

        # Layer 1 : Conv
        # L1 = tf.layers.conv2d_transpose(Zin, 1024, [4,4], strides=(1,1), padding = 'valid')
        L1 = tf.nn.conv2d_transpose(Zin, W1_g, [batch_size, 4, 4, 1024],
                                    strides=[1, 1, 1, 1], padding='VALID')
        L1 = tf.nn.relu(tf.layers.batch_normalization(L1, training=train))
        #print(L1)

        # Layer 2 : Conv
        # L2 = tf.layers.conv2d_transpose(L1, 512, [4,4], strides=(2,2), padding = 'same')
        L2 = tf.nn.conv2d_transpose(L1, W2_g, [batch_size, 8, 8, 512],
                                    strides=[1, 2, 2, 1], padding='SAME')
        L2 = tf.nn.relu(tf.layers.batch_normalization(L2, training=train))
        #print(L2)

        # Layer 3 : Conv
        # L3 = tf.layers.conv2d_transpose(L2, 256, [4,4], strides=(2,2), padding = 'same')
        L3 = tf.nn.conv2d_transpose(L2, W3_g, [batch_size, 16, 16, 256],
                                    strides=[1, 2, 2, 1], padding='SAME')
        L3 = tf.nn.relu(tf.layers.batch_normalization(L3, training=train))
        #print(L3)


        # Layer 4 : Conv
        # L4 = tf.layers.conv2d_transpose(L3, 128, [4,4], strides=(2,2), padding = 'same')
        L4 = tf.nn.conv2d_transpose(L3, W4_g, [batch_size, 32, 32, 128],
                                    strides=[1, 2, 2, 1], padding='SAME')
        L4 = tf.nn.relu(tf.layers.batch_normalization(L4, training=train))
        #print(L4)

        # Layer 5 : Conv
        # L5 = tf.layers.conv2d_transpose(L4, 1, [4,4], strides=(2,2), padding = 'same')
        L5 = tf.nn.conv2d_transpose(L4, W5_g, [batch_size, 64, 64, 1],
                                    strides=[1, 2, 2, 1], padding='SAME')
        L5 = tf.tanh(L5)
        #print(L5)
        L5 = (L5+1.0)/2.0


        return L5


def Discriminator(X, train=True, reuse=False ,name = 'd'):

    with tf.variable_scope(name, reuse=reuse):


        # Layer 1 : conv
        # L1 = tf.layers.conv2d(X, 128, [4,4], strides=(2,2), padding='same')
        L1 = tf.nn.conv2d(X, W1_d, strides=[1, 2, 2, 1], padding='SAME')
        L1 = tf.nn.leaky_relu(L1)

        # Layer 2 : conv
        # L2 = tf.layers.conv2d(L1, 256, [4,4], strides=(2,2), padding='same')
        L2 = tf.nn.conv2d(L1, W2_d, strides=[1, 2, 2, 1], padding='SAME')
        L2 = tf.nn.leaky_relu(tf.layers.batch_normalization(L2, training=train))

        # Layer 3 : conv
        # L3 = tf.layers.conv2d(L2, 512, [4,4], strides=(2,2), padding='same')
        L3 = tf.nn.conv2d(L2, W3_d, strides=[1, 2, 2, 1], padding='SAME')
        L3 = tf.nn.leaky_relu(tf.layers.batch_normalization(L3, training=train))

        # Layer 4 : conv
        # L4 = tf.layers.conv2d(L3, 1024, [4,4], strides=(2,2), padding='same')
        L4 = tf.nn.conv2d(L3, W4_d, strides=[1, 2, 2, 1], padding='SAME')
        L4 = tf.nn.leaky_relu(tf.layers.batch_normalization(L4, training=train))

        # Layer 4 : conv
        # L5 = tf.layers.conv2d(L4, 1, [4,4], strides=(1,1), padding='valid')
        L5 = tf.nn.conv2d(L4, W5_d, strides=[1, 1, 1, 1], padding='VALID')
        #L5 = tf.nn.sigmoid(tf.nn.leaky_relu(tf.layers.batch_normalization(L5, training=train)))
        L5 = tf.sigmoid(L5)


        return L5


def make_noise(batch_size, noise_n):
    return np.random.normal(0,1,(batch_size,1,1,noise_n))


G = Generator(Z, batch_size, True, False, name='G')


D_real = Discriminator(X,True,False,'D')
D_fake = Discriminator(G,True,True,'D')


D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

# D_loss_summary = tf.summary.histogram("D_loss", D_loss)
# G_loss_summary = tf.summary.histogram("G_loss", G_loss)

vars = tf.trainable_variables()
# print(vars)

D_var_list = [v for v in vars if v.name.startswith('D')]
G_var_list = [v for v in vars if v.name.startswith('G')]
# print(D_var_list)
# print(G_var_list)


D_train = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=D_var_list)
G_train = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=G_var_list)

samples = Generator(Y, 1, False, True, name='G')
samples_img = tf.cast(samples * 128, tf.uint8)
img = tf.image.encode_jpeg(tf.reshape(samples_img, [64, 64, 1]), format='grayscale')
temp_name = tf.constant("./testimages/") + E + tf.constant("_") + J + tf.constant(".jpeg")
fsave = tf.write_file(temp_name, img)





with tf.Session() as sess:


    #summary_merge = tf.summary.merge_all()
    #writer = tf.summary.FileWriter("./logs")
    #writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    print('Learing Started')

    total_batch = int(mnist.train.num_examples / batch_size)
    train_set = tf.image.resize_images(mnist.train.images, [64,64]).eval(session=sess)

    for epoch in range(training_epochs):
        if True:  # (epoch + 1) % 10 == 0 or epoch == 0:
            flag = 1
        for i in range(mnist.train.num_examples // batch_size):
            batch_xs = train_set[i*batch_size:(i+1)*batch_size]
            noise = make_noise(batch_size, noise_n)

            #_, loss_val_D, _, loss_val_G, summary = sess.run([D_train, D_loss, G_train, G_loss, summary_merge],
            #                                                 feed_dict={X: batch_xs, Z: noise})

            _, loss_val_D, _, loss_val_G = sess.run([D_train, D_loss, G_train, G_loss],
                                                            feed_dict={X: batch_xs, Z: noise})
            #writer.add_summary(summary, global_step=epoch * training_epochs + i * 100)

            print('Epoch:', '%04d' % epoch,
                  'D loss: {:.4}'.format(loss_val_D),
                  'G loss: {:.4}'.format(loss_val_G))

            if flag == 1:
                sample_size = 10
                for j in range(sample_size):
                    sample_noise = make_noise(1, noise_n)
                    # temp_name =  str(epoch) + "_" + str(j) + ".jpeg"
                    # fname = tf.constant("./testimages/"+temp_name)
                    sess.run(fsave, feed_dict={Y: sample_noise, E: str(epoch), J: str(j)})

                flag = 0

    print('Learning finished')