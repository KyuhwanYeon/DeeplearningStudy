# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
sys.path.insert(0,'./dataset')
import os
print (os.getcwd()) #현재 디렉토리의
import tensorflow as tf
from dataset.mnist import load_mnist
import numpy as np
import matplotlib.pyplot as plt
import random


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)

exMNIST = x_train[1,]

exMNIST.reshape(28,28)
npEX = np.array(exMNIST)


def batches(batch_size, features):
    output = []
    samplesize = len(features)
    for i in range(0, samplesize, batch_size):
        end_i = i +batch_size
        batch = features[i:end_i]
        output.append(batch)    
    return output


def Labels_Table(labels, outputsize):
    length_labels = len(labels)
    output = np.zeros((length_labels,outputsize))
    for i in range(0,length_labels, 1):
        output[i,labels[i]] = 1
        
    return (output)


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)


# Preprocess
# Batch
t_train_table = Labels_Table(t_train,10)
x_train_batch = batches(100, x_train)
t_train_batch = batches(100, t_train_table)



t_test_table = Labels_Table(t_test,10)
x_test_batch = batches(100, x_test)
t_test_batch = batches(100, t_test_table)


iteration = 1000



# Build the Graph
W= tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
x = tf.placeholder("float",[None,784])

y = tf.nn.softmax(tf.matmul(x,W)+b)


y_ = tf.placeholder("float", [None,10])

cross_entrophy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entrophy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


Accuracy_ANN = {'Accuracy':[]}
for i in range(iteration):
    Rand_batch=random.randrange(0,len(t_train_batch))
    sess.run(train_step, feed_dict= {x:x_train_batch[Rand_batch], y_:t_train_batch[Rand_batch]})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sys.stdout.write("\rProgress: {:2.1f}%".format(100 * i/float(iteration)))
    if i%100 == 0:
        Accuracy = sess.run(accuracy,feed_dict={x:x_test, y_:t_test_table})
        Accuracy_ANN['Accuracy'].append(Accuracy*100)


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


print(sess.run(y, feed_dict={x: x_train_batch[1]}))


x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10])
x_image = tf.reshape(x, [-1,28,28,1])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)


#%%

title_font = {'fontname':'Arial', 'size':'30', 'color':'black', 'weight':'normal','verticalalignment':'bottom'}
label_font = {'fontname':'Arial', 'size':'30', 'color':'black', 'weight':'normal'}
plt.figure(figsize=(12,8))
plt.plot(Accuracy_ANN['Accuracy'], 'b',lw=4, label='Accuracy in test set')
plt.xlabel('Iterations X100',**label_font)
plt.ylabel('Accuracy[%]',**label_font)
plt.title('Accuracy in ANN single layer',**title_font)
plt.legend(loc=5,prop={'size':30})
plt.grid()
plt.tick_params(axis='x',labelsize=30)
plt.tick_params(axis='y',labelsize=30)
plt.show()
plt.legend()
_ = plt.ylim()

#%%