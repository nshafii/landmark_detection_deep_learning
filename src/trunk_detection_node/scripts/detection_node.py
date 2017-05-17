#!/usr/bin/env python

## Node developed to inference a deep neural network to detect trunk

import rospy
import tensorflow as tf
import timeit
import rospkg
from math import exp
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

IMAGE_WIDTH  = 30
IMAGE_HEIGHT = 30
IMAGE_DEPTH  = 3
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT

rospack = rospkg.RosPack()
LEARNED_MODEL_ADDRESS = rospack.get_path('trunk_detection_node')+'/../dnn_training/learned_model/'

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_net(x, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2):
    # first convolutional leyer
    x_image = tf.reshape(x, [-1,30,30,3])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolutional leyer
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # third leyer

    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*60])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # drop out
    h_fc1_drop = tf.nn.dropout(h_fc1, 1.0)

    # rool out leyer
    out = tf.add(tf.matmul(h_fc1_drop, W_fc2) , b_fc2)	
    return out 



def callback(data):
    print '-----------------------------------------------------------------------'	
    rospy.loginfo(rospy.get_caller_id() + 'image has been recieved %s', data.data)
    # disabling the GPU use only the CPU	
    config = tf.ConfigProto( device_count = {'GPU': 0} )
	
    with tf.Session(config=config) as sess1:
   
       saver = tf.train.import_meta_graph(LEARNED_MODEL_ADDRESS +'model.ckpt.meta')
       saver.restore(sess1,LEARNED_MODEL_ADDRESS+'model.ckpt')

       start = timeit.default_timer()
  
       # Restoreing the model and weights")
 
       W_conv1 = [v for v in tf.trainable_variables() if v.name == "Variable:0"][0]

       b_conv1 = [v for v in tf.trainable_variables() if v.name == "Variable_1:0"][0]
  
       W_conv2 = [v for v in tf.trainable_variables() if v.name == "Variable_2:0"][0]

       b_conv2 = [v for v in tf.trainable_variables() if v.name == "Variable_3:0"][0]
  
       W_fc1 =  [v for v in tf.trainable_variables() if v.name == "Variable_4:0"][0]
  
       b_fc1 =  [v for v in tf.trainable_variables() if v.name == "Variable_5:0"][0]
  
       W_fc2 =  [v for v in tf.trainable_variables() if v.name == "Variable_6:0"][0]
  
       b_fc2 =  [v for v in tf.trainable_variables() if v.name == "Variable_7:0"][0]	

       image_input = data.data.reshape(30,30,3)

       img2 = tf.convert_to_tensor(image_input)
       img2 = tf.reshape( img2, [ IMAGE_PIXELS * IMAGE_DEPTH ] )
       img2.set_shape( [ IMAGE_PIXELS * IMAGE_DEPTH ] )

       image_input = tf.cast( img2, tf.float32 ) * ( 1. / 255 ) - 0.5
 
       y = conv_net(image_input,W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)

       stop = timeit.default_timer()

       print "There is no trunk with %f probablity" % (1/(1+exp(-y.eval()[0][1])))

       print "There is a trunk with %f probablity" % (1/(1+exp(-y.eval()[0][0])))

       print "calculation time :", stop - start	

def listener():

    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("floats", numpy_msg(Floats), callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    
    listener()
