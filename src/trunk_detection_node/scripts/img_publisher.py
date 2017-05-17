#!/usr/bin/env python

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

import rospy
import numpy
import rospkg
from std_msgs.msg import String
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from scipy import misc


def get_single_img():
    file_path = rospack.get_path('trunk_detection_node')+'/../dnn_training/trunk_data_set/img_test/true_seg_cube/120.png' 
    img = misc.imread(file_path)
    return img

def talker():
    pub = rospy.Publisher('floats', numpy_msg(Floats),queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(1) # 1hz
    while not rospy.is_shutdown():
        a = get_single_img()
	reshape = a.reshape(-1)
        msg = numpy.array(reshape, dtype=numpy.float32)
        print 'image has been sent'
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        rospack = rospkg.RosPack()
        talker()
    except rospy.ROSInterruptException:
        pass
