# landmark detection using deep learning 
A deep learning approach has been developed to detect trunk as static landmarks. This repository provides different modules for creating data-set, training a deep neural network, and detecting the trunks. These modules have been developed using ROS and Tensorflow.

## DNN training package 
This package provides data-set creation for validation and training. This package also provide a trained DNN model in /learned_model folder by which the validation result was about 99.8% accuracy. Alternatively it is possible to change the architecture of DNN and trained a new model. 

## Trunk Detection package
Two sample ROS nodes were developed. The first is for publishing the image (i.e. the RGB image in PNG format, 30 by 30 pixels). Another node is also provided to receive the image and perform the detection procedure.

## Installation prerequisite
Install tensorflow from the source as:
https://www.tensorflow.org/install/install_sources

Alternatively, if you want to use only tha CPU and the trunk detection nodes, you can install only the Tensorflow binary from:
https://www.tensorflow.org/install/install_linux

Install ROS Indego Base or prefrably Desktop-full version from:
http://wiki.ros.org/indigo/Installation/Ubuntu

## Installation of the package
catkin_make in the root of repository

## Running the detection node
 source devel/setup.bash 

 rosrun trunk_detection_node img_publisher.py 
 
 rosrun trunk_detection_node detection_node.py


