# landmark detection using deep learning 
A deep learning approach has been developed to detect trunk as static landmarks. This repository provides different modules for creating data-set, training a deep neural network, and detecting the trunks. These modules have been developed using ROS and Tensorflow.

## DNN training package 
This package provides data-set creation for validation and training. This package also provide a trained DNN model in /learned_model folder by which the validation result was about 99.8% accuracy. Alternatively it is possible to change the architecture of DNN and trained a new model. 

## Trunk Detection package
A two sample ROS nodes were provided the first is for publishing the image. (the RGB image should be in PNG format and 30 by 30 pixels). The other node is developed to receive the image and perform the detection.

## Installation prerequisite
Install tensorflow from the source as:
https://www.tensorflow.org/install/install_sources

Alternatively if you want to use only tha CPU and the trunk detection node, you can install only the binary form:
https://www.tensorflow.org/install/install_linux

Install ROS Indego base or prefrably Desktop-full from:
http://wiki.ros.org/indigo/Installation/Ubuntu

## Install the package
catkin_make in the root of repository

## Run the detection node
 source devel/setup.bash 

 rosrun trunk_detection_node img_publisher.py 
 
 rosrun trunk_detection_node detection_node.py


