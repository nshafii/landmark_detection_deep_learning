import glob
import csv
import os 

data = []

dir_path = os.path.dirname(os.path.realpath(__file__))

filePathTestFasle = dir_path+"/img_test/false_seg_cube/"
for file in glob.glob(filePathTestFasle +"*.png"):
    data.append( [file,"0"])

filePathTestTrue = dir_path+"/img_test/true_seg_cube/"
for file in glob.glob(filePathTestTrue +"*.png"):
    data.append( [file,"1"])

with open('test-labels.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(data)

print "CSV for test data has been created"


data_train = []
filePathTrainFasle = dir_path+"/img_train/false_seg_cube/"
for file in glob.glob(filePathTrainFasle +"*.png"):
    data_train.append( [file,"0"])


filePathTrainTrue = dir_path+"/img_train/true_seg_cube/"
for file in glob.glob(filePathTrainTrue +"*.png"):
    data_train.append( [file,"1"])

with open('train-labels.csv', 'w') as fp:
    a_train = csv.writer(fp, delimiter=',')
    a_train.writerows(data_train)

print "CSV for train data has been created"
