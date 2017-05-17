import tensorflow as tf
import numpy      as np
import math
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))

dataset_path      = dir_path+"/trunk_data_set/"
test_labels_file  = "test-labels.csv"
train_labels_file = "train-labels.csv"

IMAGE_WIDTH  = 30
IMAGE_HEIGHT = 30
IMAGE_DEPTH  = 3
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT
NUM_CLASSES  = 2

STEPS         = 50000
STEP_PRINT    = 100
STEP_VALIDATE = 100
LEARN_RATE    = 0.0014
DECAY_RATE    = 0.4
BATCH_SIZE    = 5

# Create model
def ann_net(x):

    img_width  = IMAGE_WIDTH
    img_height = IMAGE_HEIGHT
    img_depth  = IMAGE_DEPTH

    weights    = tf.Variable( tf.random_normal( [ img_width * img_height * img_depth, NUM_CLASSES ] ) )
    biases     = tf.Variable( tf.random_normal( [ NUM_CLASSES ] ) )

    # softmax layer
    out        = tf.add( tf.matmul( x, weights ), biases )
    return out 

def encode_label(label):
    return int(label)

def read_label_file(file):
    f = open(file, "r")
    filepaths = []
    labels = []
    for line in f:
        filepath, label = line.split(",")
        filepaths.append(filepath)
        labels.append(encode_label(label))
    return filepaths, labels

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    print( "read file "  )
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=IMAGE_DEPTH)
    
    example = tf.reshape( example, [ IMAGE_DEPTH*IMAGE_PIXELS ] )
    example.set_shape( [ IMAGE_DEPTH * IMAGE_PIXELS ] )

    example = tf.cast( example, tf.float32 )
    example = tf.cast( example, tf.float32 ) * ( 1. / 255 ) - 0.5

    label = tf.cast( label, tf.int64 )

    label = tf.one_hot( label, 2, 0, 1 )
    label = tf.cast( label, tf.float32 )

    print( "file read " )
    return  example, label

# reading labels and file path
train_filepaths, train_labels = read_label_file(dataset_path + train_labels_file)
test_filepaths, test_labels = read_label_file(dataset_path + test_labels_file)

print "input pipeline ready"


with tf.Session() as sess:

    ########################################
    # get filelist and labels for training


    input_queue = tf.train.slice_input_producer([train_filepaths, train_labels], num_epochs=100, shuffle=True)

    image, label = read_images_from_disk(input_queue)

    # read files for training
    image, label = read_images_from_disk( input_queue )

    # `image_batch` and `label_batch` represent the "next" batch
    # read from the input queue.
    image_batch, label_batch = tf.train.batch( [ image, label ], batch_size = BATCH_SIZE )

    # input output placeholders

    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS*IMAGE_DEPTH ])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # create the network
    y = ann_net( x )

    # loss
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits = y, labels=y_) )

    learning_rate = tf.placeholder(tf.float32, shape=[])

    # train step
    train_step   = tf.train.AdamOptimizer( 1e-3 ).minimize( cost )


    ########################################
    # get filelist and labels for validation
    input_queue_test = tf.train.slice_input_producer([test_filepaths, test_labels], num_epochs=100, shuffle=True)

    # read files for validation
    image_test, label_test = read_images_from_disk( input_queue_test )

    # `image_batch_test` and `label_batch_test` represent the "next" batch
    # read from the input queue test.
    
    print "Step 1"
    image_batch_test, label_batch_test = tf.train.batch( [ image_test, label_test ], batch_size=200 )

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

    sess.run(init_op)

	
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(STEPS):
        # the next input batch.           
	if i % STEP_PRINT == 0:
	    print ("Step: %s " % (i))	
            imgs, lbls = sess.run([image_batch_test, label_batch_test])

            print("Test Accuracy: ", sess.run(accuracy, feed_dict={
                    x: imgs,
                    y_: lbls}))
	
        imgs, lbls = sess.run([image_batch, label_batch])

        sess.run(train_step, feed_dict={x: imgs, y_: lbls}) 

    imgs, lbls = sess.run([image_batch_test, label_batch_test])

    print(sess.run(accuracy, feed_dict={
         x: imgs,
         y_: lbls}))

    coord.request_stop()
    coord.join(threads)
