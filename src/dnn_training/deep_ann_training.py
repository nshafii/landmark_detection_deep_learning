import tensorflow as tf
import numpy      as np
import math
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))

dataset_path      = dir_path+"/trunk_data_set/"
saved_model_path  = dir_path+"/learned_model/model.ckpt"
test_labels_file  = "test-labels.csv"
train_labels_file = "train-labels.csv"

IMAGE_WIDTH  = 30
IMAGE_HEIGHT = 30
IMAGE_DEPTH  = 3
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT
NUM_CLASSES  = 2

STEPS         = 50000
STEP_PRINT    = 20
STEP_VALIDATE = 100
LEARN_RATE    = 0.0014
DECAY_RATE    = 0.4
BATCH_SIZE    = 5


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Create model
def conv_net(x, keep_prob):
    # first convolutional leyer
    x_image = tf.reshape(x, [-1,30,30,3])

    W_conv1 = weight_variable([5, 5, 3, 30])
    b_conv1 = bias_variable([30])


    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolutional leyer
    W_conv2 = weight_variable([5, 5, 30, 60])
    b_conv2 = bias_variable([60])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # third leyer


    W_fc1 = weight_variable([8 * 8 * 60 , 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*60])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # drop out
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # rool out leyer
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])
    out = tf.add(tf.matmul(h_fc1_drop, W_fc2) , b_fc2)	
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


    example = tf.reshape( example, [ IMAGE_PIXELS * IMAGE_DEPTH ] )
    example.set_shape( [ IMAGE_PIXELS * IMAGE_DEPTH ] )

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
    keep_prob = tf.placeholder(tf.float32)
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # create the network
    y = conv_net( x, keep_prob )

    # loss
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits = y, labels=y_) )

    #learning_rate = tf.placeholder(tf.float32, shape=[])

    # train step
    train_step   = tf.train.AdamOptimizer( 1e-4 ).minimize( cost )


    ########################################
    # get filelist and labels for validation
    input_queue_test = tf.train.slice_input_producer([test_filepaths, test_labels], num_epochs=100, shuffle=True)

    # read files for validation
    image_test, label_test = read_images_from_disk( input_queue_test )

    # `image_batch_test` and `label_batch_test` represent the "next" batch
    # read from the input queue test.
    
    image_batch_test, label_batch_test = tf.train.batch( [ image_test, label_test ], batch_size=100 )

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train',
                                      sess.graph)
    test_writer = tf.summary.FileWriter('test')



    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

    sess.run(init_op)
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    
    # N.B. You must run this function before `sess.run(train_step)` to
    # start the input pipeline.
    #tf.train.start_queue_runners(sess)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    writer = tf.summary.FileWriter('/tmp/tensorflow_logs', graph=sess.graph)

	
    for i in range(STEPS):
	        
	# No need to feed, because `x` and `y_` are already bound to
        # the next input batch.       
        imgs, lbls = sess.run([image_batch, label_batch])
        sess.run(train_step, feed_dict={x: imgs, y_: lbls, keep_prob: 0.5}) 
	
        if i % STEP_PRINT == 0:
	    train_accuracy = accuracy.eval(feed_dict={x:imgs, y_: lbls, keep_prob: 1.0})
    	    print("step %d, training accuracy %g"%(i, train_accuracy))
	
        if i % STEP_VALIDATE == 0:
            imgs, lbls = sess.run([image_batch_test, label_batch_test])
	    validation_accuracy = accuracy.eval(feed_dict={x:imgs, y_: lbls, keep_prob: 1.0})
    	    print("step %d, validation accuracy %g"%(i, validation_accuracy))

	
	
	
    #save_path = saver.save(sess, saved_model_path)
    #print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)

	
