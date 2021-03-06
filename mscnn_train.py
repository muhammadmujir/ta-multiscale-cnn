# -*- coding:utf-8 -*-
"""
@Function: MSCNN crowd counting model training
@Source: Multi-scale Convolution Neural Networks for Crowd Counting
         https://arxiv.org/abs/1702.02359
@Data set: https://pan.baidu.com/s/12EqB1XDyFBB0kyinMA7Pqw Password: sags --> Have some problems

@Author: Ling Bao
@Code verification: Ling Bao
@illustrate:
    Learning rate: 1e-4
    Average loss : 14.

@Data: Sep. 11, 2017
@Version: 0.1
"""

# system library
import os.path
import random
import cv2
from six.moves import xrange

# machine learning library
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

# project library
import mscnn

# model parameter settings
FLAGS = tf.app.flags.FLAGS

# model training parameters
num_epochs_per_decay = 20
learning_rate_per_decay = 0.9
initial_learning_rate = 1.0e-1


def train():
    """
    Train mscnn model on ShanghaiTech test set
    :return:
    """
    with tf.Graph().as_default():
       # Read the file directory txt
        dir_file = open(FLAGS.data_train_index)
        dir_name = dir_file.readlines()

        # parameter settings
        nums_train = len(dir_name) # The number of images to train a batch
        global_step = tf.Variable(0, trainable=False) # Define the number of global decay steps

        # place_holder for training data
        image = tf.placeholder("float")
        label = tf.placeholder("float")
        avg_loss = tf.placeholder("float")

        # Initialization work related to model training
        # predicts = mscnn.inference(image) # build mscnn model
        predicts = mscnn.inference_bn(image) # Build an improved mscnn model
        loss = mscnn.loss(predicts, label) # Calculate loss
        train_op = mscnn.train(loss, global_step, nums_train) # Get the training operator

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) # create a session
        saver = tf.train.Saver(tf.all_variables()) # create a saver

        init = tf.initialize_all_variables() # variable initialization
        sess.run(init) # initialize all variables of the model

        checkpoint_dir = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if checkpoint_dir and checkpoint_dir.model_checkpoint_path:
            saver.restore(sess, checkpoint_dir.model_checkpoint_path)
        else:
            print('Not found checkpoint file')

        summary_op = tf.summary.merge_all() # summary summary
        add_avg_loss_op = mscnn.add_avg_loss(avg_loss) # Add the op of average loss
        summary_writer = tf.summary.FileWriter(FLAGS.train_log, graph_def=sess.graph_def) # Create a summaryr

        # parameter settings
        steps = 100000
        avg_loss_1 = 0

        for step in xrange(steps):
            if step < nums_train * 10:
                # Start 10 iterations of round robin training in sample order
                num_batch = [divmod(step, nums_train)[1] + i for i in range(FLAGS.batch_size)]
            else:
                # Randomly select batch_size samples
                num_batch = random.sample(range(nums_train), nums_train)[0:FLAGS.batch_size]

            xs, ys = [], []
            for index in num_batch:
                # get the path
                file_name = dir_name[index]
                im_name, gt_name = file_name.split(' ')
                gt_name = gt_name.split('\n')[0]

                # training data (image)
                batch_xs = cv2.imread(FLAGS.data_train_im + im_name)
                batch_xs = np.array(batch_xs, dtype=np.float32)

                # training data (density map)
                batch_ys = np.array(np.load(FLAGS.data_train_gt + gt_name))
                batch_ys = np.array(batch_ys, dtype=np.float32)
                batch_ys = batch_ys.reshape([batch_ys.shape[0], batch_ys.shape[1], -1])

                xs.append(batch_xs)
                ys.append(batch_ys)

            np_xs = np.array(xs)
            np_ys = np.array(ys)

            # Get loss value and predicted density map
            _, loss_value = sess.run([train_op, loss], feed_dict={image: np_xs, label: np_ys})
            output = sess.run(predicts, feed_dict={image: np_xs})
            avg_loss_1 += loss_value

            # save overview data
            if step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict={image: np_xs, label: np_ys,
                                                              avg_loss: avg_loss_1 / 100})
                summary_writer.add_summary(summary_str, step)
                avg_loss_1 = 0

            if step % 1 == 0:
                print("avg_loss:%.7f\t ??????counting:%.7f\t ??????predict:%.7f" % \
                      (loss_value, sum(sum(sum(np_ys))), sum(sum(sum(output)))))
                sess.run(add_avg_loss_op, feed_dict={avg_loss: loss_value})

            # save model parameters
            if step % 2000 == 0:
                checkpoint_path = os.path.join(FLAGS.model_dir, 'skip_mcnn.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            # output predicted density map (for testing)
            if step % 500 == 0:
                out_path = os.path.join(FLAGS.output_dir, str(step) + "out.npy")
                np.save(out_path, output)


def main(argv=None):
    if gfile.Exists(FLAGS.train_log):
        gfile.DeleteRecursively(FLAGS.train_log)
    gfile.MakeDirs(FLAGS.train_log)

    if not gfile.Exists(FLAGS.model_dir):
        gfile.MakeDirs(FLAGS.model_dir)

    if not gfile.Exists(FLAGS.output_dir):
        gfile.MakeDirs(FLAGS.output_dir)

    train()


if __name__ == '__main__':
    tf.app.run()
