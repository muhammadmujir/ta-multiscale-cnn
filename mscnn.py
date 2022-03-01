# -*- coding:utf-8 -*-
"""
@Function: Structure of MSCNN crowd counting
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

# 系统模块
import re

# 机器学习库
import tensorflow as tf

# 项目模块
import mscnn_train

# 模型参数设置
MP_NAME = 'mp'
train_log = 'train_log'
model = 'model'
output = 'output'
data_train_gt = 'Data_original/Data_gt/train_gt/'
data_train_im = 'Data_original/Data_im/train_im/'
data_train_index = 'Data_original/dir_name.txt'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 1, """Number of batched pictures""")
tf.app.flags.DEFINE_string('train_log', train_log, """training log""")
tf.app.flags.DEFINE_string('model_dir', model, """Model save""")
tf.app.flags.DEFINE_string('output_dir', output, """Output intermediate results""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log the device layout""")
tf.app.flags.DEFINE_string('data_train_gt', data_train_gt, """Training set label""")
tf.app.flags.DEFINE_string('data_train_im', data_train_im, """Training set image""")
tf.app.flags.DEFINE_string('data_train_index', data_train_index, """Training set image""")


def _activation_summary(x):
    """
    Summary summary function
    :param x: variable to save
    :return: None
    """
    tensor_name = re.sub('%s_[0-9]*/' % MP_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """
    create variable
    :param name: name_scope
    :param shape: tensor dimension
    :param initializer: initializer value
    :return: tensor variable
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)

    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    Create a variable with a weight decay term
    :param name: name_scope
    :param shape: tensor dimension
    :param stddev: standard deviation for initialization
    :param wd: weight
    :return: tensor variable
    """
    # wd is the attenuation factor, if it is None, there is no attenuation term
    var = _variable_on_cpu(name, shape, tf.random_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


class BatchNorm(object):
    """
    BN operation class
    """
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        """
        initialization function
        :param epsilon: precision
        :param momentum: momentum factor
        :param name: name_scope
        """
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x):
        """
        BN operator
        :param x: input variable
        :return:
        """
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None,
                                            epsilon=self.epsilon, scale=True, scope=self.name)


def multi_scale_block(in_con, in_dim, out_dim, is_bn=False):
    """
    Multiscale Block MSB
    :param in_con: input tensor variables [batch_size, filter_w, filter_h, in_dim]
    :param in_dim: number of input channels
    :param out_dim: number of output channels
    :param is_bn: whether to add Batch Normal
    :return: output tensor variable [4 * batch_size, filter_w, filter_h, in_dim]
    """
    with tf.variable_scope('con_9') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[9, 9, in_dim, out_dim],
                                             stddev=0.01, wd=0.0005)
        con_9 = tf.nn.conv2d(in_con, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        _activation_summary(con_9)

    with tf.variable_scope('con_7') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[7, 7, in_dim, out_dim],
                                             stddev=0.01, wd=0.0005)
        con_7 = tf.nn.conv2d(in_con, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        _activation_summary(con_7)

    with tf.variable_scope('con_5') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, in_dim, out_dim],
                                             stddev=0.01, wd=0.0005)
        con_5 = tf.nn.conv2d(in_con, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        _activation_summary(con_5)

    with tf.variable_scope('con_3') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, in_dim, out_dim],
                                             stddev=0.01, wd=0.0005)
        con_3 = tf.nn.conv2d(in_con, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        _activation_summary(con_3)

    with tf.variable_scope('concat') as scope:
        concat = tf.concat([con_9, con_7, con_5, con_3], 3, name=scope.name)
        biases = _variable_on_cpu('biases', [out_dim * 4], tf.constant_initializer(0))
        bias = tf.nn.bias_add(concat, biases)

        if is_bn:
            bn = BatchNorm()
            bias = bn(bias)

        msb = tf.nn.relu(bias)
        _activation_summary(msb)

    return msb


def inference(images):
    """
    Build the MSCNN model
    :param images: original image
    :return: crowd density estimation image
    """
    # -------------------------------------------------------------------------------------------- #
    # Create a model
    # con1_1
    with tf.variable_scope('con1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[9, 9, 3, 64],
                                             stddev=0.01, wd=0.0005)
        con = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0))
        bias = tf.nn.bias_add(con, biases)
        con1 = tf.nn.relu(bias)
        _activation_summary(con1)

    # msb_con2
    with tf.variable_scope('msb_con2'):
        msb_con2 = multi_scale_block(con1, 64, 16)

    # pool_msb_con2
    with tf.variable_scope('pool_msb_con2') as scope:
        pool_msb_con2 = tf.nn.max_pool(msb_con2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                       name=scope.name)

    # msb_con3
    with tf.variable_scope('msb_con3'):
        msb_con3 = multi_scale_block(pool_msb_con2, 64, 32)

    # msb_con4
    with tf.variable_scope('msb_con4'):
        msb_con4 = multi_scale_block(msb_con3, 128, 32)

    # pool_msb_con4
    with tf.variable_scope('pool_msb_con4') as scope:
        pool_msb_con4 = tf.nn.max_pool(msb_con4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                       name=scope.name)

    # msb_con5
    with tf.variable_scope('msb_con5'):
        msb_con5 = multi_scale_block(pool_msb_con4, 128, 64)

    # msb_con6
    with tf.variable_scope('msb_con6'):
        msb_con6 = multi_scale_block(msb_con5, 256, 64)

    # mpl_con7
    with tf.variable_scope('mpl_con7') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, 1000], stddev=0.001, wd=0.0005)
        con = tf.nn.conv2d(msb_con6, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        biases = _variable_on_cpu('biases', [1000], tf.constant_initializer(0))
        bias = tf.nn.bias_add(con, biases)
        mpl_con7 = tf.nn.relu(bias)
        _activation_summary(mpl_con7)

    # con_out
    with tf.variable_scope('con_out') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 1000, 1], stddev=0.001, wd=0.0005)
        con = tf.nn.conv2d(mpl_con7, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        biases = _variable_on_cpu('biases', [1], tf.constant_initializer(0))
        bias = tf.nn.bias_add(con, biases)

        con_out = tf.nn.relu(bias)
        _activation_summary(con_out)

    # Delete the fourth dimension channel, channel=1
    image_out = con_out

    tf.summary.image("con_img", image_out)

    return image_out


def inference_bn(images):
    """
    Added Batch Normal after the cnn layer of the MSCNN model; improved the output activation function f(x)=relu(sigmoid(x))
    $$sigmod(x)=\frac{1}{1+e^{-x}}$$
    $$relu(x)=
    \begin{equation}
    \begin{cases}
     x, & x \geq 0 \\
    0, & x < 0
    \end{cases}
    \end{equation}$$
    $$f(x)=relu(sigmod(x))$$

    :param images: original image
    :return: crowd density estimation image
    """
    # -------------------------------------------------------------------------------------------- #
    # Create a model
    # con1_1
    with tf.variable_scope('con1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[9, 9, 3, 64],
                                             stddev=0.01, wd=0.0005)
        con = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0))
        bias = tf.nn.bias_add(con, biases)
        con1 = tf.nn.relu(bias)
        _activation_summary(con1)

    # msb_con2
    with tf.variable_scope('msb_con2'):
        msb_con2 = multi_scale_block(con1, 64, 16, is_bn=True)

    # pool_msb_con2
    with tf.variable_scope('pool_msb_con2') as scope:
        pool_msb_con2 = tf.nn.max_pool(msb_con2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                       name=scope.name)

    # msb_con3
    with tf.variable_scope('msb_con3'):
        msb_con3 = multi_scale_block(pool_msb_con2, 64, 32, is_bn=True)

    # msb_con4
    with tf.variable_scope('msb_con4'):
        msb_con4 = multi_scale_block(msb_con3, 128, 32, is_bn=True)

    # pool_msb_con4
    with tf.variable_scope('pool_msb_con4') as scope:
        pool_msb_con4 = tf.nn.max_pool(msb_con4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                       name=scope.name)

    # msb_con5
    with tf.variable_scope('msb_con5'):
        msb_con5 = multi_scale_block(pool_msb_con4, 128, 64, is_bn=True)

    # msb_con6
    with tf.variable_scope('msb_con6'):
        msb_con6 = multi_scale_block(msb_con5, 256, 64, is_bn=True)

    # mpl_con7
    with tf.variable_scope('mpl_con7') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, 1000], stddev=0.001, wd=0.0005)
        con = tf.nn.conv2d(msb_con6, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        biases = _variable_on_cpu('biases', [1000], tf.constant_initializer(0))
        bias = tf.nn.bias_add(con, biases)
        mpl_con7 = tf.nn.relu(bias)
        _activation_summary(mpl_con7)

    # con_out
    with tf.variable_scope('con_out') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 1000, 1], stddev=0.001, wd=0.0005)
        con = tf.nn.conv2d(mpl_con7, kernel, [1, 1, 1, 1], padding='SAME', name=scope.name)
        biases = _variable_on_cpu('biases', [1], tf.constant_initializer(0))
        bias = tf.nn.bias_add(con, biases)

        bn = BatchNorm()
        bias = bn(bias)

        con_out = tf.nn.relu(tf.nn.sigmoid(bias))
        _activation_summary(con_out)

    # remove the fourth dimension channel, channel=1
    image_out = con_out

    tf.summary.image("con_img", image_out)

    return image_out


def loss(predict, label):
    """
    Calculate the loss
    :param predict: mscnn estimated density map
    :param label: ground truth crowd counting map
    :return: L2 loss
    """
    # L2 Loss
    predict = tf.squeeze(predict, 3)
    l2_loss = tf.reduce_sum((predict - label) * (predict - label))

    # Add summary
    tf.summary.histogram('loss', l2_loss)

    return l2_loss


def add_avg_loss(avg_loss):
    """
    Calculate the average loss
    :param avg_loss:
    :return:
    """
    add_avg_loss_op = avg_loss * 1
    tf.summary.histogram('avg_loss', avg_loss)

    return add_avg_loss_op


def _add_loss_summaries(total_loss):
    """
    Add loss summary information
    :param total_loss:
    :return:
    """
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step, nums_per_train):
    """
    Construct RMSProp optimization operator based on loss
    :param total_loss: loss
    :param global_step:
    :param nums_per_train:
    :return: RMSProp optimization operator
    """
    num_batches_per_epoch = nums_per_train / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * mscnn_train.num_epochs_per_decay)

    lr = tf.train.exponential_decay(mscnn_train.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    mscnn_train.learning_rate_per_decay,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # optimization
    opt = tf.train.RMSPropOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

    # apply gradient
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    train_op = apply_gradient_op

    # Add summary
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    return train_op
