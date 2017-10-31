#!/usr/bin/env python3

import os
import sys
from sklearn.preprocessing import StandardScaler
import biutils  # used to load the dataset
import utils

def model(dataset, n_layers, n_hidden, activation, dropout_rate, use_batchnorm):

    x_tr, y_tr, x_va, y_va = biutils.load_dataset(dataset)
    s = StandardScaler()
    s.fit(x_tr)
    x_tr = s.transform(x_tr)
    x_va = s.transform(x_va)

    if n_hidden == -1:  # use as many hidden# as there are input features
        n_hidden = x_tr.shape[1]

    if activation == 'relu':
        act_fn = tf.nn.relu
        init_scale = 2.0
    elif activation == 'tanh':
        act_fn = tf.nn.tanh
        init_scale = 1.0
    elif activation == 'selu':
        act_fn = utils.selu
        init_scale = 1.0
    else:
        assert False, "Unknown activation"

    tf.reset_default_graph()
    x = tf.placeholder(np.float32, [None, x_tr.shape[1]], name="x")
    y = tf.placeholder(np.float32, [None, y_tr.shape[1]], name="y")
    is_training = tf.placeholder_with_default(tf.constant(False, tf.bool), shape=[], name='is_training')

    h = x
    if dropout_rate > 0.0:
        h = tf.layers.dropout(h, 0.2, training=is_training)

    for i in range(n_layers):
        s = np.sqrt(init_scale/h.get_shape().as_list()[1])
        init = tf.random_normal_initializer(stddev=s)
        h = tf.layers.dense(h, n_hidden, activation=act_fn, name='layer%d' % i, kernel_initializer=init)
        if use_batchnorm:
            h = tf.layers.batch_normalization(h, training=is_training)
        if dropout_rate > 0.0:
            h = tf.layers.dropout(h, dropout_rate, training=is_training)

    with tf.variable_scope('output_layer') as scope:
        o = tf.layers.dense(h, y_tr.shape[1], activation=None, name=scope)
        scope.reuse_variables()

    return (x_tr, y_tr, x_va, y_va), (x, y, is_training), o


def run(n_layers, n_hidden, n_epochs, learning_rate, dataset, activation, logdir_base='/tmp',
        batch_size=64, dropout_rate=0.0, use_batchnorm=False):

    ld = '%s%s_d%02d_h%d_l%1.0e_%s' % (activation,
        'bn' if use_batchnorm else '',
        n_layers, n_hidden, learning_rate,
        utils.get_timestamp())
    logdir = os.path.join(logdir_base, dataset, ld)
    print(logdir)

    dataset, variables, logits = model(dataset, n_layers, n_hidden, activation, dropout_rate, use_batchnorm)
    x_tr, y_tr, x_va, y_va = dataset
    x, y, is_training = variables

    prob_op = tf.nn.softmax(logits)
    loss_op = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    variables_to_train = tf.trainable_variables()
    grads = optimizer.compute_gradients(loss_op, variables_to_train)
    global_step = tf.train.get_global_step()
    train_op = optimizer.apply_gradients(grads, global_step=global_step)

    loss_val = tf.Variable(0.0, trainable=False, dtype=np.float32)
    acc_op, acc_upd = tf.metrics.accuracy(tf.argmax(y, 1), tf.argmax(prob_op, 1), name='accuracy')
    acc_tr_op = tf.summary.scalar('acc_tr', acc_op)
    acc_va_op = tf.summary.scalar('acc_va', acc_op)
    loss_tr_op = tf.summary.scalar('loss_tr', loss_val / x_tr.shape[0])
    loss_va_op = tf.summary.scalar('loss_va', loss_val / x_va.shape[0])
    metric_vars = [i for i in tf.local_variables() if i.name.split('/')[0] == 'accuracy']
    reset_op = [tf.variables_initializer(metric_vars), loss_val.assign(0.0)]
    loss_upd = loss_val.assign_add(tf.reduce_sum(loss_op))
    smry_tr = tf.summary.merge([acc_tr_op, loss_tr_op])

    smry_va = tf.summary.merge([acc_va_op, loss_va_op])
    config = tf.ConfigProto(intra_op_parallelism_threads=2,
                            use_per_session_threads=True,
                            gpu_options = tf.GPUOptions(allow_growth=True)
                            )
    with tf.Session(config=config) as sess:
        log = tf.summary.FileWriter(logdir, sess.graph)
        saver = tf.train.Saver(max_to_keep=100)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        fd_tr = {is_training: True}

        for cur_epoch in range(n_epochs):
            # get stats over whole training set
            for fd in utils.generate_minibatches(batch_size, [x, y], [x_tr, y_tr], feed_dict=fd_tr, shuffle=False):
                sess.run([acc_upd, loss_upd], feed_dict=fd)
            log.add_summary(sess.run(smry_tr, feed_dict=fd), cur_epoch)
            sess.run(reset_op)

            # training
            for fd in utils.generate_minibatches(batch_size, [x, y], [x_tr, y_tr], feed_dict=fd_tr):
                sess.run([train_op], feed_dict=fd)

            # validation
            for fd in utils.generate_minibatches(batch_size, [x, y], [x_va, y_va], shuffle=False):
                sess.run([acc_upd, loss_upd], feed_dict=fd)
            smry, acc = sess.run([smry_va, acc_op])
            log.add_summary(smry, cur_epoch)
            sess.run(reset_op)
            print("%3d: %3.3f" % (cur_epoch, acc), flush=True)

            if cur_epoch % 250 == 0 and cur_epoch > 0:
                saver.save(sess, os.path.join(logdir, 'model'), global_step=cur_epoch)


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-n", "--nhidden", type=int, help='hidden units (-1: use input size)', default=-1)
parser.add_argument("-d", "--depth", type=int, help='number of hidden layers', default=3)
parser.add_argument("-a", "--activation",  choices=['relu', 'selu', 'tanh'], default='relu')
parser.add_argument("-b", "--batchsize", type=int, help='batch size', default=128)
parser.add_argument("-e", "--epochs", type=int, help='number of training epochs', default=30)
parser.add_argument("-l", "--learningrate", type=float, help='learning rate', default=1e-5)
parser.add_argument("-g", "--gpuid", type=str, help='GPU to use (leave blank for CPU only)', default="")
parser.add_argument("--batchnorm", help='use batchnorm', action="store_true")
parser.add_argument("--dropout", type=float, help='hidden dropout rate (implies input-dropout of 0.2)', default=0.0)
parser.add_argument("--dataset", type=str, help='name of dataset', default='mnist_bgimg')
parser.add_argument("--logdir", type=str, help='directory for TF logs and summaries', default="/publicwork/tom/selfregularizing_nets/logs")

# by parsing the arguments already, we can bail out now instead of waiting
# for TF to load, in case the arguments aren't ok
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import clip_ops

logdir_base = os.getcwd()
run(args.depth, args.nhidden, args.epochs, args.learningrate, args.dataset,
    args.activation, args.logdir, args.batchsize, args.dropout, args.batchnorm)
