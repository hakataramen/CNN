import tensorflow as tf
import numpy as np
import random
import pickle

import param
import preprocess


def print_param():
    with open('param.py', 'r') as f:
        print(f.read())

def dump_pickle(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj, f)

def chunked(iterable, n):
    chunk = [iterable[x:x + n] for x in range(0, len(iterable), n)]
    while len(chunk[-1]) != len(chunk[0]):
        chunk[-1].append(chunk[0][0])
    return chunk

def cnn(train, dev, W_e, train_new):
    train_x_ = train[0]
    train_y_ = train[1]
    tr_te = train[2]
    tr_wp = train[4]

    test_x = dev[0]
    test_y = dev[1]
    dev_te = dev[2]
    dev_eval = dev[3]
    dev_wp1 = dev[4][0]
    dev_wp2 = dev[4][1]

    chunk = int(len(test_x)/param.NUM_MINI_BATCH)
    test_x_chunked = chunked(test_x, param.NUM_MINI_BATCH)
    test_y_chunked = chunked(test_y, param.NUM_MINI_BATCH)
    dev_te_chunked = chunked(dev_te, param.NUM_MINI_BATCH)
    dev_eval_chunked = chunked(dev_eval, param.NUM_MINI_BATCH)
    dev_wp1_chunked = chunked(dev_wp1, param.NUM_MINI_BATCH)
    dev_wp2_chunked = chunked(dev_wp2, param.NUM_MINI_BATCH)


    R_denom = 0.0
    for i in test_y:
        if i[0] != 1:
            R_denom += 1.0

    max_len = len(train_x_[0])
    num_classes = len(train_y_[0])

    x = tf.placeholder(tf.int32, shape=[param.NUM_MINI_BATCH, max_len])
    y_ = tf.placeholder(tf.float32, shape=[param.NUM_MINI_BATCH, num_classes])
    x_p1 = tf.placeholder(tf.int32, shape=[param.NUM_MINI_BATCH, max_len])
    x_p2 = tf.placeholder(tf.int32, shape=[param.NUM_MINI_BATCH, max_len])
    t_e = tf.placeholder(tf.int32, shape=[param.NUM_MINI_BATCH, 2])

    # Define Word Embedding layer
    w_e = np.array(W_e)
    w_init = tf.placeholder(tf.float32, shape=w_e.shape)
    w = tf.Variable(w_init, trainable=True, name='w')
    w = tf.nn.l2_normalize(w,-1)
    w_pad = tf.Variable(tf.constant(0.0, shape=[1,param.EMBEDDING_SIZE]),trainable=param.PAD_TRAINABLE, name='w_pad')
    w_con = tf.concat([w_pad,w],0)
    e = tf.nn.embedding_lookup(w_con, x)

    w_pe = tf.Variable(tf.random_uniform(shape=[max_len*2, param.POSITION_EMBEDDING_SIZE], minval=-param.EMBEDDING_RANGE, maxval=param.EMBEDDING_RANGE),trainable=True)
    w_pe = tf.nn.l2_normalize(w_pe,-1)
    w_pe = w_pe / float(param.EMBEDDING_SIZE/param.POSITION_EMBEDDING_SIZE)
    #w_pe = tf.Variable(tf.random_normal(shape=[max_len*2, param.POSITION_EMBEDDING_SIZE]), trainable=True)
    w_pe_pad = tf.Variable(tf.constant(0.0, shape=[1,param.POSITION_EMBEDDING_SIZE]),trainable=param.PAD_TRAINABLE)
    w_pe_con = tf.concat([w_pe_pad,w_pe],0)
    pe1 = tf.nn.embedding_lookup(w_pe_con, x_p1)
    pe2 = tf.nn.embedding_lookup(w_pe_con, x_p2)

    e_all = tf.concat([e, pe1, pe2],2)
    emb_size = param.EMBEDDING_SIZE+2*param.POSITION_EMBEDDING_SIZE

    ex = tf.expand_dims(e_all, -1)
    pad = tf.zeros([param.NUM_MINI_BATCH, 1, emb_size])

    p_array = []
    phase_train = tf.placeholder(tf.bool)
    for filter in param.FILTER_SIZES:
        e_pad = e_all
        for i in range(int((filter-1)/2)):
            e_pad = tf.concat([pad, e_pad, pad],1)
        if filter%2 == 0:
            e_pad = tf.concat([e_pad, pad],1)
        ex = tf.expand_dims(e_pad, -1)
        W = tf.Variable(
            tf.truncated_normal(
                [filter, emb_size, 1, param.NUM_FILTERS], stddev=0.02))
        b = tf.Variable(tf.constant(0.1, shape=[param.NUM_FILTERS]))
        c0 = tf.nn.conv2d(ex, W, [1, 1, 1, 1], 'VALID')
        c1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.bias_add(c0, b), is_training=phase_train))

        c2 = tf.nn.max_pool(c1, [1, max_len, 1, 1], [1, 1, 1, 1], 'VALID')
        p_array.append(c2)
    p = tf.concat(p_array,3)
    total_filters = param.NUM_FILTERS * len(param.FILTER_SIZES)
    w1 = tf.Variable(tf.truncated_normal([total_filters, num_classes], stddev=0.02))
    b1 = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    keep = tf.placeholder(tf.float32)
    h0 = tf.nn.dropout(tf.reshape(p, [-1, total_filters]), keep)
    predict_y = tf.nn.softmax(tf.matmul(h0, w1) + b1)
    cross_entropy = -tf.reduce_sum(y_*tf.log(predict_y,1))
    cross_entropy = cross_entropy + param.L2_LAMDA * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(W) + tf.nn.l2_loss(w_con) + tf.nn.l2_loss(w_pe_con))

    train_step = tf.train.AdamOptimizer(param.LEARNING_RATE).minimize(cross_entropy)
    #train_step = tf.train.GradientDescentOptimizer(param.LEARNING_RATE).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(predict_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    restore = 1
    sess.run(tf.global_variables_initializer(), feed_dict={w_init: w_e})

    len_test= len(test_x)
    len_chunk = len(test_x_chunked)

    for epoch in range(param.NUM_ITERATIONS):
        random_indice = np.random.permutation(len(train_x_))
        i=0
        while(1):
            j=0
            mini_batch_x = []
            mini_batch_y = []
            mini_batch_te = []
            mini_batch_x_p1 = []
            mini_batch_x_p2 = []
            while(j!=param.NUM_MINI_BATCH):
                if i >= len(train_x_):
                    break
                nega_flag = False
                if train_y_[random_indice[i]][0] == 1:
                    rand = random.randint(0,param.NEGA)
                    if rand != 0:
                        nega_flag = True
                if nega_flag:
                    pass
                else:
                    mini_batch_x.append(train_x_[random_indice[i]])
                    mini_batch_y.append(train_y_[random_indice[i]])
                    mini_batch_te.append(tr_te[random_indice[i]])
                    mini_batch_x_p1.append(tr_wp[0][random_indice[i]])
                    mini_batch_x_p2.append(tr_wp[1][random_indice[i]])
                    j+=1
                i+=1
            if i >= len(train_x_):
                break
            v0, v1, v2 = sess.run([train_step, cross_entropy, accuracy], feed_dict={x: mini_batch_x, y_: mini_batch_y, t_e: mini_batch_te, x_p1: mini_batch_x_p1, x_p2: mini_batch_x_p2, keep: param.KEEP, phase_train: True})

        # My Evaluation
        P_numer = 0.0
        P_denom = 0.0
        for i in range(len_chunk):
            check1, acc, correct, predict_y_index = sess.run([d_d, accuracy, correct_prediction, tf.argmax(predict_y, 1)], feed_dict={x: test_x_chunked[i], y_: test_y_chunked[i], t_e: dev_te_chunked[i], x_p1: dev_wp1_chunked[i], x_p2: dev_wp2_chunked[i], keep: 1.0, phase_train: True})

            for j in range(len(predict_y_index)):
                if i == len_chunk - 1:
                    if j+1 > len_test % param.NUM_MINI_BATCH:
                        break
                if predict_y_index[j] != 0:
                    P_denom += 1.0
                    if correct[j]:
                        P_numer += 1.0

        P, R = P_numer / (P_denom+0.00001), P_numer / (R_denom+0.00001)
        F = 2 * P * R / (P + R + 0.00001)
        print('EPOCH: %d, P/R/F=%f/%f/%f' % (epoch+1, P, R, F))
        #print(check1)
        with open('log','a') as f:
            f.write(str(epoch+1)+'\t'+str(round(P,4))+'\t'+str(round(R,4))+'\t'+str(round(F,4))+'\n')


if __name__ == '__main__':
    print_param()
    train, dev, W_e, train_new = preprocess.main()
    cnn(train, dev, W_e, train_new)
