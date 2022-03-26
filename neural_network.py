import numpy as np
import tensorflow as tf
import matplotlib.pylot as plt

from util import get_normalized_data, y2indicator

def error_rate(p, t):
    return np.mean(p != t)


def main():
    X, Y = get_normalized_data()

    max_iter = 30
    print_period = 10
    lr = 0.00004
    reg = 0.01

    Xtrain = X[:-1000,]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:,]
    Ytest = Y[-1000:]
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N / batch_sz

    M1 = 300
    M2 = 100
    K = 10
    W1_init = np.random.randn(D, M1) / 28
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2) / np.sqrt(M1)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K) / np.sqrt(M2)
    b3_init = np.zeros(K)

    Z = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))

    LL = []
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)

        for i in xrange(max_iter):
            for j in xrange(n_batches):
                Xbatch = Xtrain[j*batch_sz:(j*batch_sz * batch_sz),]
                Ybatch = Ytrain[j*batch_sz:(j*batch_sz * batch_sz),]

                session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
                if j % print_period == 0:
                    test_cost = session.run(cost, feed_dict={X: Xtest, T: Ytest_ind})
                    prediction = session.run(predict_op, feed_dict={X: Xtest})
                    err = error_rate(prediction, Ytest)
                    print "Cost / err at iteration i=%d, j=%d: %.3f" % (i, j, test_cost, err)
                    LL.append(test_cost)
    plt.plot(LL)
    plt.show()


if __name__ == '__main__':
    main()