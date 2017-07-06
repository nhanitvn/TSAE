import numpy as np

import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tsae import SimpleTSAutoEncoder, DataFeeder
from variational_tsae import LatentTSAE, VariationalTSAE

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print mnist.train.images.shape, mnist.test.images.shape


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

# Reshape into shape [None, num_steps, num_features]
X_train = X_train.reshape([-1, 28, 28])
X_test = X_test.reshape([-1, 28, 28])


class TestConfig(object):
    learning_rate = 0.01
    max_grad_norm = 5
    num_layers = 1
    num_steps = 28
    hidden_size = 200
    latent_size = 20
    scale_l1 = 0.0,
    scale_l2 = 0.0,
    keep_prob = 0.5,
    max_epoch = 5
    max_max_epoch = 20
    lr_decay = 0.5
    batch_size = 20
    feature_sizes = [1] * 28
    log_dir = 'log_variational'


config = TestConfig()

n_samples = int(mnist.train.num_examples)
display_step = 1
random_seed = 123

with tf.Graph().as_default():
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)

    # tsae = SimpleTSAutoEncoder(config=config, feeder=DataFeeder())
    tsae = LatentTSAE(config=config, feeder=DataFeeder())
    # tsae = VariationalTSAE(config=config, feeder=DataFeeder())

    tsae.fit(X_train)

    print "Total cost: " + str(tsae.calc_total_cost(X_test))
