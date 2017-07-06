import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import logging


class DataFeeder(object):
    def get_random_block_from_data(self, data, batch_size):
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index:(start_index + batch_size)]


class TSAutoEncoder(object):
    # This is to avoid tf var conflicts when we create many TSAutoEncoder instances
    i = 0

    def __init__(self, config, feeder, session=None):
        self.config = config
        self.num_features = len(self.config.feature_sizes)
        self.feature_dim = sum(self.config.feature_sizes)

        self.feeder = feeder

        self._build_graph()

        self._setup_tf(session)

        TSAutoEncoder.i += 1

    def _build_graph(self):
        with tf.variable_scope('TSAE_%d' % TSAutoEncoder.i):
            self._build_input()
            self._build_layers()
            self._build_train_op()
            self._build_summary_op()

    def _build_input(self):
        with tf.variable_scope('Input'):
            self.inputs = tf.placeholder(self._data_type(), [None, self.config.num_steps, self.num_features])
            self.keep_prob = tf.placeholder(self._data_type())
            self.inputs = tf.nn.dropout(self.inputs, self.keep_prob)

    def _build_layers(self):
        raise NotImplementedError('Must be defined by subclasses')

    def _build_train_op(self):
        self._lr = tf.Variable(0.0, trainable=False)
        tf.summary.scalar("learning_rate", self._lr)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), self.config.max_grad_norm)

        optimizer = tf.train.RMSPropOptimizer(self._lr)
        self._global_step = tf.contrib.framework.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                   global_step=self._global_step)

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def _build_summary_op(self):
        self._merged = tf.summary.merge_all()

    def _setup_tf(self, session):
        # This is for logging down the summaries for displaying in TensorBoard
        self._writer = tf.summary.FileWriter(self.config.log_dir, graph=tf.get_default_graph())

        # This is for checkpointing epoches
        self._saver = tf.train.Saver(tf.global_variables())

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) if session is None else session
        self.session.run(tf.global_variables_initializer())

    def _data_type(self):
        return tf.float32

    def _run_epoch(self, X):
        cost, summary, global_step, _ = self.session.run((self._cost, self._merged, self._global_step, self._train_op),
                                                         feed_dict={self.inputs: X,
                                                                    self.keep_prob: self.config.keep_prob})

        self._writer.add_summary(summary, global_step)

        return cost

    def fit(self, X, display_step=1):
        n_samples = len(X)
        for epoch in range(self.config.max_max_epoch):
            lr_decay = self.config.lr_decay ** max(epoch - self.config.max_epoch, 0.0)
            logging.debug("lr_decay = %s" % lr_decay)
            self.assign_lr(self.session, self.config.learning_rate * lr_decay)

            avg_cost = 0.
            total_batch = int(n_samples / self.config.batch_size)
            # Loop over batches
            for i in range(total_batch):
                batch_xs = self.feeder.get_random_block_from_data(X, self.config.batch_size)
                cost = self._run_epoch(batch_xs)
                avg_cost += cost / total_batch

            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

            # Save a checkpoint
            self._saver.save(self.session, self.config.log_dir + '/checkpoints-%d-%f' % (epoch, avg_cost))

    def calc_total_cost(self, X):
        return self.session.run(self._cost, feed_dict={self.inputs: X, self.keep_prob: 1.0})

    def transform(self, X):
        return self.session.run(self._final_state, feed_dict={self.inputs: X, self.keep_prob: 1.0})

    def reconstruct(self, X):
        return self.session.run(self.reconstruction, feed_dict={self.inputs: X, self.keep_prob: 1.0})


class SimpleTSAutoEncoder(TSAutoEncoder):
    def _build_layers(self):
        with tf.variable_scope('RNN'):
            cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

            self.stacked_lstm = cell = tf.contrib.rnn.MultiRNNCell([cell] * self.config.num_layers,
                                                                   state_is_tuple=True)

            inputs = tf.unstack(self.inputs, num=self.config.num_steps, axis=1)
            # outputs = [Tensor[batch_size, hidden_size]] of length num_steps
            outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, dtype=self._data_type())

        with tf.variable_scope('Loss'):
            loss = tf.zeros(shape=[])
            output = tf.transpose(tf.stack(outputs, axis=0), [1, 0, 2])  # [batch_size, num_steps, hidden_size]

            feature_sizes = self.config.feature_sizes
            # loop over features and cont
            feature_index = 0
            features = []
            for feature_size in feature_sizes:
                feature_output = fully_connected(inputs=output,
                                                 num_outputs=feature_size,
                                                 activation_fn=tf.nn.relu,
                                                 biases_initializer=tf.constant_initializer(0.1),
                                                 weights_initializer=tf.contrib.layers.xavier_initializer()
                                                 )
                feature_output = tf.reshape(feature_output, [-1, self.config.num_steps, feature_size])
                features.append(feature_output)
                feature_target = tf.slice(self.inputs, [0, 0, feature_index], [-1, -1, 1])
                if feature_size == 1:
                    loss += tf.nn.l2_loss(feature_output - feature_target)
                else:
                    # Categorical features
                    loss += tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(feature_output, feature_target))

                feature_index += feature_size

            self.reconstruction = tf.stack(features, axis=2)

        self._cost = loss
        tf.summary.scalar("loss", loss)
        self._final_state = state
