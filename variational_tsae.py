import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import fully_connected
from tsae import TSAutoEncoder
from utils import l1_l2_regularizer


class LatentTSAE(TSAutoEncoder):
    '''
    This is an effort to extend https://github.com/RobRomijnders/AE_ts to support multiple features (both continuous
    and categorical) per time step.
    '''
    def _build_encoder(self):
        with tf.variable_scope('Encoder'):
            cell = tf.contrib.rnn.LSTMCell(self.config.hidden_size, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

            self.stacked_lstm = cell = tf.contrib.rnn.MultiRNNCell([cell] * self.config.num_layers,
                                                                   state_is_tuple=True)

            initial_state_enc = cell.zero_state(self.config.batch_size, tf.float32)

            inputs = tf.unstack(self.inputs, num=self.config.num_steps, axis=1)
            # outputs = [Tensor[batch_size, hidden_size]] of length num_steps
            outputs_enc, _ = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, initial_state_enc, cell)

            cell_output = outputs_enc[-1]

        return cell_output

    def _build_decoder(self, z):
        with tf.variable_scope('Decoder'):
            cell_dec = tf.contrib.rnn.LSTMCell(self.config.hidden_size, state_is_tuple=True)
            cell_dec = tf.contrib.rnn.DropoutWrapper(cell_dec, output_keep_prob=self.keep_prob)

            self.stacked_lstm_dec = cell_dec = tf.contrib.rnn.MultiRNNCell([cell_dec] * self.config.num_layers,
                                                                           state_is_tuple=True)

            initial_state_dec = tuple([tf.contrib.rnn.LSTMStateTuple(z, z)] * self.config.num_layers)

            dec_inputs = [tf.zeros([tf.shape(self.inputs)[0], self.feature_dim])] * self.config.num_steps

            outputs_dec, _ = tf.contrib.legacy_seq2seq.rnn_decoder(dec_inputs, initial_state_dec, cell_dec)

        return outputs_dec

    def _build_latent_loss(self):

        with tf.variable_scope('Loss'):
            lat_mean, lat_var = tf.nn.moments(self.z, axes=[1])
            latent_loss = tf.square(lat_mean) + lat_var - tf.log(lat_var) - 1

        return latent_loss

    def _build_loss(self, outputs_dec):
        with tf.variable_scope('Loss'):

            latent_loss = self._build_latent_loss()

            self.reconstr_loss = tf.zeros(shape=[])
            output = tf.transpose(tf.stack(outputs_dec, axis=0), [1, 0, 2])  # [batch_size, num_steps, hidden_size]

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
                feature_target = tf.slice(self.inputs, [0, 0, feature_index], [-1, -1, feature_size])
                # print(feature_target.get_shape(), feature_output.get_shape())
                if feature_size == 1:
                    # MAE
                    self.reconstr_loss += tf.reshape(tf.sum(tf.abs(feature_output - feature_target), axis=1), [-1])
                else:
                    # Categorical features
                    # softmax_cross_entropy_with_logits shape: [batch_size, num_steps]
                    self.reconstr_loss += tf.reduce_sum(
                        tf.nn.softmax_cross_entropy_with_logits(feature_output, feature_target), axis=1)

                feature_index += feature_size

            self.reconstruction = tf.stack(features, axis=2)

        loss = tf.reduce_mean(self.reconstr_loss + latent_loss)

        # if self.config.scale_l1 > 0 and self.config.scale_l2 > 0:
        #     weights = tf.trainable_variables()
        #     reg = l1_l2_regularizer(scale_l1=self.config.scale_l1, scale_l2=self.config.scale_l2)
        #     print [weights[i].name for i, p in enumerate([reg(w) for w in weights]) if p.get_shape().ndims != 0]
        #     penalty = tf.contrib.layers.apply_regularization(reg, weights)
        #     loss = loss + penalty

        self._cost = loss

        tf.summary.scalar("latent loss", tf.reduce_mean(latent_loss))
        tf.summary.scalar("reconstruction loss", tf.reduce_mean(self.reconstr_loss))
        tf.summary.scalar("overall loss", self._cost)

    def _build_latent(self, encoder_out):
        with arg_scope([fully_connected], activation_fn=None,
                       biases_initializer=tf.constant_initializer(0.1),
                       weights_initializer=tf.contrib.layers.xavier_initializer()):
            with tf.variable_scope('Enc_2_lat'):
                self.z = fully_connected(inputs=encoder_out,
                                         num_outputs=self.config.latent_size)

            with tf.variable_scope('Lat_2_dec'):
                z_state = fully_connected(inputs=self.z,
                                          num_outputs=self.config.hidden_size)

        return z_state

    def _build_layers(self):

        encoder_out = self._build_encoder()

        z_state = self._build_latent(encoder_out)

        decoder_out = self._build_decoder(z_state)

        self._build_loss(decoder_out)

    def transform(self, X):
        return self.session.run(self.z, feed_dict={self.inputs: X, self.keep_prob: 1.0})

    def calc_restruct_cost(self, X):
        return self.session.run(self.reconstr_loss, feed_dict={self.inputs: X, self.keep_prob: 1.0})


class VariationalTSAE(LatentTSAE):
    def _build_latent(self, encoder_out):
        with tf.variable_scope('Enc_2_lat'):
            with arg_scope([fully_connected], activation_fn=tf.nn.relu,
                           biases_initializer=tf.constant_initializer(0.1),
                           weights_initializer=tf.contrib.layers.xavier_initializer()):
                self.z_mean = fully_connected(inputs=encoder_out,
                                              num_outputs=self.config.latent_size)

                self.z_log_sigma_sq = fully_connected(inputs=encoder_out,
                                                      num_outputs=self.config.latent_size,
                                                      activation_fn=tf.nn.relu)

            # sample from gaussian distribution
            eps = tf.random_normal(tf.stack([tf.shape(self.inputs)[0], self.config.latent_size]), 0, 1,
                                   dtype=tf.float32)
            self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        with tf.variable_scope('Lat_2_dec'):
            z_state = fully_connected(inputs=self.z,
                                      num_outputs=self.config.hidden_size,
                                      activation_fn=None,
                                      biases_initializer=tf.constant_initializer(0.1),
                                      weights_initializer=tf.contrib.layers.xavier_initializer())

        return z_state

    def _build_latent_loss(self):
        with tf.variable_scope('Loss'):
            latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                                              - tf.square(self.z_mean)
                                                              - tf.exp(self.z_log_sigma_sq), 1)

        return latent_loss
