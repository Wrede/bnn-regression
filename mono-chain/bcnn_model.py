from sklearn.model_selection import train_test_split
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import pandas as pd

def MCinferMOG(data, model, num_monte_carlo, output_dim):
        m = np.empty((num_monte_carlo,1,output_dim))
        S = np.empty((num_monte_carlo,1,output_dim,output_dim))

        # Take samples from posterior predictive, one sample is a Gaussian
        for i in range(num_monte_carlo):
            normal = model(data)
            m[i] = normal.mean()
            S[i] = normal.covariance()

        m_hat = m.mean(axis=0) #mean of a mixture of gaussians with equal weight
        S_hat_mean = S.mean(axis=0)

        #not efficent, avoid loop
        # following corresponds to:
        # (m_i - \hat{m})(m_i - \hat{m})^T
        matmul = np.empty((num_monte_carlo, 1, output_dim,output_dim))
        for i in range(num_monte_carlo):
            inner = m[i] - m_hat
            matmul[i] = np.matmul(inner.T, inner)

        #1/N \sum_{i=1}^N S_i + (m_i - \hat{m})(m_i - \hat{m})^T
        S_hat = S_hat_mean + matmul.mean(axis=0) #Covariance of a mixture of gaussians

        return m_hat, S_hat

def MCinfer(data, model, num_monte_carlo):
        # draws sample parameters directly, i.e not returning a gaussian
        # Compute log prob of heldout set by averaging draws from the model:
        # p(heldout | train) = int_model p(heldout|model) p(model|train)
        #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
        # where model_i is a draw from the posterior p(model|train).
        print(' ... Running monte carlo inference')
        probs = tf.stack([model.predict(data, verbose=0)
                        for _ in range(num_monte_carlo)], axis=0)
        mean_probs = tf.reduce_mean(probs, axis=0)
        print(f'Sum probs mean: {np.sum(mean_probs)}')
        heldout_log_prob = tf.reduce_mean(tf.math.log(mean_probs))
        print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))
        return probs

class Regressor():
    def __init__(self, name_id):
        self.name_id = name_id

    
    def construct_small(self, input_shape, output_shape, NUM_TRAIN_EXAMPLES, pooling_len=10):
        """BCNN used in manuscript"""
        
        poolpadding = 'valid'
        pool = tf.keras.layers.MaxPooling1D
        
        kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /
                                  tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))
        
        model_in = tf.keras.layers.Input(shape=input_shape)
        conv_1 = tfp.layers.Convolution1DFlipout(25, kernel_size=5, padding="same", strides=1,
                                                 kernel_divergence_fn=kl_divergence_function,
                                                 activation=tf.nn.relu)
        x = conv_1(model_in)
        
        x = pool(pooling_len, padding=poolpadding)(x)
        
        conv_1 = tfp.layers.Convolution1DFlipout(6, kernel_size=5, padding="same", strides=1,
                                                 kernel_divergence_fn=kl_divergence_function,
                                                 activation=tf.nn.relu)
        x = conv_1(x)
        
        x = pool(pooling_len, padding=poolpadding)(x)
        
        x = tf.keras.layers.Flatten()(x)
        
        
        dense = tfp.layers.DenseFlipout(20, kernel_divergence_fn=kl_divergence_function,
                                        activation=None)
        
        x = dense(x)
        
        dense = tfp.layers.DenseFlipout(output_shape, kernel_divergence_fn=kl_divergence_function,
                                        activation=None)
        
        x = dense(x)


        #Adding a multivariate Gaussian
        param_size = tfp.layers.MultivariateNormalTriL.params_size(output_shape)
        dense = tf.keras.layers.Dense(param_size, activation=None)
        x = dense(x)
        normal = tfp.layers.MultivariateNormalTriL(output_shape)
        model_out = normal(x)
        
        #If one like to use a MoG with 1 component instead
        
        #event_shape = [output_shape]
        #num_components = 1
        
        #params_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)
        
        #dense = tf.keras.layers.Dense(params_size,
        #                                activation=None)  
        #x = dense(x) 
        
        #mix = tfp.layers.MixtureNormal(num_components, event_shape)
        
        #model_out = mix(x)

        model = tf.keras.Model(model_in, model_out)
        
        return model


    def train(self, batch_size=32, epochs=400, pooling_len=10):
        tf.keras.backend.clear_session()
        tf.keras.backend.set_floatx('float32')

        output_dim = self.train_thetas.shape[-1]
        input_dim = self.train_ts.shape[1:]

        model = self.construct_small(input_dim, output_dim, len(self.train_ts), pooling_len)
        model.summary()

        # If one like to use early stopping
        #es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', min_delta=0.1, verbose=1,
        #                                      patience=5)


        # Model compilation
        negloglik = lambda y, rv_y: -rv_y.log_prob(y)
        optimizer = tf.keras.optimizers.Adam(0.001)
        #loss = 'mse'
        loss = negloglik
        model.compile(optimizer, loss=loss, #metrics=['mse'],
                         experimental_run_tf_function=False)
        
        model.fit(self.train_ts, self.train_thetas, batch_size=batch_size, epochs=epochs, verbose=True,
                     validation_freq=10, validation_data=(self.val_ts, self.val_thetas),#callbacks=[es]
                    )
        
        return model

    def retrain(self, batch_size=32, epochs=400):
        #es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='auto', min_delta=0, verbose=1,
        #                                      patience=10)
        self.model.fit(self.train_ts, self.train_thetas, batch_size=batch_size, epochs=epochs, verbose=True,
                     validation_freq=1, validation_data=(self.val_ts, self.val_thetas),#callbacks=[es]
                       )

    def run(self, target, batch_size=32, num_monte_carlo=500, epochs=100, retrain=False, infer_pdf=True, verbose=True,
           pooling_len=10):
        """"Construct and/or train/retrain model"""
        if retrain:
            self.retrain(batch_size=batch_size, epochs = epochs)
        else:
            self.model = self.train(batch_size=batch_size, epochs=epochs, pooling_len=pooling_len)
        if infer_pdf:
            output_dim = self.train_thetas.shape[-1]
            self.proposal_mean, self.proposal_covar = MCinferMOG(target, self.model, num_monte_carlo, output_dim)  
        else:
            self.probs = MCinfer(target, model1, num_monte_carlo)
      