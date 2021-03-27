import pickle
import numpy as np
from absl import app
from absl import flags
import pandas as pd

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp
tfd = tfp.distributions

import matplotlib.pyplot as plt
from scipy.stats import norm

from sklearn.model_selection import train_test_split 

class LVClassifier():
    def __init__(self, name_id, seed=None):
        self.name_id = name_id
        self.seed = seed

    def create_train_val(self, num_bins, idx, train_size=0.8, seed=36, strat=False):

        self.val_ts_file = "using split on train_ts_file"
        self.val_theta_file = "using split on train_thetas_file"

        if strat:
            all_theta_cut = pd.qcut(self.train_thetas[:,idx], q=num_bins)
        else:
            all_theta_cut = pd.cut(self.train_thetas[:,idx], bins=num_bins)

        bins = all_theta_cut.categories
        all_ = all_theta_cut.rename_categories(range(num_bins)).to_numpy()
        all_ = tf.keras.utils.to_categorical(all_)

        dummy = list(range(len(self.train_thetas)))
        self.train_ts, self.val_ts, self.train_thetas, self.val_thetas = train_test_split(self.train_ts, self.train_thetas, train_size=train_size, random_state=seed)
        _, _, train_, val_ = train_test_split(dummy, all_, train_size=train_size, random_state=seed)

        return train_, val_, bins
    
    def create_train_val_2D(self, num_bins, idx, train_size=0.8, seed=36):
        """
        idx      = [idx1, idx2] which 2 dimensions to map.
        num_bins = number of bins per parameter, total bins = num_bins^2 
        """
        df = pd.DataFrame(dict(A=self.train_thetas[:,idx[0]], B=self.train_thetas[:,idx[1]]))
        d1 = df.assign(
            A_cut=pd.cut(df.A, num_bins),
            B_cut=pd.cut(df.B, num_bins)
        )
        d2 = d1.assign(
            A_label=pd.Categorical(d1.A_cut).rename_categories(range(num_bins)),
            B_label=pd.Categorical(d1.B_cut).rename_categories(range(num_bins))
        )
        d3 = d2.assign(cartesian_label=pd.Categorical(d2.filter(regex='_label').apply(tuple, 1)))
        d4 = d3.assign(cartesian_cut=pd.Categorical(d3.filter(regex='_cut').apply(tuple, 1)))
        labels = pd.Categorical(d4.cartesian_label).rename_categories(range(len(pd.Categorical(d4.cartesian_label).categories))).to_numpy()
        all_ = tf.keras.utils.to_categorical(labels)
        d5 = d4.assign(label=labels)
        bins_ = pd.Categorical(d5.cartesian_cut).categories
        
        dummy = list(range(len(self.train_thetas)))

        self.train_ts, self.val_ts, self.train_thetas, self.val_thetas = train_test_split(self.train_ts, self.train_thetas, train_size=train_size, random_state=seed)
        _, _, train_, val_ = train_test_split(dummy, all_, train_size=train_size, random_state=seed)
        
        return train_, val_, bins_

    def reshape_data(self, data):
        data = np.transpose(data, (0,2,1))
        return data

    def catogorize_thetas(self, num_bins, idx, strat=False):
        if strat:
            train_thetas_cut = pd.qcut(self.train_thetas[:,idx], q=num_bins)
            bins = train_thetas_cut.categories
            #test_thetas_cut = pd.cut(self.test_thetas[:,idx], bins)
            validation_thetas_cut = pd.qcut(self.val_thetas[:,idx], q=bins)
            train_ = train_thetas_cut.rename_categories(range(num_bins)).to_numpy()
            #test_ = test_thetas_cut.rename_categories(range(num_bins)).to_numpy()
            validation_ = validation_thetas_cut.rename_categories(range(num_bins)).to_numpy()
            #make K scheme, one-hot representation
            train_ = tf.keras.utils.to_categorical(train_)
            #test_ = tf.keras.utils.to_categorical(test_)
            validation_ = tf.keras.utils.to_categorical(validation_)

            test_ = [0]
        else:
            train_thetas_cut = pd.cut(self.train_thetas[:,idx], bins=num_bins)
            bins = train_thetas_cut.categories
            #test_thetas_cut = pd.cut(self.test_thetas[:,idx], bins)
            validation_thetas_cut = pd.cut(self.val_thetas[:,idx], bins)

            train_ = train_thetas_cut.rename_categories(range(num_bins)).to_numpy()
            #test_ = test_thetas_cut.rename_categories(range(num_bins)).to_numpy()
            validation_ = validation_thetas_cut.rename_categories(range(num_bins)).to_numpy()

            #make K scheme, one-hot representation
            train_ = tf.keras.utils.to_categorical(train_)
            test_ = [0]
            validation_ = tf.keras.utils.to_categorical(validation_)
       
        return train_, test_, validation_, bins

    def construct_small(self, input_shape, output_shape, NUM_TRAIN_EXAMPLES, pooling_len=5):

        poolpadding = 'valid'
        pool = tf.keras.layers.MaxPooling1D
        
        kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                  tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))
        
        model_in = tf.keras.layers.Input(shape=input_shape)
        conv_1 = tfp.layers.Convolution1DFlipout(20, kernel_size=3, padding="same", strides=1,
                                                 kernel_divergence_fn=kl_divergence_function,
                                                 activation=tf.nn.relu)
        x = conv_1(model_in)
               
        
        x = pool(pooling_len, padding=poolpadding)(x)
        x = tf.keras.layers.Flatten()(x)
        
            
        dense = tfp.layers.DenseFlipout(output_shape, kernel_divergence_fn=kl_divergence_function,
                                        activation=tf.nn.softmax)
        
        model_out = dense(x)
        model = tf.keras.Model(model_in, model_out)
        
        return model
    
    def construct_large(self, input_shape, output_shape, NUM_TRAIN_EXAMPLES, pooling_len=3):

        poolpadding = 'valid'

        pool = tf.keras.layers.MaxPooling1D

        kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                  tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))

        model_in = tf.keras.layers.Input(shape=input_shape)
        conv_1 = tfp.layers.Convolution1DFlipout(100, kernel_size=3, padding="same", strides=1,
                                                 kernel_divergence_fn=kl_divergence_function,
                                                 activation=tf.nn.relu)
        x = conv_1(model_in)
        x = pool(pooling_len, padding=poolpadding)(x)

        conv_2_1 = tfp.layers.Convolution1DFlipout(50, kernel_size=3, padding="same", strides=1,
                                                 kernel_divergence_fn=kl_divergence_function,
                                                 activation=tf.nn.relu)#, data_format='channels_first')
        x = conv_2_1(x)
        x = pool(pooling_len, padding=poolpadding)(x)

        conv_2_2 = tfp.layers.Convolution1DFlipout(50, kernel_size=3, padding="same", strides=1,
                                                 kernel_divergence_fn=kl_divergence_function,
                                                 activation=tf.nn.relu)
        x = conv_2_2(x)
        x = pool(pooling_len, padding=poolpadding)(x)


        conv_3 = tfp.layers.Convolution1DFlipout(25, kernel_size=3, padding="same", strides=1,
                                                 kernel_divergence_fn=kl_divergence_function,
                                                 activation=tf.nn.relu)

        x = conv_3(x)
        x = tf.keras.layers.Flatten()(x)

        dense_1_1 = tfp.layers.DenseFlipout(50, kernel_divergence_fn=kl_divergence_function)
        x = dense_1_1(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        dense_1_2 = tfp.layers.DenseFlipout(50, kernel_divergence_fn=kl_divergence_function)
        x = dense_1_2(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        dense_1_3 = tfp.layers.DenseFlipout(50, kernel_divergence_fn=kl_divergence_function)
        x = dense_1_3(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        dense_1_4 = tfp.layers.DenseFlipout(50, kernel_divergence_fn=kl_divergence_function)
        x = dense_1_4(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        dense_2 = tfp.layers.DenseFlipout(output_shape, kernel_divergence_fn=kl_divergence_function,
                                          activation=tf.nn.softmax)
        #dense_3 = tfp.layers.DenseVariational(output_shape, activation=None)
        model_out = dense_2(x)
        model = tf.keras.Model(model_in, model_out)
        return model

    def train(self, num_bins=10, batch_size=32, split=False, verbose=True, use_small=False, multi_dim=False):
        np.random.seed(self.seed)

        tf.keras.backend.clear_session()
        tf.keras.backend.set_floatx('float32')

      
        self.num_bins = num_bins
            
        if use_small:
            construct_BayesClassifier = self.construct_small
        else:
            construct_BayesClassifier = self.construct_large

        
            

       

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', verbose=1, min_delta=0.001,
                                              patience=5)

        # Model compilation.
        optimizer = tf.keras.optimizers.Adam(0.001)
        # We use the categorical_crossentropy loss since the MNIST dataset contains
        # ten labels. The Keras API will then automatically add the
        # Kullback-Leibler divergence (contained on the individual layers of
        # the model), to the cross entropy loss, effectively
        # calcuating the (negated) Evidence Lower Bound Loss (ELBO)

        loss = 'categorical_crossentropy'
        #loss = 'binary_crossentropy'

        #loss = self.NLL_logits
        
        
        

        
        if multi_dim:
            # param k1 x k2
            _train, _val, _bins = self.create_train_val_2D(num_bins=num_bins, idx=[0,1])
            self.multidim_bins_rate12 = _bins
            model1_c = construct_BayesClassifier((self.train_ts.shape[1],2), len(self.multidim_bins_rate12),
                                                 len(self.train_ts))
            model1_c.compile(optimizer, loss=loss,
                         metrics=['accuracy'], experimental_run_tf_function=False)
            
            if verbose:
                print("model 1")
                model1_c.summary()
                print(f'len(_bins): {len(_bins)}')
            
            model1_c.fit(self.train_ts, _train, batch_size=batch_size, epochs=1000, verbose=verbose,
                         validation_freq=1, validation_data=(self.val_ts, _val),
                         callbacks=[es])
                                                           
            # param k1 x k3
            _train, _val, _bins = self.create_train_val_2D(num_bins=num_bins, idx=[0,2])
            self.multidim_bins_rate13 = _bins
            model2_c = construct_BayesClassifier((self.train_ts.shape[1],2), len(self.multidim_bins_rate13),
                                                 len(self.train_ts))
            model2_c.compile(optimizer, loss=loss,
                         metrics=['accuracy'], experimental_run_tf_function=False)
            model2_c.fit(self.train_ts, _train, batch_size=batch_size, epochs=1000, verbose=verbose,
                         validation_freq=1, validation_data=(self.val_ts, _val),
                         callbacks=[es])
                                                           
            # param k2 x k3
            _train, _val, _bins = self.create_train_val_2D(num_bins=num_bins, idx=[1,2])
            self.multidim_bins_rate23 = _bins
            model3_c = construct_BayesClassifier((self.train_ts.shape[1],2), len(self.multidim_bins_rate23), 
                                                 len(self.train_ts))
            model3_c.compile(optimizer, loss=loss,
                         metrics=['accuracy'], experimental_run_tf_function=False)
            model3_c.fit(self.train_ts, _train, batch_size=batch_size, epochs=1000, verbose=verbose,
                         validation_freq=1, validation_data=(self.val_ts, _val),
                         callbacks=[es])
        else:
            # param k1
            if split:
                _train, _val, _bins = self.create_train_val(num_bins=num_bins, idx=0)
            else:
                _train, _test, _val, _bins = self.catogorize_thetas(num_bins=num_bins, idx=0)
        
            self.bins_rate1 = _bins    
            model1_c = construct_BayesClassifier((self.train_ts.shape[1],2), self.num_bins, len(self.train_ts))
            model1_c.compile(optimizer, loss=loss,
                         metrics=['accuracy'], experimental_run_tf_function=False)
            model1_c.fit(self.train_ts, _train, batch_size=batch_size, epochs=1000, verbose=verbose,
                         validation_freq=1, validation_data=(self.val_ts, _val),
                         callbacks=[es])
            if split:
                _train, _val, _bins = self.create_train_val(num_bins=num_bins, idx=1)
            else:
                _train, _test, _val, _bins = self.catogorize_thetas(num_bins=num_bins, idx=1)

            # param k2
            self.bins_rate2 = _bins
            model2_c = construct_BayesClassifier((self.train_ts.shape[1],2), self.num_bins, len(self.train_ts))
            model2_c.compile(optimizer, loss=loss,
                         metrics=['accuracy'], experimental_run_tf_function=False)
            model2_c.fit(self.train_ts, _train, batch_size=batch_size, epochs=1000, verbose=verbose,
                         validation_freq=1, validation_data=(self.val_ts, _val),
                         callbacks=[es])
            if split:
                _train, _val, _bins = self.create_train_val(num_bins=num_bins, idx=2)
            else:
                _train, _test, _val, _bins = self.catogorize_thetas(num_bins=num_bins, idx=2)
        
             # param k3

            self.bins_rate3 = _bins
            model3_c = construct_BayesClassifier((self.train_ts.shape[1],2), self.num_bins, len(self.train_ts))   
            model3_c.compile(optimizer, loss=loss,
                         metrics=['accuracy'], experimental_run_tf_function=False)
            model3_c.fit(self.train_ts, _train, batch_size=batch_size, epochs=1000, verbose=verbose,
                         validation_freq=1, validation_data=(self.val_ts, _val),
                         callbacks=[es])
        if verbose:
            print("model 1")
            model1_c.summary()
            print("model 2")
            model2_c.summary()
            print("model 3")
            model3_c.summary()
        return model1_c, model2_c, model3_c
    

    def MCinferC(self, data, model, num_monte_carlo, verbose=True):
        # Compute log prob of heldout set by averaging draws from the model:
        # p(heldout | train) = int_model p(heldout|model) p(model|train)
        #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
        # where model_i is a draw from the posterior p(model|train).
        if verbose:
            print(' ... Running monte carlo inference')
        probs = tf.stack([model.predict(data, verbose=0)
                        for _ in range(num_monte_carlo)], axis=0)
        mean_probs = tf.reduce_mean(probs, axis=0)
        heldout_log_prob = tf.reduce_mean(tf.math.log(mean_probs))
        if verbose:
            print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))
        return probs


    def run(self, target, batch_size=132, num_bins = 10, num_monte_carlo=500, split=False, toload = True, verbose=True, plot=False, use_small=False, multi_dim=False):
        if toload:
            self.load_data()
            self.train_ts = self.reshape_data(self.train_ts)
            self.test_ts = self.reshape_data(self.test_ts)
            self.val_ts = self.reshape_data(self.val_ts)
        model1, model2, model3 = self.train(batch_size=batch_size, num_bins=num_bins, split=split, verbose=verbose,use_small=use_small, multi_dim=multi_dim)
        #self.probs1 = self.MCinferC(target[:,:,1], model1, num_monte_carlo)
        self.probs1 = self.MCinferC(target, model1, num_monte_carlo)
        self.probs2 = self.MCinferC(target, model2, num_monte_carlo)
        self.probs3 = self.MCinferC(target, model3, num_monte_carlo)
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        
def sample_local(probs, bins, num_samples=1000, use_thresh=False, thresh=0.1):
    probs_pred = tf.reduce_mean(probs, axis=0).numpy()
    if use_thresh:
        probs_pred[probs_pred < thresh] = 0.0
    #print(probs_pred)
    dist = tfd.Categorical(
    logits=None, probs=probs_pred, dtype=tf.int32, validate_args=False,
    allow_nan_stats=False, name='Categorical')

    samples_bins = dist.sample(num_samples)
    samples = np.empty((num_samples))
    for e,i in enumerate(samples_bins):
        interval = bins[i.numpy()[0]]
        u = np.random.uniform(interval.left, interval.right)
        samples[e] = u
    return samples, samples_bins.numpy()
                                                          
def sample_local_2D(probs, bins, num_samples=1000, use_thresh=False, thresh=0.1):
    probs_pred = tf.reduce_mean(probs, axis=0).numpy()
    if use_thresh:
        probs_pred[probs_pred < thresh] = 0.0
    print('predictive posterior:')
    print(probs_pred)
    dist = tfd.Categorical(
    logits=None, probs=probs_pred, dtype=tf.int32, validate_args=False,
    allow_nan_stats=False, name='Categorical')

    samples_bins = dist.sample(num_samples)
    print(samples_bins.shape)
    samples = np.empty((num_samples,2))
    for e,i in enumerate(samples_bins):
        interval = bins[i.numpy()[0]]
        u1 = np.random.uniform(interval[0].left, interval[0].right)
        u2 = np.random.uniform(interval[1].left, interval[1].right)
        samples[e] = np.array([u1,u2])
    return samples, samples_bins.numpy()
                                                      

# Given bins, find data inside of bins
def inBin(data, thetaOld, thetaNew):
    flag = None
    for j in range(thetaNew.shape[1]):
        flag_max = (thetaOld[:,j] < max(thetaNew[:,j]))
        flag_min = (thetaOld[:,j] > min(thetaNew[:,j]))
        if j > 0:
            flag = flag & flag_max & flag_min
        else:
            flag = flag_max & flag_min
  
    return data[flag,:,:], thetaOld[flag,:]
