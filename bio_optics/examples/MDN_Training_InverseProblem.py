import numpy as np
import matplotlib.pyplot as plt
import math
import random
from keras import callbacks
import keras
import tensorflow as tf
from keras import layers
from keras import optimizers
from tensorflow_probability import distributions as tfd
from sklearn.model_selection import train_test_split  # was: sklearn.cross_validation

# Helper functions
from scipy.stats import norm as normal


def plot_normal_mix(pis, mus, sigmas, x, ax, label='', comp=True):
    """
    Plots the mixture of Normal models to axis=ax
    comp=True plots all components of mixture model
    """
    x = np.linspace(np.min(x), np.max(x), 250)
    final = np.zeros_like(x)
    for i, (weight_mix, mu_mix, sigma_mix) in enumerate(zip(pis, mus, sigmas)):
        temp = normal.pdf(x, mu_mix, sigma_mix) * weight_mix
        if not any(np.isnan(temp)):
            final = final + temp
            if comp:
                ax.plot(x, temp)  # label='Normal ' + str(i)
    # print(final)
    ax.plot(x, final, label='Mixture of Normals ' + label)
    ax.legend(fontsize=13)


def sample_from_mixture(x, pred_weights, pred_means, pred_std, amount):
    """
    Draws samples from mixture model.
    Returns 2 d array with input X and sample from prediction of Mixture Model
    """
    samples = np.zeros((amount, 2))
    n_mix = len(pred_weights[0])
    to_choose_from = np.arange(n_mix)
    for j, (weights, means, std_devs) in enumerate(zip(pred_weights, pred_means, pred_std)):
        index = np.random.choice(to_choose_from, p=weights)
        samples[j, 1] = normal.rvs(means[index], std_devs[index], size=1)
        samples[j, 0] = x[j]
        if j == amount - 1:
            break
    return samples


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def create_noisy_spiral(n, jitter_std=0.2, revolutions=2):
    angle = np.random.uniform(0, 2 * np.pi * revolutions, [n])
    r = angle

    x = r * np.cos(angle)
    y = r * np.sin(angle)

    result = np.stack([x, y], axis=1)
    result = result + np.random.normal(scale=jitter_std, size=[n, 2])
    result = 5 * normalize(result)
    return result

def build_toy_dataset(nsample=40000):
    y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, nsample))).T
    r_data = np.float32(np.random.normal(size=(nsample,1))) # random noise
    x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)
    return train_test_split(x_data, y_data, random_state=42, train_size=0.1)


def elu_plus_one_plus_epsilon(x):
    return keras.activations.elu(x) + 1 + keras.backend.epsilon()

class MixtureDensityOutput(layers.Layer):
    def __init__(self, output_dimension, num_mixtures, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        self.mdn_mus = layers.Dense(
            self.num_mix * self.output_dim, name="mdn_mus"
        )  # mix*output vals, no activation
        self.mdn_sigmas = layers.Dense(
            self.num_mix * self.output_dim,
            activation=elu_plus_one_plus_epsilon,
            name="mdn_sigmas",
        )  # mix*output vals exp activation
        self.mdn_pi = layers.Dense(self.num_mix, name="mdn_pi")  # mix vals, logits

    def build(self, input_shape):
        self.mdn_mus.build(input_shape)
        self.mdn_sigmas.build(input_shape)
        self.mdn_pi.build(input_shape)
        super().build(input_shape)

    @property
    def trainable_weights(self):
        return (
            self.mdn_mus.trainable_weights
            + self.mdn_sigmas.trainable_weights
            + self.mdn_pi.trainable_weights
        )

    @property
    def non_trainable_weights(self):
        return (
            self.mdn_mus.non_trainable_weights
            + self.mdn_sigmas.non_trainable_weights
            + self.mdn_pi.non_trainable_weights
        )

    def call(self, x, mask=None):
        return layers.concatenate(
            [self.mdn_mus(x), self.mdn_sigmas(x), self.mdn_pi(x)], name="mdn_outputs"
        )



def setup_NN_and_train(x, y):
    N_HIDDEN = 128

    model = keras.Sequential(
        [
            layers.Dense(N_HIDDEN, activation="relu"),
            layers.Dense(N_HIDDEN, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        x,
        y,
        epochs=300,
        batch_size=128,
        validation_split=0.15,
        callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=10)],
    )
    y_pred = model.predict(x)
    plt.scatter(x, y)
    plt.scatter(x, y_pred)
    plt.show()

def split_mixture_params(params, output_dim, num_mixes):
    mus = params[: num_mixes * output_dim]
    sigs = params[num_mixes * output_dim : 2 * num_mixes * output_dim]
    pi_logits = params[-num_mixes:]
    return mus, sigs, pi_logits


def softmax(w, t=1.0):
    e = np.array(w) / t  # adjust temperature
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    return dist


def sample_from_categorical(dist):
    r = np.random.rand(1)  # uniform random number in [0,1]
    accumulate = 0
    for i in range(0, dist.size):
        accumulate += dist[i]
        if accumulate >= r:
            return i
    tf.logging.info("Error sampling categorical model.")
    return -1


def sample_from_output(params, output_dim, num_mixes, temp=1.0, sigma_temp=1.0):
    mus, sigs, pi_logits = split_mixture_params(params, output_dim, num_mixes)
    pis = softmax(pi_logits, t=temp)
    m = sample_from_categorical(pis)
    # Alternative way to sample from categorical:
    # m = np.random.choice(range(len(pis)), p=pis)
    mus_vector = mus[m * output_dim : (m + 1) * output_dim]
    sig_vector = sigs[m * output_dim : (m + 1) * output_dim]
    scale_matrix = np.identity(output_dim) * sig_vector  # scale matrix from diag
    cov_matrix = np.matmul(scale_matrix, scale_matrix.T)  # cov is scale squared.
    cov_matrix = cov_matrix * sigma_temp  # adjust for sigma temperature
    sample = np.random.multivariate_normal(mus_vector, cov_matrix, 1)
    return sample

def get_mixture_loss_func(output_dim, num_mixes):
    def mdn_loss_func(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistributed layer
        y_pred = tf.reshape(
            y_pred,
            [-1, (2 * num_mixes * output_dim) + num_mixes],
            name="reshape_ypreds",
        )
        y_true = tf.reshape(y_true, [-1, output_dim], name="reshape_ytrue")
        # Split the inputs into parameters
        out_mu, out_sigma, out_pi = tf.split(
            y_pred,
            num_or_size_splits=[
                num_mixes * output_dim,
                num_mixes * output_dim,
                num_mixes,
            ],
            axis=-1,
            name="mdn_coef_split",
        )
        # Construct the mixture models
        cat = tfd.Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [
            tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            for loc, scale in zip(mus, sigs)
        ]
        mixture = tfd.Mixture(cat=cat, components=coll)
        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
        return loss

    return mdn_loss_func

def setup_MDN_and_train(X_train, X_test, y_train, y_test, N_HIDDEN = 128):
    OUTPUT_DIMS = 1
    N_MIXES = 20

    mdn_network = keras.Sequential(
        [
            layers.Dense(N_HIDDEN, activation="relu"),
            layers.Dense(N_HIDDEN, activation="relu"),
            MixtureDensityOutput(OUTPUT_DIMS, N_MIXES),
        ]
    )

    mdn_network.compile(loss=get_mixture_loss_func(OUTPUT_DIMS, N_MIXES), optimizer="adam")
    mdn_network.fit(
        X_train,
        y_train,
        epochs=300,
        batch_size=128,
        validation_split=0.15,
        callbacks=[
            callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="loss", patience=5),
        ],
    )

    y_pred_mixture = mdn_network.predict(X_test)
    print(y_pred_mixture.shape)

    # Sample from the predicted distributions
    y_samples = np.apply_along_axis(
        sample_from_output, 1, y_pred_mixture, 1, N_MIXES, temp=1.0
    )
    plt.scatter(X_test, y_test, alpha=0.05, color="blue", label="Ground Truth")
    plt.scatter(
        X_test,
        y_samples[:, :, 0],
        marker='+',
        color="green",
        alpha=0.05,
        label="Mixture Density Network prediction",
    )
    plt.show()

    mus, sigs, pi_logits = split_mixture_params(y_pred_mixture, 1, N_MIXES)
    print(mus.shape)
    # print(pi_logits)
    print(np.min(X_test), np.max(X_test), len(X_test))

    pred_weights, pred_means, pred_std = [pi_logits, mus, sigs]

    obj = [0, 4, 6]
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 6))
    plot_normal_mix(pred_weights[obj][0], pred_means[obj][0], pred_std[obj][0], X_test, axes[0], comp=True)
    axes[0].axvline(x=y_test[obj][0], color='black', alpha=0.5)

    plot_normal_mix(pred_weights[obj][2], pred_means[obj][2], pred_std[obj][2], X_test, axes[1], comp=True)
    axes[1].axvline(x=y_test[obj][2], color='black', alpha=0.5)

    plot_normal_mix(pred_weights[obj][1], pred_means[obj][1], pred_std[obj][1], X_test, axes[2], comp=True)
    axes[2].axvline(x=y_test[obj][1], color='black', alpha=0.5)


def main_NN_test():
    # xy = create_noisy_spiral(10000)
    #
    # x, y = xy[:, 0:1], xy[:, 1:]
    #
    # plt.scatter(x, y)
    # plt.show()

    X_train, X_test, y_train, y_test = build_toy_dataset()
    # X_test, X_train, y_test, y_train = build_toy_dataset()
    print("Size of features in training data: ", X_train.shape)
    print("Size of output in training data: ", y_train.shape)
    print("Size of features in test data: ", X_test.shape)
    print("Size of output in test data: ", y_test.shape)

    plt.scatter(X_train, y_train, marker='+', alpha=0.01)
    plt.show()

    setup_MDN_and_train(X_train, X_test, y_train, y_test, N_HIDDEN=50)

main_NN_test()

