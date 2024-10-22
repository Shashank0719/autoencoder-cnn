from sklearn.svm import OneClassSVM
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.ensemble import IsolationForest

# Define the VAE model
class VAE(tf.keras.Model):
    def __init__(self, input_shape, latent_dim):
        super(VAE, self).__init__()
        self.encoder = models.Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(latent_dim + latent_dim)  # mu and log_var
        ])
        self.decoder = models.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(np.prod(input_shape), activation='sigmoid'),
            layers.Reshape(input_shape)
        ])

    def call(self, inputs):
        mu_log_var = self.encoder(inputs)
        mu, log_var = tf.split(mu_log_var, num_or_size_splits=2, axis=1)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(log_var - tf.square(mu) - tf.exp(log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed

    def reparameterize(self, mu, log_var):
        eps = tf.random.normal(tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps

# Prepare data (e.g., MNIST)
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape((-1, 28 * 28))

# Train VAE
latent_dim = 2
vae = VAE(input_shape=(28*28,), latent_dim=latent_dim)
vae.compile(optimizer='adam')
vae.fit(x_train, x_train, epochs=30, batch_size=128)

# Get reconstruction errors
x_test_reconstructed = vae.predict(x_test)
reconstruction_errors = np.mean(np.square(x_test - x_test_reconstructed), axis=1)

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.1)
iso_forest.fit(reconstruction_errors.reshape(-1, 1))

# Predict anomalies
anomaly_predictions = iso_forest.predict(reconstruction_errors.reshape(-1, 1))
anomalies = np.where(anomaly_predictions == -1)[0]  # Anomaly indices


# Train One-Class SVM
oc_svm = OneClassSVM(gamma='auto', nu=0.1)
oc_svm.fit(reconstruction_errors.reshape(-1, 1))

# Predict anomalies
oc_anomaly_predictions = oc_svm.predict(reconstruction_errors.reshape(-1, 1))
oc_anomalies = np.where(oc_anomaly_predictions == -1)[0]  # Anomaly indices
