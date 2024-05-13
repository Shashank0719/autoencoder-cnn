from autoencoder import Autoencoder
from tensorflow.keras.datasets import mnist

BATCH_SIZE = 38
EPOCHS = 22
LEARNING_RATE = 0.002

def train(x_train, batch_size, epochs, learning_rate):
    autoencoder = Autoencoder(
        input_shape=(28,28,1),
        filters=(32,64,64,64),
        kernel_size=(3,3,3,3),
        kernel_strides=(1,2,2,1),
        latent_dim=2
        )
    autoencoder.summary()
    autoencoder.compile(learning_rate=learning_rate)
    autoencoder.train(x_train=x_train,batch_size=batch_size,epochs=epochs,x_test=x_test)
    return autoencoder

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32")/255
    x_train = x_train.reshape(x_train.shape + (1,))

    x_test = x_test.astype("float32")/255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test
    
if __name__=="__main__":
    x_train, _ , x_test, _ = load_mnist()
    autoencoder = train(x_train=x_train[:10000], x_test=x_test[:5000], batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate= LEARNING_RATE)
