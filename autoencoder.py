import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


class Autoencoder():
    
    def __init__(self,
                 input_shape,
                 filters,
                 kernel_size,
                 kernel_strides,
                 latent_dim):
        
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_strides = kernel_strides
        self.latent_dim = latent_dim
        
        self.encoder = None
        self.decoder = None
        self.model = None

        self._before_dense_layer = None
        self._num_conv_layers = len(filters)
        self._model_input = None

        self._build()

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name=f"autoencoder")
    
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_decoder_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layer = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layer)
        self.decoder = Model(decoder_input, decoder_output, name=f"Decoder")
        
    def _add_decoder_input(self):
        return Input(shape=(self.latent_dim,), name="decoder_input")
    
    def _add_decoder_dense_layer(self,decoder_input):
        return Dense(np.prod(self._before_dense_layer), name="decoder_dense_layer")(decoder_input)
    
    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._before_dense_layer)(dense_layer)
    
    def _add_conv_transpose_layers(self, x): # x -> reshape_layer output
        for input_range in range(self._num_conv_layers,1,-1):
            print(input_range)
            x = self._add_conv_transpose_layer(input_range-1,x)
        return x
    
    def _add_conv_transpose_layer(self, input_range, input):
        conv_num = self._num_conv_layers - input_range
        conv_transpose_layer = Conv2DTranspose(
            filters=self.filters[input_range],
            kernel_size=self.kernel_size[input_range],
            strides=self.kernel_strides[input_range],
            padding="same",
            name=f"decoder_conv_transpose_layer{conv_num}"
        )
        x = conv_transpose_layer(input)
        x = ReLU(name=f"decoder_relu_layer{conv_num}")(x)
        return BatchNormalization(name=f"decoder_batch_normalized_layer{conv_num}")(x)
    
    def _add_decoder_output(self, input):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.kernel_size[0],
            strides=self.kernel_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer{self._num_conv_layers}")
        x = conv_transpose_layer(input)
        return Activation("sigmoid", name="sigmoid_layer")(x)
    
    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        dense_layer = self._add_dense_layer(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, dense_layer, name="encoder")
        
    
    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_inputs")
    
    def _add_conv_layers(self, encoder_input):
        x = encoder_input
        for input_range in range(self._num_conv_layers):
            x = self._add_conv_layer(input_range, x)
        return x
    
    def _add_conv_layer(self, input_range, input):
        conv_num = input_range+1
        conv_layer =  Conv2D(
            filters=self.filters[input_range],
            kernel_size=self.kernel_size[input_range],
            strides=self.kernel_strides[input_range],
            padding="same",
            name=f"encoder_conv_layer{conv_num}"
        )
        x = conv_layer(input)
        x = ReLU(name=f"encoder_relu_layer{conv_num}")(x)
        return BatchNormalization(name=f"encoder_batch_normalized_layer{conv_num}")(x)
    
    def _add_dense_layer(self,conv_layers):
        self._before_dense_layer = K.int_shape(conv_layers)[1:] # (2,6,6,32) -> first 2 represent batches in which we are not interested thats why x[1:]
        x = Flatten()(conv_layers)
        return Dense(self.latent_dim, name="encoder_dense_layer")(x)
    
    def predict(self,input):
        encoded_output = self.encoder.predict(input)
        return self.decoder.predict(encoded_output)

    
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse)

    def train(self, x_train, batch_size, epochs,):
        self.model.fit(x_train, x_train,
                       batch_size=batch_size, epochs=epochs, shuffle=True)

if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape=(28,28,1),
        filters=(32,64,64,64),
        kernel_size=(3,3,3,3),
        kernel_strides=(1,2,2,1),
        latent_dim=2
        )
    
    autoencoder.summary()
