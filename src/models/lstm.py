import tensorflow as tf


class LSTMModel(tf.keras.Model):
    def __init__(self, num_units, num_features, dtype=tf.float32):
        """
        Initializes the LSTM model.

        Args:
          num_units: The number of LSTM units.
          num_features: The number of input features.
          dtype: The data type for the model.
        """
        super(LSTMModel, self).__init__(dtype=dtype)
        self.lstm_layer = tf.keras.layers.LSTM(num_units, return_sequences=True)
        self.dense_layer = tf.keras.layers.Dense(num_features)

    def call(self, inputs):
        """
        Defines the forward pass of the model.

        Args:
          inputs: The input tensor.

        Returns:
          The output tensor.
        """
        x = self.lstm_layer(inputs)
        output = self.dense_layer(x)
        return output

    def fit(self, window, epochs=25, batch_size=32, **kwargs):
        """
        Trains the LSTM model.

        Args:
          x_train: The training data.
          y_train: The training labels.
          epochs: The number of training epochs.
          batch_size: The batch size.
          **kwargs: Additional arguments to pass to the `fit` method
                   of `tf.keras.Model`.
        """
        self.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()],
        )  # Compile with Adam optimizer and MSE loss
        history = super(LSTMModel, self).fit(
            window.train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=window.val,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],
            **kwargs,
        )
        return history
