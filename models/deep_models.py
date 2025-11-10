import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten,
    BatchNormalization, GlobalMaxPooling1D, Bidirectional,
    LSTM, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class BaseDeepModel(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=50, batch_size=256, verbose=0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y):
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, X):
        raise NotImplementedError("Subclasses must implement predict method")

    def predict_proba(self, X):
        raise NotImplementedError("Subclasses must implement predict_proba method")

class CNNModel(BaseDeepModel):
    def __init__(self, input_shape, epochs=50, batch_size=256, verbose=0, 
                 dropout_rate=0.3, l2_reg=0.001):
        super().__init__(epochs, batch_size, verbose)
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Conv1D(256, 1, activation='relu', padding='same',
                   kernel_regularizer=l2(self.l2_reg),
                   input_shape=self.input_shape),
            BatchNormalization(),

            tf.keras.layers.SeparableConv1D(128, 3, activation='relu', padding='same'),
            MaxPooling1D(2, padding='same'),
            
            Conv1D(64, 1, activation='relu', padding='same'),
            GlobalMaxPooling1D(),
            
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            
            Dense(1, activation='sigmoid')
        ])

        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
        )
        return model

    def fit(self, X, y, callbacks=None):
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=callbacks if callbacks else []
        )
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        return self.model.predict(X)

class BLSTMModel:
    def __init__(self, input_shape, epochs=100, batch_size=256,  verbose=0, 
                 dropout_rate=0.2, recurrent_dropout=0.1, 
                 learning_rate=0.001):
        self.model = self._build_model(input_shape, dropout_rate, recurrent_dropout, learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def _build_model(self, input_shape, dropout_rate, recurrent_dropout, lr):
        model = tf.keras.Sequential([
            # The input shape must be consistent with the data preprocessing
            tf.keras.layers.InputLayer(input_shape=input_shape),
            
            # Use more efficient CuDNNLSTM
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(128,
                    return_sequences=True,
                    recurrent_dropout=recurrent_dropout,
                    kernel_initializer='he_normal'
                )
            ),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64,
                    recurrent_dropout=recurrent_dropout)
            ),
            
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        optimizer = Adam(learning_rate=lr, clipnorm=1.0)  # Add gradient clipping
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, X, y):
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose
        )
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int).reshape(-1,)

    def predict_proba(self, X):
        return self.model.predict(X)

class TransformerModel(BaseDeepModel):
    def __init__(self, input_dim, num_heads=2, ff_dim=64, num_layers=2, 
                 dropout_rate=0.1, epochs=50, batch_size=256, verbose=0, 
                 l2_reg=0.001, learning_rate=1e-3):
        super().__init__(epochs, batch_size, verbose)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=(None, self.input_dim))
        x = inputs
        for _ in range(self.num_layers):
            # Multi-head attention
            attn_output = MultiHeadAttention(
                num_heads=self.num_heads, 
                key_dim=self.input_dim // self.num_heads,
                kernel_regularizer=l2(self.l2_reg)
            )(x, x)
            attn_output = Dropout(self.dropout_rate)(attn_output)
            x = LayerNormalization(epsilon=1e-6)(x + attn_output)

            # Feed-forward network
            ff_output = Dense(self.ff_dim, activation='relu', 
                             kernel_regularizer=l2(self.l2_reg))(x)
            ff_output = Dense(self.input_dim, 
                             kernel_regularizer=l2(self.l2_reg))(ff_output)
            ff_output = Dropout(self.dropout_rate)(ff_output)
            x = LayerNormalization(epsilon=1e-6)(x + ff_output)

        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
        )
        return model

    def fit(self, X, y):
        def lr_scheduler(epoch, lr):
            return lr * tf.math.exp(-0.1) if epoch >= 10 else lr
            
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=[LearningRateScheduler(lr_scheduler)]
        )
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int).reshape(-1,)

    def predict_proba(self, X):
        return self.model.predict(X)