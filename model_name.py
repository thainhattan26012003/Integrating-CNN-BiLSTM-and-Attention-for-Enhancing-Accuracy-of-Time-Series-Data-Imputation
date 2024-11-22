import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

# List the available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs available: {gpus}")
    # Optionally, you can set memory growth to avoid allocating all GPU memory
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found, using CPU.")

from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Layer, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


def model_combine(X_train):
    combine = tf.keras.models.Sequential()

    combine.add(Conv1D(filters=8, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    
    combine.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    
    combine.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    
    combine.add(Bidirectional(LSTM(units=8, return_sequences=True)))
    
    combine.add(Bidirectional(LSTM(units=16, return_sequences=True)))
    
    combine.add(Bidirectional(LSTM(units=32, return_sequences=True)))

    class Attention(Layer):
        def __init__(self, **kwargs):
            super(Attention, self).__init__(**kwargs)
    
        def build(self, input_shape):
            self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), 
                                     initializer='random_normal', trainable=True)
            self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), 
                                     initializer='zeros', trainable=True)        
            super(Attention, self).build(input_shape)
     
        def call(self, x):
            e = K.tanh(K.dot(x, self.W) + self.b)
            e = K.squeeze(e, axis=-1)
            alpha = K.softmax(e)
            alpha = K.expand_dims(alpha, axis=-1)
            context = x * alpha
            context = K.sum(context, axis=1)
            return context

    combine.add(Attention())
    
    combine.add(Dense(units=32, activation='relu'))
    
    combine.add(Dense(units=1))
    
    combine.compile(optimizer=Adam(), loss="mean_squared_error", metrics=['mae'])
    
    return combine
    

def model_CNN(X_train):
    CNN = tf.keras.models.Sequential()

    CNN.add(Conv1D(filters=8, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    
    CNN.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    
    CNN.add(Conv1D(filters=32, kernel_size=3, activation='relu'))

    CNN.add(Flatten())
    
    CNN.add(Dense(units=32, activation='relu'))
    
    CNN.add(Dense(units=1))
    
    CNN.compile(optimizer=Adam(), loss="mean_squared_error", metrics=['mae'])
    
    return CNN


def model_BiLSTM(X_train):
    BiLSTM = tf.keras.models.Sequential()
    
    BiLSTM.add(Bidirectional(LSTM(units=8, return_sequences=True), input_shape=(X_train.shape[1], 1)))
    
    BiLSTM.add(Bidirectional(LSTM(units=16, return_sequences=True)))
    
    BiLSTM.add(Bidirectional(LSTM(units=32)))
    
    BiLSTM.add(Dense(units=32, activation='relu'))
    
    BiLSTM.add(Dense(units=1))
    
    BiLSTM.compile(optimizer=Adam(), loss="mean_squared_error", metrics=['mae'])
    
    return BiLSTM


def model_CNN_Attention(X_train):
    CNN_Attention = tf.keras.models.Sequential()

    CNN_Attention.add(Conv1D(filters=8, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    
    CNN_Attention.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    
    CNN_Attention.add(Conv1D(filters=32, kernel_size=3, activation='relu'))

    class Attention(Layer):
        def __init__(self, **kwargs):
            super(Attention, self).__init__(**kwargs)
    
        def build(self, input_shape):
            self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), 
                                     initializer='random_normal', trainable=True)
            self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), 
                                     initializer='zeros', trainable=True)        
            super(Attention, self).build(input_shape)
     
        def call(self, x):
            e = K.tanh(K.dot(x, self.W) + self.b)
            e = K.squeeze(e, axis=-1)
            alpha = K.softmax(e)
            alpha = K.expand_dims(alpha, axis=-1)
            context = x * alpha
            context = K.sum(context, axis=1)
            return context

    CNN_Attention.add(Attention())
    
    CNN_Attention.add(Dense(units=32, activation='relu'))
    
    CNN_Attention.add(Dense(units=1))
    
    CNN_Attention.compile(optimizer=Adam(), loss="mean_squared_error", metrics=['mae'])
    
    return CNN_Attention


def model_Attention(X_train):
    AttentionModel  = tf.keras.models.Sequential()
    
    class Attention(Layer):
        def __init__(self, **kwargs):
            super(Attention, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(name='attention_weight', 
                                    shape=(input_shape[-1], 1), 
                                    initializer='random_normal', 
                                    trainable=True)
            self.b = self.add_weight(name='attention_bias', 
                                    shape=(input_shape[1], 1), 
                                    initializer='zeros', 
                                    trainable=True)
            super(Attention, self).build(input_shape)

        def call(self, x):
            e = K.tanh(K.dot(x, self.W) + self.b)
            e = K.squeeze(e, axis=-1)
            alpha = K.softmax(e)
            alpha = K.expand_dims(alpha, axis=-1)
            context = x * alpha
            context = K.sum(context, axis=1)
            return context

    AttentionModel.add(Attention(input_shape=(X_train.shape[1], 1)))
    
    AttentionModel.add(Dense(units=32, activation='relu'))
    
    AttentionModel.add(Dense(units=1))
    
    AttentionModel.compile(optimizer=Adam(), loss="mean_squared_error", metrics=['mae'])
    
    return AttentionModel


def model_CNN_BiLSTM(X_train):
    CNN_BiLSTM = tf.keras.models.Sequential()

    CNN_BiLSTM.add(Conv1D(filters=8, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    
    CNN_BiLSTM.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    
    CNN_BiLSTM.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    
    CNN_BiLSTM.add(Bidirectional(LSTM(units=8, return_sequences=True)))
    
    CNN_BiLSTM.add(Bidirectional(LSTM(units=16, return_sequences=True)))
    
    CNN_BiLSTM.add(Bidirectional(LSTM(units=32)))
    
    CNN_BiLSTM.add(Dense(units=32, activation='relu'))
    
    CNN_BiLSTM.add(Dense(units=1))
    
    CNN_BiLSTM.compile(optimizer=Adam(), loss="mean_squared_error", metrics=['mae'])
    
    return CNN_BiLSTM


def model_BiLSTM_Attention(X_train):
    BiLSTM_Attention = tf.keras.models.Sequential()
    
    BiLSTM_Attention.add(Bidirectional(LSTM(units=8, return_sequences=True), input_shape=(X_train.shape[1], 1)))
    BiLSTM_Attention.add(Bidirectional(LSTM(units=16, return_sequences=True)))
    BiLSTM_Attention.add(Bidirectional(LSTM(units=32, return_sequences=True)))

    class Attention(Layer):
        def __init__(self, **kwargs):
            super(Attention, self).__init__(**kwargs)
    
        def build(self, input_shape):
            self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), 
                                     initializer='random_normal', trainable=True)
            self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), 
                                     initializer='zeros', trainable=True)        
            super(Attention, self).build(input_shape)
     
        def call(self, x):
            # Apply weights and biases
            e = tf.keras.activations.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
            e = tf.squeeze(e, axis=-1)
            alpha = tf.nn.softmax(e)
            alpha = tf.expand_dims(alpha, axis=-1)
            context = x * alpha
            context = tf.reduce_sum(context, axis=1)
            return context
        
        def compute_output_shape(self, input_shape):
            # Output shape is (batch_size, input_dim)
            return (input_shape[0], input_shape[-1])

    BiLSTM_Attention.add(Attention())
    BiLSTM_Attention.add(Dense(units=32, activation='relu'))
    BiLSTM_Attention.add(Dense(units=1))
    
    BiLSTM_Attention.compile(optimizer=Adam(), loss="mean_squared_error", metrics=['mae'])
    
    return BiLSTM_Attention

def select_model(model_name, X_train):
    if model_name == "combine":
        return model_combine(X_train)
    elif model_name == "CNN":
        return model_CNN(X_train)
    elif model_name == "BiLSTM":
        return model_BiLSTM(X_train)
    elif model_name == "Attention":
        return model_Attention(X_train)
    elif model_name == "CNN_BiLSTM":
        return model_CNN_BiLSTM(X_train)
    elif model_name == "CNN_Attention":
        return model_CNN_Attention(X_train)
    elif model_name == "BiLSTM_Attention":
        return model_BiLSTM_Attention(X_train)
    else:
        raise("There are not model model name accepted!")    