import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Model ,models, layers, optimizers, utils
from keras.layers import Dense, Activation, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf           
import tensorflow.keras.layers as L
from tensorflow.keras import optimizers, Sequential, Model
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# tf.__version__

#GPU 용량 할당
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

#정상 데이터
tr_data = os.listdir("10S1_FL/train/01") # 정상인 데이터


# scaler = MinMaxScaler()
scaler = StandardScaler()

#parameter
epochs = 150
batch = 64
lr = 0.0001

# lstm_autoencoder model
lstm_ae = models.Sequential()
# Encoder
lstm_ae.add(layers.LSTM(32, activation='relu', input_shape=(1000, 1), return_sequences=True))
lstm_ae.add(tf.keras.layers.Dropout(rate=0.2))
lstm_ae.add(layers.LSTM(16, activation='relu', return_sequences=False))
lstm_ae.add(layers.RepeatVector(1))

# Decoder
lstm_ae.add(layers.LSTM(16, activation='relu', return_sequences=True))
lstm_ae.add(tf.keras.layers.Dropout(rate=0.2))
lstm_ae.add(layers.LSTM(32, activation='relu', return_sequences=True))
lstm_ae.add(layers.TimeDistributed(layers.Dense(1)))

# lstm_ae.summary()

#model 학습
k=0
for i in tr_data:
    if k < 41:
        path = "10S1_FL/train/01/"+ i
        print(path)
#     print("patient_number : ",i)
        test_sheet = pd.read_csv(path, index_col = 0).dropna()
        test_num = test_sheet[["x_angle","y_angle","z_angle"]]
        test_num = scaler.fit_transform(test_num)
        test_num = np.reshape(test_num,(test_num.shape[0],3,1))
        lstm_ae.compile(loss='mse', optimizer=optimizers.Adam(lr))
        history = lstm_ae.fit(test_num, test_num, epochs=epochs, batch_size=batch)
    
        plt.plot(history.history['loss'], label='train loss')
        plt.legend()
        plt.xlabel('Epoch'); plt.ylabel('loss')
        plt.show()

        k += 1