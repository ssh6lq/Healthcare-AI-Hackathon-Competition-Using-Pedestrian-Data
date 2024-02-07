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
  # Restrict TensorFlow to only allocate 2 GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)]) # limit in megabytes
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# 정상 데이터(01)
tr_data = os.listdir("10S1_FL/train/01") # 정상인 데이터


# scaler = MinMaxScaler()
scaler = StandardScaler()

#파라미터
epochs = 150
batch = 64
lr = 0.0001

# LSTM-Autoencoder
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

lstm_ae.summary()

# 모델 학습
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

# threshold 지정
threshold_ave = np.mean(threshold)

# 비정상 데이터
sar_data = os.listdir("10S1_FL/train/02") # 근감소증 환자 데이터


# 이상치 탐지 
for i in sar_data:
    path = "10S1_FL/train/02/"+ i
    print(path)
#   print("patient_number : ",i)
    test_sheet = pd.read_csv(path, index_col = 0).dropna()
    test_num = test_sheet[["x_angle","y_angle","z_angle"]]
    test_num = scaler.fit_transform(test_num)
    test_num = np.reshape(test_num,(test_num.shape[0],3,1))
    test_x_predictions = lstm_ae.predict(test_num)
    test_mae_loss = np.mean(np.power(test_x_predictions - test_num,2), axis=1).flatten()
    
    print(len(test_sheet))
    print(len(test_mae_loss))

    plt.plot(test_mae_loss)
    plt.axhline(y=threshold_ave, color='red', linewidth=2)

    for i in range(len(test_sheet)):
        if test_mae_loss[i] >= threshold_ave:
            plt.scatter(i, test_mae_loss[i],c='r')

    plt.grid()
    plt.show()
    
    loss_max = np.max(test_mae_loss)
    print(f'Reconstruction error threshold: {loss_max}')


# 정상 데이터 테스트
test_num = os.listdir("10S1_FL/test/01") # 정상인 데이터

for i in test_num:
    path = "10S1_FL/test/01/"+ i
    print(path)
#   print("patient_number : ",i)
    test_sheet = pd.read_csv(path, index_col = 0).dropna()
    test_num = train_sheet[["x_angle","y_angle","z_angle"]]
    test_num = scaler.fit_transform(test_num)
    test_num = np.reshape(test_num,(test_num.shape[0],3,1))
    test_no_x_predictions= lstm_ae.predict(test_num)
    test_no_mae_loss = np.mean(np.power(test_no_x_predictions - test_num,2), axis=1).flatten()
    
    plt.plot(test_no_mae_loss)
    plt.axhline(y=threshold_ave, color='red', linewidth=1)
    plt.grid()
    plt.show()
    
    test_no_mae_loss_max = np.max(test_no_mae_loss)
    print(f'Reconstruction error threshold: {test_no_mae_loss_max}')