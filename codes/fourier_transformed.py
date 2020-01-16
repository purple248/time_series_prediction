# steps:
# read data
# split train test
# do sliding window - get X, Y
# do fft, get |F_X|, |F_Y|, <F_X, <F_Y
# train 2 predictors (1: magnitude, 2: phase)
# do ifft

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import metrics
import csv


def sine_data_func():
    # generating simple sine data
    t = np.arange(-10,10,0.1)
    f = 1  # frequency
    T = 1/f  # period duration
    w = 2 * math.pi * f  # angular frequency
    signal = np.sin(w*t)
    return signal


def get_sliding_window_data(signal, seq_length_x, seq_length_y):
    # generating windows from data
    X_sliding_window = []
    y_sliding_window = []
    assert seq_length_y % 2 != 0, "seq_length_y is even - insert an odd length"
    half_seq_y = int(round((seq_length_y - 1) / 2, 0))
    for i in range(signal.shape[0] - seq_length_x - half_seq_y - 1):
        window_x = signal[i:(i + seq_length_x)]
        window_y = signal[(i + seq_length_x - half_seq_y):(i + seq_length_x + half_seq_y + 1)]
        X_sliding_window.append(window_x)
        y_sliding_window.append(window_y)

    return np.array(X_sliding_window), np.array(y_sliding_window), np.array(y_sliding_window)[:, (half_seq_y+1)]


def fourier_func(signal):
    # calculate the fourier trasform of the signal, magnitude and phase
    F_signal = np.fft.fft(signal, axis=1)
    magnitude = np.abs(F_signal)
    phase = np.angle(F_signal)

    # applying threshold because of noisy phase
    threshold = np.max(magnitude) / 10000 # maybe replace max with percentile ~95 -- np.percentile(magnitude, 97)
    mask = magnitude < threshold
    phase[mask] = 0

    return magnitude, phase

def inverse_func(magnitude, phase):
    # inverse fourier -> should retrieve the original signal
    real = magnitude * np.cos(phase)
    imaginary = magnitude * np.sin(phase)
    F_signal_composed = real + 1j * imaginary
    inverse = np.fft.ifft(F_signal_composed, axis=1)
    mid = int(round((inverse.shape[1] - 1) / 2)) #the signal point in interest
    return inverse[:,  (mid+1)]


def random_forest(x_train, y_train):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(max_depth=4, random_state=0, n_estimators=100)
    model.fit(x_train, y_train)
    preds_train = model.predict(x_train)
    print(f"random forest - train mse error: {metrics.mean_squared_error(y_train, preds_train)}")
    return model


def linear_regression(x_train, y_train):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(x_train, y_train)
    preds_train = model.predict(x_train)
    print(f"LinearRegression - train mse error: {metrics.mean_squared_error(y_train, preds_train)}")
    return model


def graph(signal, inverse,title):
    plt.plot(signal, color = 'b', label='original signal')
    plt.plot(inverse, color = 'r', label='predicted')
    plt.title(title)
    plt.legend()
    plt.show()


def save_model_res(model_name, seq_len_x, seq_len_y,true_signal,predicted_signal):
    #calculate abs mse because of the imaginary numbers:
    abs_mse = np.sum(np.square(np.abs(true_signal - predicted_signal))) / len(true_signal)
    print(f'model {model_name} - window x: {seq_len_x}, window y:{seq_len_y}, mse:{abs_mse}')

    file_path = "../results/fourier_rez.csv"
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([model_name, seq_len_x, seq_len_y,abs_mse])
    print("test rez saved")
    return

## simple sine data:
# signal = sine_data_func()

#reading other sigal data:
import pickle
with open('../data/triple_sine_signal.pickel', 'rb') as handle:
    generated_data = pickle.load(handle)

signal = generated_data

# uncomment on first run:
# file_path = "../results/fourier_rez.csv"
# with open(file_path, 'a', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["model_name", "seq_len_x", "seq_len_y", "mse"])

#split to train-test
train_test_ratio = 0.7
split_point = int(len(signal)*train_test_ratio)
data_train = signal[:split_point]
data_test = signal[split_point:]


model_name ="random_forest" #for saving results

window_range = [3,10,15]
for seq_length_x in window_range:
    seq_length_y = 3
    print(seq_length_y)
    print(seq_length_x)

    # ftt to all rows in the X , Y after sliding windows:
    X_train, Y_train, y_train_signal = get_sliding_window_data(data_train, seq_length_x, seq_length_y)
    mag_X_train, phase_X_train = fourier_func(X_train)
    mag_Y_train, phase_Y_train = fourier_func(Y_train)

    X_test, Y_test, y_test_signal = get_sliding_window_data(data_test, seq_length_x, seq_length_y)
    mag_X_test, phase_X_test = fourier_func(X_test)
    mag_Y_test, phase_Y_test = fourier_func(Y_test)

    # applying ML model to the magnitude and phase:
    mag_model = random_forest(mag_X_train, mag_Y_train)
    phase_model = random_forest(phase_X_train, phase_Y_train)

    # predict the magnitude and phase with the model:
    mag_Y_pred_train = mag_model.predict(mag_X_train)
    phase_Y_pred_train = phase_model.predict(phase_X_train)

    mag_Y_pred_test = mag_model.predict(mag_X_test)
    phase_Y_pred_test = phase_model.predict(phase_X_test)

    # predict the signal in time with inverse fft:
    pred_signal_train = inverse_func(mag_Y_pred_train, phase_Y_pred_train)
    graph(y_train_signal, pred_signal_train, "model res on train data")

    pred_signal_test = inverse_func(mag_Y_pred_test, phase_Y_pred_test)
    graph(y_test_signal, pred_signal_test, "model res on test data")

    save_model_res(model_name, seq_length_x, seq_length_y,y_test_signal,pred_signal_test)



