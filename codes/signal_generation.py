import math
import numpy as np
import matplotlib.pyplot as plt
import pickle




def sine_data_func(f):
    #input f = frequency
    t = np.arange(-10,10,0.1)
    T = 1/f  # period duration
    w = 2 * math.pi * f  # angular frequency
    signal = np.sin(w*t)
    return signal

def white_noise(len, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len) * noise_level

### generating 2 sine signal ###
signal_1 = sine_data_func(0.1)
signal_2 = sine_data_func(7)
signal  = signal_1+signal_2
noise = white_noise(len(signal), noise_level=0.5, seed=42)
signal += noise

plt.plot(signal)
plt.show()

with open('../data/2_sine_signal.pickel', 'wb') as handle:
    pickle.dump(signal, handle, protocol=pickle.HIGHEST_PROTOCOL)