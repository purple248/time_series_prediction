# Time series analysis with Fourier Transform


This project is trying to predict the next signal in time  using Fourier Transform. 
The motivation of taking this approach is the assumption that signals which are highly periodic can be represented very clearly in frequency domain. Moreover, the time-shifted property of the transform should allow very straight forward predictions for these signals.



# Data
The data for this project is generated data with a clear periodicity to test the benefits of this approach.

# Method
To predict a specific data point in time, "x" will be a window before this data point, and "y" will be a window around this data point, symmetrically.
(The length of the windows is a parameter)

The next step is doing fft to X and Y, and get the magnitude and phase for each window.
Train two models (using random forest)
- model 1: receive the magnitude of x window and predict the magnitude of y window
- model 2: receive the phase of x window and predict the phase of y window
    
After having the predicted magnitude and phase of y - do ifft (inverse fft) to receive the signal on the time domain, and extract the middle point.


# Benchmark models
Using some familiar machine learning regression models to see how the Fourier Transform approach performs compared to them.

The data_handler file is preparing the data for benchmarks with adding 3 historical points to predict the future next point. And normalize the data.


