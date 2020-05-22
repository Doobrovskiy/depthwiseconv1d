import numpy as np
import tensorflow.keras as keras
from dwconv1d import *

# Generating data
# There are 5 independent channels
# Each channel represents 10 points randomly scattered along the line with a constant slope
#  where slope value is {channel_number}
# y is highest standard deviation from channel baseline among the channels

samples = 10000
length = 10
channels = 5
scatter = 1

shape = (samples, length, channels)

x = np.zeros(shape)
y = np.zeros((samples))

for idx, n in np.ndenumerate(x):
    x[idx] = idx[1] * idx[2] + np.random.normal(0, scatter)

for i in range(samples):
    y[i] = np.max([np.std(x[i,:,j] - [j * k for k in range(length)]) for j in range(channels)])

print(x.shape, y.shape)

# This data is processed separately on each channel
# Only the last MaxPooling layer merges channel-wise data and makes final decision
# Training this model quickly converges

model = keras.models.Sequential()
model.add(DepthwiseConv1D(kernel_size = 10, padding = 'same', input_shape=shape[1:]))
model.add(keras.layers.ReLU())
model.add(DepthwiseConv1D(kernel_size = 10, padding = 'same'))
model.add(keras.layers.ReLU())
model.add(DepthwiseConv1D(kernel_size = 3, strides = 2))
model.add(keras.layers.ReLU())
model.add(DepthwiseConv1D(kernel_size = 3, strides = 2))
model.add(keras.layers.ReLU())
model.add(keras.layers.Reshape((channels,1)))
model.add(keras.layers.GlobalMaxPooling1D())
model.compile(optimizer='adam', loss='mse')

print(model.summary())

model.fit(x, y, epochs=15, validation_split=0.3, verbose=1, batch_size = 64)
