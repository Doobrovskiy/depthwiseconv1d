# Keras 1D Depthwise Convolutional layer

As of Tensorflow 2.2 still does not implement DepthwiseConv1D layer I have implemented it myself, as I find it useful in many applications, such as those collecting timeseries data from multiple channels having same or similar nature.

There are only 2 files in this project
- dwconv1d/depthwiseconv1d.py contains the layer code
- example.py contains example code

A flag common_kernel is also added to a standard Keras parameter set. This is useful if you need to pre-process multiple 1D channels with the same nature such as a sensor array, stock market data on multiple instruments, weather data from multiple stations etc.

    common_kernel: if set to True, same kernel is applied to each channel,
      if False, separate kernel is applied to each channel (default case)

Please notify me of any problems.
Nikolai Kovshov @Doobrovskiy
