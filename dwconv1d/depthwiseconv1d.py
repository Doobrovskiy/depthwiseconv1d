import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import models, layers, initializers, regularizers, constraints
from tensorflow.python.keras.layers import Conv1D
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.tf_export import keras_export
    
"""Depthwise separable 1D convolution.
  Depthwise Separable convolutions consists in performing
  just the first step in a depthwise spatial convolution
  (which acts on each input channel separately).
  The `depth_multiplier` argument controls how many
  output channels are generated per input channel in the depthwise step.
  Arguments:
    kernel_size: An integer, specifying the
      length of the 1D convolution window.
    strides: An integer specifying the strides of the convolution.
    padding: one of `'valid'` or `'same'` (case-insensitive).
    common_kernel: if set to True, same kernel is applied to each channel,
      if False, separate kernel is applied to each channel (default case)
    depth_multiplier: The number of depthwise convolution output channels
      for each input channel.
      The total number of depthwise convolution output
      channels will be equal to `filters_in * depth_multiplier`.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, length, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, length)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be 'channels_last'.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    depthwise_initializer: Initializer for the depthwise kernel matrix (
      see `keras.initializers`).
    bias_initializer: Initializer for the bias vector (
      see `keras.initializers`).
    depthwise_regularizer: Regularizer function applied to
      the depthwise kernel matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its 'activation') (
      see `keras.regularizers`).
    depthwise_constraint: Constraint function applied to
      the depthwise kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).
  Input shape:
    3D tensor with shape:
    `[batch_size, channels, length]` if data_format='channels_first'
    or 3D tensor with shape:
    `[batch_size, length, channels]` if data_format='channels_last'.
  Output shape:
    3D tensor with shape:
    `[batch_size, filters, new_length]` if data_format='channels_first'
    or 3D tensor with shape:
    `[batch_size, new_length, filters]` if data_format='channels_last'.
    `length` value might have changed due to padding.
  Returns:
    A tensor of rank 3 representing
    `activation(depthwiseconv1d(inputs, kernel) + bias)`.
  """
    
class DepthwiseConv1D(Conv1D):
    def __init__(self,
                kernel_size,
                strides=1,
                padding='valid',
                common_kernel = False,
                depth_multiplier=1,
                data_format=None,
                activation=None,
                use_bias=True,
                depthwise_initializer='glorot_uniform',
                bias_initializer='zeros',
                depthwise_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                depthwise_constraint=None,
                bias_constraint=None,
                **kwargs):
        super(DepthwiseConv1D, self).__init__(
                filters=None,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                activation=activation,
                use_bias=use_bias,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                bias_constraint=bias_constraint,
                **kwargs)
        
        self.common_kernel = common_kernel
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        
    # For compatibility with some older versions of Keras
    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return 1
        else:
            return -1

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Inputs to `DepthwiseConv1D` should have rank 3. '
                       'Received input shape:', str(input_shape))
        input_shape = tensor_shape.TensorShape(input_shape)
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs to '
                       '`DepthwiseConv1D` should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis]) 
        kernel_dim = 1 if self.common_kernel==True else input_dim
        depthwise_kernel_shape = (self.kernel_size[0], kernel_dim , self.depth_multiplier)

        self.channels = input_dim
        
        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(kernel_dim * self.depth_multiplier,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        if self.padding == 'causal':
            inputs = array_ops.pad(inputs, self._compute_causal_padding())
        if self.data_format == 'channels_last':
            strides = (1,) + self.strides * 2 + (1,)
            spatial_start_dim = 1
        else:
            strides = (1, 1) + self.strides * 2
            spatial_start_dim = 2

        # Explicitly broadcast inputs and kernels to 4D.
        inputs = array_ops.expand_dims(inputs, spatial_start_dim)
        
        if self.common_kernel == True:
            #Need to replicate kernel {channels} times over axis 1
            dw_kernel = tf.tile(self.depthwise_kernel, (1, self.channels, 1))
            bias_kernel = tf.tile(self.bias, (self.channels, ))
        else:
            dw_kernel = self.depthwise_kernel
            bias_kernel = self.bias
            
        dw_kernel = array_ops.expand_dims(dw_kernel, 0)
        
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        outputs = nn.depthwise_conv2d(
            inputs,
            dw_kernel,
            strides=strides,
            padding=op_padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        outputs = array_ops.squeeze(outputs, [spatial_start_dim])

        if self.use_bias:
            outputs = backend.bias_add(outputs, bias_kernel, data_format=self.data_format)
            
        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            length = input_shape[2]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            length = input_shape[1]
            out_filters = input_shape[2] * self.depth_multiplier

        length_new = conv_utils.conv_output_length(length, self.kernel_size,
                                         self.padding,
                                         self.strides)
        
        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, length_new)
        elif self.data_format == 'channels_last':
            return (input_shape[0], length_new, out_filters)

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
        return config
