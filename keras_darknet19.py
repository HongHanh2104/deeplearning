"""Darknet19 Model Defined in Keras."""
import functools
from functools import partial

from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from utils import compose

# Partial wrapper for Convolution2D with static default argument.
_DarknetConv2D = partial(Conv2D, padding = 'same')

@functools.wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
	"""Wrapper to set Darknet weight regularizer for Convolution2D."""
	darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
	darknet_conv_kwargs.update(kwargs)
	return _DarknetConv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
	"""Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
	"""Normalize the activations of the previous layer at each batch, i.e. 
	applies a transformation that maintains the mean activation close to 0 
	and the activation standard deviation close to 1."""

	no_bias_kwargs = {'use_bias': False}
	no_bias_kwargs.update(kwargs)
	return compose(
		DarknetConv2D(*args, **no_bias_kwargs),
		BatchNormalization(),
		LeakyReLU(alpha = 0.1))


def bottleneck_block(outer_filters, bottleneck_filters):
	"""Bottleneck block of 3x3, 1x1, 3x3 convolutions."""
	return compose(
		DarknetConv2D_BN_Leaky(outer_filters, (3, 3)),
		DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
		DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))


def bottleneck_x2_block(outer_filters, bottleneck_filters):
	"""Bottleneck block of 3x3, 1x1, 3x3, 1x1, 3x3 convolutions."""
	return compose(
		bottleneck_block(outer_filters, bottleneck_filters),
		DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
		DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))


def darknet_body():
	"""Generate first 18 conv layers of Darknet-19."""
	"""
		Parsing section convolutional_0
		conv2d bn leaky (3, 3, 3, 32)
		Parsing section maxpool_0
		Parsing section convolutional_1
		conv2d bn leaky (3, 3, 32, 64)
		Parsing section maxpool_1
		Parsing section convolutional_2
		conv2d bn leaky (3, 3, 64, 128)
		Parsing section convolutional_3
		conv2d bn leaky (1, 1, 128, 64)
		Parsing section convolutional_4
		conv2d bn leaky (3, 3, 64, 128)
		Parsing section maxpool_2
		Parsing section convolutional_5
		conv2d bn leaky (3, 3, 128, 256)
		Parsing section convolutional_6
		conv2d bn leaky (1, 1, 256, 128)
		Parsing section convolutional_7
		conv2d bn leaky (3, 3, 128, 256)
		Parsing section maxpool_3
		Parsing section convolutional_8
		conv2d bn leaky (3, 3, 256, 512)
		Parsing section convolutional_9
		conv2d bn leaky (1, 1, 512, 256)
		Parsing section convolutional_10
		conv2d bn leaky (3, 3, 256, 512)
		Parsing section convolutional_11
		conv2d bn leaky (1, 1, 512, 256)
		Parsing section convolutional_12
		conv2d bn leaky (3, 3, 256, 512)
		Parsing section maxpool_4
		Parsing section convolutional_13
		conv2d bn leaky (3, 3, 512, 1024)
		Parsing section convolutional_14
		conv2d bn leaky (1, 1, 1024, 512)
		Parsing section convolutional_15
		conv2d bn leaky (3, 3, 512, 1024)
		Parsing section convolutional_16
		conv2d bn leaky (1, 1, 1024, 512)
		Parsing section convolutional_17
		conv2d bn leaky (3, 3, 512, 1024)
		Parsing section convolutional_18
		conv2d bn leaky (3, 3, 1024, 1024)
		Parsing section convolutional_19
		conv2d bn leaky (3, 3, 1024, 1024)
	"""
	return compose(
		DarknetConv2D_BN_Leaky(32, (3, 3)),
		MaxPooling2D(),
		DarknetConv2D_BN_Leaky(64, (3, 3)),
		MaxPooling2D(),
		bottleneck_block(128, 64),
		MaxPooling2D(),
		bottleneck_block(256, 128),
		MaxPooling2D(),
		bottleneck_x2_block(512, 256),
		MaxPooling2D(),
		bottleneck_x2_block(1024, 512))


def darknet19(inputs):
	"""Generate Darknet-19 model for Imagenet classification."""
	body = darknet_body()(inputs)
	logits = DarknetConv2D(1000, (1, 1), activation='softmax')(body)
	return Model(inputs, logits)