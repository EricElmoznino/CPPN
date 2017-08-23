import tensorflow as tf
from PIL import Image
import numpy as np
import os


def variables(shape, stddev=1.0, name='weights'):
    initial = tf.random_normal_initializer(stddev=stddev)
    return tf.get_variable(name, shape=shape,
                           initializer=initial)


def fully_connected(prev_layer, layer_size, stddev=1.0, bias=True,
                    name='fully_connected', activation=None):
    with tf.variable_scope(name):
        assert len(prev_layer.shape) == 2
        prev_layer_size = int(prev_layer.shape[1])
        weights = variables([prev_layer_size, layer_size], stddev=stddev)
        layer = tf.matmul(prev_layer, weights)
        if bias:
            biases = variables([layer_size], stddev=stddev, name='biases')
            layer += biases
        if activation is not None:
            layer = activation(layer)
    return layer


def save_image(image, name):
    im = np.uint8(image*255.0)
    if len(im.shape) == 3:
        im = np.transpose(im, [1, 0, 2])
    else:
        im = np.transpose(im, [1, 0])
    im = Image.fromarray(im)
    im.save(os.path.join('generated_images', name+'.png'))
