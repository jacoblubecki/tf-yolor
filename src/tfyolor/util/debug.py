from datetime import datetime

import tensorflow as tf
import tensorflow.keras.utils
from tensorflow import keras


def trace_for_tensorboard(tag: str, model, shape):
    """ Traces a graph of a model so that it can be visualized in tensorboard."""
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    logdir = f'logs/func/{stamp}'
    writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True, profiler=False)

    fn = tf.function(model)

    inputs = tf.ones(shape)
    fn(inputs)

    with writer.as_default():
        tf.summary.trace_export(
            name=tag,
            step=0,
            profiler_outdir=logdir)


def plot_model(model, shape):
    """ Creates a `model.png` showing the basic structure of a Keras model."""
    inputs = keras.Input(shape[1:])
    ones = tf.ones(shape)
    model(ones)  # init graph
    outputs = model.call(inputs)
    wrapped_model = keras.Model(inputs, outputs)
    return tensorflow.keras.utils.plot_model(
        wrapped_model, expand_nested=True, show_shapes=True)
