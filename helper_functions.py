import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
from matplotlib import pyplot as plt


def plot_loss_curves(history, fig_size=(16, 16)):
    """
    Returns separate loss curves for training and validation metrics.

    Args:
      history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
    """
    metrics = history.history
    plt.figure(figsize=fig_size)
    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, metrics["loss"], metrics["val_loss"])
    plt.legend(["loss", "val_loss"])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel("Epoch")
    plt.ylabel("Loss [CrossEntropy]")

    plt.subplot(1, 2, 2)
    plt.plot(
        history.epoch,
        100 * np.array(metrics["accuracy"]),
        100 * np.array(metrics["val_accuracy"]),
    )
    plt.legend(["accuracy", "val_accuracy"])
    plt.ylim([0, 100])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [%]")


def add_white_noise(audio, label):
    """add white noise to an audio clip and return the new clip"""
    noise_factor = 0.007
    audio = tf.cast(audio, tf.float32)
    noise = tf.random.normal(
        shape=tf.shape(audio), mean=0.0, stddev=1.0, dtype=tf.float32
    )
    augmented_audio = audio + noise_factor * noise
    audio = tf.clip_by_value(augmented_audio, -1.0, 1.0)
    return audio, label


def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


def get_spectrogram(waveform):
    # input_len = 16000
    # waveform = waveform[:input_len]
    # zero_padding = tf.zeros(input_len - tf.shape(waveform), dtype=tf.float32)
    # waveform = tf.cast(waveform, dtype=tf.float32)
    # equal_length = tf.concat([waveform, zero_padding], 0)

    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    # add a channels dimention so that spectrogram become like an image
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram




def get_spectrogram_and_label(audio, label):
    spectrogram = get_spectrogram(audio)
    return spectrogram, label


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(
        map_func=get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE # type: ignore
    )
    output_ds = output_ds.map(
        map_func=get_spectrogram_and_label, num_parallel_calls=tf.data.AUTOTUNE
    )
    return output_ds
