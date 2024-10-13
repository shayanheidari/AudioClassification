import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
from matplotlib import pyplot as plt


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




def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns),
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


def plot_wave_spectrogram_and_audio(num, example_audio, example_label, label_names):
    for i in range(num):
        label = label_names[example_label[i]]
        waveform = example_audio[i]
        white_waveform, _ = add_white_noise(example_audio[i], label)
        spectrogram = get_spectrogram(waveform)
        spectrogram_white = get_spectrogram(white_waveform)

        print("Label: ", label)
        print("Waveform shape: ", waveform.shape)
        print("Spectrogram shape: ", spectrogram.shape)
        print("Audio playback")
        display.display(display.Audio(waveform, rate=16000))
        display.display(display.Audio(white_waveform, rate=16000))

        plt.figure(figsize=(9, 5))
        timescale = np.arange(waveform.shape[0])
        plt.subplot(2, 2, 1)
        plt.plot(timescale, waveform.numpy())
        plt.title("waveform")
        plt.xlim(0, 16000)

        plt.subplot(2, 2, 2)
        plot_spectrogram(spectrogram.numpy(), plt)
        plt.title("Spectrogram")
        plt.suptitle(label.title())

        plt.subplot(2, 2, 3)
        timescale = np.arange(white_waveform.shape[0])
        plt.plot(timescale, white_waveform.numpy())
        plt.title("white_waveform")
        plt.xlim(0, 16000)

        plt.subplot(2, 2, 4)
        plot_spectrogram(spectrogram_white.numpy(), plt)
        plt.title("white_Spectrogram")
        plt.suptitle(label.title())
        plt.show()

