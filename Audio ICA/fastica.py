# https://programming.rhysshea.com/Independent_component_analysis/
from sklearn.decomposition import FastICA
from scipy.io import wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import wave
import os

PLOT_PATH = "plots/"
AUDIO_PATH = "audio/"
AUDIO_FILE_1 = "bboombboom.wav"
AUDIO_FILE_2 = "baam.wav"
AUDIO_SEP_FILE = "sources.wav"


def plot_output(waveform, title, filename):
    plt.figure(figsize=(12, 2))
    plt.title(title)
    plt.plot(waveform)
    plt.savefig(os.path.join(PLOT_PATH, filename), dpi=100)
    print("Showing plot for {}".format(title))
    plt.show()


def plot_channel(waveform, title, filename, color):
    plt.figure(figsize=(12, 2))
    plt.title(title)
    plt.plot(waveform, color=color)
    plt.savefig(os.path.join(PLOT_PATH, filename), dpi=100)
    print("Showing plot for {}".format(title))
    plt.show()


def get_waveform(waveform, filename, savefile, color="#3ABFE7"):
    signal_raw = waveform.readframes(-1)
    signal = np.fromstring(signal_raw, 'Int16')
    rate = waveform.getframerate()
    timing = np.linspace(0, len(signal)/rate, num=len(signal))
    plt.figure(figsize=(12, 2))
    plt.title(filename)
    plt.plot(timing, signal, c=color)
    plt.ylim(-35000, 35000)
    plt.savefig(os.path.join(PLOT_PATH, "plot-{}".format(savefile)), dpi=100)
    print("Showing plot for {}".format(filename))
    plt.show()
    return {"signal": signal, "rate": rate, "timing": timing}


if __name__ == "__main__":
    data_1 = wave.open(os.path.join(AUDIO_PATH, AUDIO_FILE_1), 'r')
    data_2 = wave.open(os.path.join(AUDIO_PATH, AUDIO_FILE_2), 'r')

    audio_1 = get_waveform(data_1, AUDIO_FILE_1, "audio_1")
    audio_2 = get_waveform(data_2, AUDIO_FILE_2, "audio_2", color="orange")

    X = list(zip(audio_1["signal"], audio_2["signal"]))

    # https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html
    ica = FastICA(n_components=2)
    transformed = ica.fit_transform(X)  # Reconstruct signals
    print(transformed.shape)
    plot_output(transformed[:np.int(len(transformed)/4),
                            :], title="Transformed", filename="transformed")
    plot_channel(transformed[:np.int(len(transformed)/4),
                             0], title="ICA Component 1", filename="ica-component-1", color="#3ABFE7")
    plot_channel(transformed[:np.int(len(transformed)/4),
                             1], title="ICA Component 2", filename="ica-component-2", color="orange")
    wav.write(os.path.join(AUDIO_PATH, AUDIO_SEP_FILE), 2 *
              audio_1["rate"], np.int16(transformed * 3500000))
