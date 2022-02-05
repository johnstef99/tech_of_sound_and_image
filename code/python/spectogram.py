from signal import signal
from turtle import shape
import numpy as np
import librosa
import matplotlib.pyplot as plt
from mfcc import read_audio, get_s1
import librosa.display


def generateMel_Spectogram(name: str):
    """
    Parameters
    ----------
    name    : str
        name of the sample.
        Ex: for sample a0001.wav the name is a0001
    """
    signal, sr = read_audio(name)
    s1_ms = get_s1(name)
    s1_index = int(np.round((s1_ms) * sr/1000)) - 1
    # check for negative index
    s1_index = s1_index if s1_index >= 0 else 0
    # get samples from s1 to s1 + 3 seconds
    samples = signal[s1_index:(s1_index-20+3*sr)]
    print(samples.shape)
    ps = librosa.feature.melspectrogram(
        y=samples, sr=sr, hop_length=int(sr*0.01), win_length=int(sr*0.005), n_mels=13)
    ps_dB = librosa.power_to_db(ps, ref=np.max)
    #fig, ax = plt.subplots()
    img = librosa.display.specshow(ps_dB, sr=sr, x_axis="time",
                                   y_axis="mel", hop_length=int(sr*0.01), ax=ax)
    #fig.colorbar(img, ax=ax, format='%+2.0f dB')
    #ax.set(title="Mel-Frequency Spectogram")
    return ps_dB
