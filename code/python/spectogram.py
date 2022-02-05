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


def generate_or_load_spectograms(samples: np.ndarray = None):
    if(exists('./spectograms.csv')):
        spectograms = pd.read_csv('./spectograms.csv', index_col='audio')
    else:
        if(samples is None):
            log.error(
                '''spectograms.csv file doesn\'n exists and you havent passed
                an array of samples to generate it''')
            exit(1)

        log.info(f'Start generating Spectograms for {len(samples)} samples')
        data = np.array(list(map(generate_melspectogram, samples)))
        log.info('Done generatinge Spectograms')

        data = data.reshape(len(data), 13*300)
        spectograms = pd.DataFrame(data)
        spectograms.set_index(samples, inplace=True)
        spectograms.to_csv('./spectograms.csv', index_label='audio')

    spectograms = np.array(spectograms)
    spectograms = spectograms.reshape(len(spectograms), 13, 300)
    spectograms = np.array([normalize(i) for i in spectograms])
    spectograms = spectograms.reshape(len(spectograms), 13, 300, 1)
    return spectograms
