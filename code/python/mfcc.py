import logging
import numpy as np
import pandas as pd
from os.path import exists
from librosa import load as load_wav
from matplotlib import pyplot as plt
from librosa.display import specshow
from python_speech_features import mfcc
from sklearn.preprocessing import normalize

log = logging.getLogger(__name__)


def read_audio(name: str):
    """
    Parameters
    ----------
    name : str
        name of the sample.
        Ex: for sample a0001.wav the name is a0001

    Returns
    -------
    y    : np.ndarray [shape=(n,) or (2, n)]
        audio time series

    sr   : number > 0 [scalar]
        sampling rate of ``y``
    """
    path = '../../../physionet/samples/all'
    audio_file = f"{path}/{name}.wav"
    return load_wav(audio_file, sr=2000)


def get_s1(name: str):
    """
    Parameters
    ----------
    name : str
        name of the sample.
        Ex: for sample a0001.wav the name is a0001


    Returns
    -------
    s1   : int
        miliseconds of the first S1 for the passed sample

    """
    s1_df: pd.DataFrame = pd.read_csv('../matlab/s1.csv', sep=',')
    s1_df.set_index('audio', inplace=True)
    s1 = s1_df.loc[name].s1
    return s1


def generate_mfcc(name: str, savefig: bool = False, show: bool = False):
    """
    Parameters
    ----------
    name    : str
        name of the sample.
        Ex: for sample a0001.wav the name is a0001

    savefig : bool
        if True the figure of the heatmap will be saved to name.png

    show : bool
        if True plt.show() will run

    Returns
    -------
    mfccs   : np.ndarray
        array that contains all the mfccs values

    """
    signal, sr = read_audio(name)
    s1_ms = get_s1(name)
    s1_index = int(np.round((s1_ms) * sr/1000)) - 1
    s1_index = s1_index if s1_index >= 0 else 0  # check for negative index
    # get samples from s1 to s1 + 3 seconds
    samples = signal[s1_index:(s1_index+20+3*sr)]
    mfccs = mfcc(samples, sr, winlen=0.02, winstep=0.01, appendEnergy=True)
    mfccs = mfccs.T
    if(savefig):
        fig, ax = plt.subplots(figsize=(16, 4))
        img = specshow(mfccs, sr=sr, x_axis="time", ax=ax,
                       hop_length=int(sr*0.01))
        fig.colorbar(img, ax=ax)
        ax.set(title=f"MFCC of {name}")
        fig.savefig(f"{name}_mfcc.png", bbox_inches='tight', dpi=120)
    else:
        if(show):
            fig, ax = plt.subplots()
            img = specshow(mfccs, sr=sr, x_axis="time", ax=ax,
                           hop_length=int(sr*0.01))
            fig.colorbar(img, ax=ax)
            ax.set(title=f"MFCC of {name}")
            fig.show()
    return mfccs


def generate_or_load_mfccs(samples: np.ndarray = None):
    if(exists('./mfccs.csv')):
        mfccs = pd.read_csv('./mfccs.csv', index_col='audio')
    else:
        if(samples is None):
            log.error(
                '''mfccs.csv file doesn\'n exists and you havent passed an
                array of samples to generate it''')
            exit(1)

        log.info(f'Start generating MFCCs for {len(samples)} samples')
        data = np.array(list(map(generate_mfcc, samples)))
        log.info('Done generatinge MFCCs')

        data = data.reshape(len(data), 13*300)
        mfccs = pd.DataFrame(data)
        mfccs.set_index(samples, inplace=True)
        mfccs.to_csv('./mfccs.csv', index_label='audio')

    mfccs = np.array(mfccs)
    mfccs = mfccs.reshape(len(mfccs), 13, 300)
    mfccs = np.array([i+abs(np.min(i)) for i in mfccs])
    mfccs = np.array([normalize(i) for i in mfccs])
    mfccs = mfccs.reshape(len(mfccs), 13, 300, 1)
    return mfccs
