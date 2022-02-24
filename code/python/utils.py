import matplotlib.pyplot as plt
from mfcc import read_audio, get_s1
import numpy as np


def setup():
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    # ax.xaxis.label.set_color('white')
    # ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    # ax.spines['left'].set_color('white')
    # ax.spines['top'].set_color('white')
    # ax.spines['right'].set_color('white')
    # ax.spines['bottom'].set_color('white')


def plot_history(history, use_spectograms=False):
    prefix = 'spectogram_' if use_spectograms else 'mfcc_'
    setup()
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.7, 1])
    plt.legend(loc='best', framealpha=0)
    plt.savefig(prefix+'accuracy.png', dpi=120,
                transparent=False, bbox_inches='tight')
    plt.close()

    setup()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best', framealpha=0)
    plt.savefig(prefix+'loss.png', dpi=120,
                transparent=False, bbox_inches='tight')
    plt.close()

    setup()
    plt.plot(history.history['precision'], label='loss')
    plt.plot(history.history['val_precision'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend(loc='best', framealpha=0)
    plt.savefig(prefix+'precision.png', dpi=120,
                transparent=False, bbox_inches='tight')
    plt.close()

    setup()
    plt.plot(history.history['recall'], label='loss')
    plt.plot(history.history['val_recall'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('recall')
    plt.legend(loc='best', framealpha=0)
    plt.savefig(prefix+'recall.png', dpi=120,
                transparent=False, bbox_inches='tight')
    plt.close()


def plot_audio(audio: str):
    sample, sr = read_audio(audio)
    s1 = get_s1(audio)
    s1_index = int(s1 * sr / 1000) - 1
    cutted = sample[s1_index: (s1_index + 3*sr + 20)]
    time = np.linspace(0, len(cutted)/sr, num=len(cutted))
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set(title=f"Sample {audio}", xlabel="Time")
    ax.plot(time, cutted)
    fig.savefig(f"{audio}.png", bbox_inches="tight", dpi=120)
