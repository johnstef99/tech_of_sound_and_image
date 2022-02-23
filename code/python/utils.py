import matplotlib.pyplot as plt


def setup():
    fig = plt.figure(figsize=(16, 9))
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
    plt.savefig(prefix+'accuracy.png', dpi=80,
                transparent=False, bbox_inches='tight')
    plt.close()

    setup()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best', framealpha=0)
    plt.savefig(prefix+'loss.png', dpi=80,
                transparent=False, bbox_inches='tight')
    plt.close()

    setup()
    plt.plot(history.history['precision'], label='loss')
    plt.plot(history.history['val_precision'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend(loc='best', framealpha=0)
    plt.savefig(prefix+'precision.png', dpi=80,
                transparent=False, bbox_inches='tight')
    plt.close()
