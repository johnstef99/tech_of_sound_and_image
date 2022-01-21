import matplotlib.pyplot as plt


def setup():
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['bottom'].set_color('white')


def plot_history(history):
    setup()
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.7, 1])
    plt.legend(loc='best', framealpha=0, labelcolor='white')
    plt.savefig('accuracy.png', dpi=200, transparent=True)
    plt.close()

    setup()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best', framealpha=0, labelcolor='white')
    plt.savefig('loss.png', dpi=200, transparent=True)
    plt.close()