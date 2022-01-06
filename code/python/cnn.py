from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import logging
import pandas as pd
import tensorflow as tf
from mfcc import generate_or_load_mfccs
from matplotlib import pyplot as plt

# config logger
logging.basicConfig(level=logging.INFO,
                    format="---------------------\n%(levelname)s %(asctime)s:\n%(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def get_labels():
    return pd.read_csv('../../../physionet/samples/labels.csv')


def plot_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.7, 1])
    plt.legend(loc='lower right')
    plt.show()


def main():
    labels = get_labels()
    mfccs = generate_or_load_mfccs(labels.audio.values)
    labels = labels.label.replace(-1, 0).values.reshape(len(labels), 1)
    x_train, x_test, y_train, y_test = train_test_split(mfccs,
                                                        labels,
                                                        test_size=0.25,
                                                        random_state=69)

    log.info('Creating model')
    cnn = models.Sequential()
    cnn.add(Conv2D(32, (3, 3), activation='tanh', input_shape=(13, 299, 1)))
    cnn.add(MaxPooling2D((1, 10)))

    cnn.add(Conv2D(64, (3, 3), activation='tanh'))
    cnn.add(MaxPooling2D((1, 4)))

    cnn.add(Flatten())
    cnn.add(Dense(32, activation='tanh'))

    cnn.add(Dense(1))
    cnn.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

    log.info('Start training')

    history = cnn.fit(x_train,
                      y_train,
                      epochs=50,
                      batch_size=int(len(x_train)/300),
                      validation_data=(x_test, y_test))

    plot_history(history)

    test_loss, test_acc = cnn.evaluate(x_test, y_test)

    log.info(f"Accuracy: {test_acc}")


if __name__ == '__main__':
    main()
