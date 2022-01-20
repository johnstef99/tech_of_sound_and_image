from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import logging
import pandas as pd
import tensorflow as tf
from mfcc import generate_or_load_mfccs
from utils import plot_history
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import BinaryCrossentropy

# config logger
logging.basicConfig(level=logging.INFO,
                    format="---------------------\n%(levelname)s %(asctime)s:\n%(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def get_labels():
    return pd.read_csv('../../../physionet/samples/labels.csv')


def get_dataset():
    labels = get_labels()
    mfccs = generate_or_load_mfccs(labels.audio.values)
    labels = labels.label.replace(-1, 0).values.reshape(len(labels), 1)
    x_train, x_test, y_train, y_test = train_test_split(mfccs,
                                                        labels,
                                                        test_size=0.40,
                                                        random_state=69)
    return x_train, x_test, y_train, y_test


def create_model():
    log.info('Creating model')
    cnn = models.Sequential()
    cnn.add(Conv2D(32, (13, 1),
                   activation='tanh',
                   padding='same',
                   input_shape=(13, 300, 1)))
    cnn.add(MaxPooling2D((2, 10)))

    cnn.add(Conv2D(64, (2, 5), activation='tanh', padding='same'))
    cnn.add(MaxPooling2D((3, 15)))

    cnn.add(Flatten())

    cnn.add(Dense(64, activation='tanh'))

    cnn.add(Dense(1))
    cnn.compile(optimizer=tf.keras.optimizers.Adam(0.0008),
                loss=BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return cnn


def train_model(model, x_train, x_test, y_train, y_test):
    log.info('Start training')

    history = model.fit(x_train,
                        y_train,
                        epochs=50,
                        batch_size=32,
                        validation_data=(x_test, y_test))
    return history


def main():
    x_train, x_test, y_train, y_test = get_dataset()
    cnn = create_model()
    history = train_model(cnn, x_train, x_test, y_train, y_test)
    plot_model(cnn, 'model.png', show_layer_names=False,
               show_shapes=True, dpi=200)
    plot_history(history)
    test_loss, test_acc = cnn.evaluate(x_test, y_test)
    log.info(f"Accuracy: {test_acc}")
    return cnn, history


if __name__ == '__main__':
    cnn, history = main()
