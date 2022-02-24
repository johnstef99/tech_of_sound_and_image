import logging
import pandas as pd
import tensorflow as tf
from keras import models
from utils import plot_history
from mfcc import generate_or_load_mfccs
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras.utils import plot_model
from spectogram import generate_or_load_spectograms
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
# config logger
logging.basicConfig(level=logging.INFO,
                    format="---------------------\n%(levelname)s %(asctime)s:\n%(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def get_labels():
    return pd.read_csv('../../../physionet/samples/labels.csv')


def get_dataset(use_spectograms: bool = False):
    labels = get_labels()
    audios = labels.audio.values
    log.info(f"Using spectograms: {use_spectograms}")
    data = generate_or_load_spectograms(audios) if(
        use_spectograms) else generate_or_load_mfccs(audios)
    labels = labels.label.replace(-1, 0).values.reshape(len(labels), 1)
    x_train, x_test, y_train, y_test = train_test_split(data,
                                                        labels,
                                                        test_size=0.40,
                                                        random_state=69)
    return x_train, x_test, y_train, y_test


def create_model():
    log.info('Creating model')
    cnn = models.Sequential()
    cnn.add(Conv2D(32, (13, 1),
                   activation='relu',
                   padding='same',
                   input_shape=(13, 300, 1)))
    cnn.add(MaxPooling2D((2, 10)))

    cnn.add(Conv2D(64, (2, 5), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((3, 15)))
    cnn.add(Dropout(0.25))

    cnn.add(Flatten())

    cnn.add(Dense(64, activation='relu'))
    cnn.add(Dropout(0.25))

    cnn.add(Dense(1, activation='sigmoid'))
    cnn.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=BinaryCrossentropy(from_logits=False),
                metrics=[BinaryAccuracy(name='accuracy'),
                         Precision(name='precision'),
                         Recall(name='recall')])
    return cnn


def train_model(model, x_train, x_test, y_train, y_test):
    log.info('Start training')

    history = model.fit(x_train,
                        y_train,
                        epochs=100,
                        batch_size=32,
                        validation_data=(x_test, y_test))
    return history


def main():
    use_spectograms = True
    x_train, x_test, y_train, y_test = get_dataset(
        use_spectograms=use_spectograms)
    cnn = create_model()
    history = train_model(cnn, x_train, x_test, y_train, y_test)
    plot_model(cnn, 'model.png', show_layer_names=False,
               show_shapes=True, dpi=200)
    plot_history(history, use_spectograms=use_spectograms)
    print("Evaluation:")
    cnn.evaluate(x_test, y_test)
    return cnn, history


if __name__ == '__main__':
    cnn, history = main()
