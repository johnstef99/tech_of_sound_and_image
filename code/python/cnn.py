import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

data = pd.read_csv('../../../physionet/samples/labels.csv')

# create a function that generates this file
mfccs = pd.read_csv('./mfccs.csv')
mfccs.set_index('index', inplace=True)
mfccs = np.array(mfccs)
mfccs = mfccs.reshape(len(mfccs), 13, 299)
mfccs = np.array([normalize(i) for i in mfccs])
mfccs = mfccs.reshape(len(mfccs), 13, 299, 1)

x_train, x_test, y_train, y_test = train_test_split(mfccs,
                                                    data.label.replace(-1, 0)
                                                    .values.reshape(
                                                        len(data), 1),
                                                    test_size=0.25,
                                                    random_state=69)


cnn = models.Sequential()
cnn.add(layers.Conv2D(32, (3, 3), activation='tanh', input_shape=(13, 299, 1)))
cnn.add(layers.MaxPooling2D((1, 10)))

cnn.add(layers.Conv2D(64, (3, 3), activation='tanh'))
cnn.add(layers.MaxPooling2D((1, 4)))

cnn.add(layers.Flatten())
cnn.add(layers.Dense(32, activation='tanh'))

cnn.add(layers.Dense(1))
cnn.summary()
cnn.compile(optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'])

history = cnn.fit(x_train,
                  y_train,
                  epochs=50,
                  batch_size=10,
                  validation_data=(x_test, y_test))


test_loss, test_acc = cnn.evaluate(x_test, y_test, verbose=2)

print("HERE".center(40, '='))
print(f"Accuracy{test_acc}")
