# Adapted KERAS tutorial 
#%%
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, AlphaDropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


batch_size = 128
num_classes = 10
epochs = 20


# input image dimensions
img_rows, img_cols = 28, 28

# list devices so you can check whether your gpu is available
print(tf.config.list_physical_devices())

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
#x_train = (x_train - np.mean(x_train))/np.std(x_train)

x_test /= 255
#x_test = (x_test - np.mean(x_train))/np.std(x_train)

# create validation file
x_val = x_train[:10000]
x_train = x_train[10000:]
y_val = y_train[:10000]
y_train = y_train[10000:]


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'val samples')
print(x_test.shape[0], 'test samples')


# convert class vectors to one-hot vecotrs
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential([
    Flatten(input_shape = (28,28)),
    Dense(512, activation='selu',kernel_initializer='lecun_normal',bias_initializer='zeros'),
    AlphaDropout(0.05),
    Dense(256, activation='selu',kernel_initializer='lecun_normal',bias_initializer='zeros'),
    AlphaDropout(0.05),
    Dense(num_classes, activation='softmax',kernel_initializer='glorot_normal') #best practice to use glorot with softmax
])


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

#use early stopping callbacks
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 6)

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val),
          callbacks = [early_stopping_cb]
          )

#visualize training curves
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

