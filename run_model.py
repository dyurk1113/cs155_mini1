import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
import numpy as np

from load_input import load_train_data

input, output = load_train_data()
model = Sequential()
in_shape = np.shape(input)
model.add(Dense(50, input_shape=in_shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
fit = model.fit(input, output, batch_size=32, nb_epoch=10, verbose=1)
score = model.evaluate(input, output, verbose=0)
print('Accuracy:', score[1])