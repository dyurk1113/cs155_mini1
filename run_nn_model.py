import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
import numpy as np

from load_input import load_data

train_in, train_out, test_in_2008, test_in_2012 = load_data()
print('Loaded Data')
model = Sequential()
in_shape = np.shape(train_in)
model.add(Dense(100, input_dim=in_shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
fit = model.fit(train_in, train_out, batch_size=32, epochs=5, verbose=2)
score = model.evaluate(train_in, train_out, verbose=0)
print('Accuracy:', score[1])

out = model.predict(test_in_2008)
print(np.min(out), np.max(out))
ids = np.reshape(np.arange(0, len(out), 1, dtype=np.int32), (len(out), 1))
out = np.concatenate((ids, out), axis=1)
np.savetxt('2008_out.csv', out, delimiter=',', header='id,target',
           fmt=['%d','%0.6f'], comments='')