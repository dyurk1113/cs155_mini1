from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import numpy as np
from matplotlib import pyplot as plt

from load_input import load_data

train_in, train_out, test_in_2008, test_in_2012, drop_cols = load_data()
# print('Loaded Data')

num_cols = len(train_in[0])
num_reps, drop_rate = 100, 0.2
pres_accs, miss_accs = [[] for _  in range(num_cols)], [[] for _ in range(num_cols)]
for rep in range(num_reps):
    print('HERE 0')
    train_in, train_out, test_in_2008, test_in_2012, drop_cols = load_data(drop_rate)
    model = Sequential()
    in_shape = np.shape(train_in)
    model.add(Dense(100, input_dim=in_shape[1], activation='relu'))
    model.add(Dropout(0.15)) #Determined from hyperparameter optimization
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print('HERE 1')
    fit = model.fit(train_in, train_out, batch_size=32, epochs=3, verbose=0)
    print('HERE 2')
    score = model.evaluate(train_in, train_out, verbose=0)
    print(rep, 'Accuracy:', score[1])
    for col in range(num_cols):
        if col in drop_cols:
            miss_accs[col].append(score[1])
        else:
            pres_accs[col].append(score[1])

results = [[col, np.mean(pres_accs[col]), np.std(pres_accs[col]),
           np.mean(miss_accs[col]), np.std(miss_accs[col])] for col in range(num_cols)]
np.savetxt('drop_col_res.csv', results, delimiter=',',
           header='col,present mean,present std,dropped mean,dropped std',
           comments='', fmt=['%d', '%.6f', '%.6f', '%.6f', '%.6f'])
#     accs.append(score[1])
#
# print(accs)
# plt.plot(regs, accs)
# plt.show()

# out = model.predict(test_in_2008)
# ids = np.reshape(np.arange(0, len(out), 1, dtype=np.int32), (len(out), 1))
# out = np.concatenate((ids, out), axis=1)
# np.savetxt('2008_out.csv', out, delimiter=',', header='id,target',
#            fmt=['%d','%0.6f'], comments='')