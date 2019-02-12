from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
import time

from load_input import load_raw_data, process_inputs

#5x NN, 0.5 Rate: Train 78.14, Val 77.59
#1x NN, 1.0 Rate: 78.55, 77.76
#5x NN, 0.75 Rate: 78.67, 77.82
#10x NN, 0.8 Rate, -1-sig dropout: 78.87, 77.99
#10x NN, 0.5 Rate, -1-sig dropout: 78.33, 77.67
#10x NN, 0.8 Rate, 0-sig dropout: 78.96, 77.95
#10x NN, 0.8 Rate, -1-sig dropout, extra layer: 78.91, 78.03
#10x NN, 0.8 Rate, -1-sig dropout, 2x extra layer: 78.97, 77.97
#10x NN, 0.8 Rate, -1-sig dropout, 2x extra layer, 0.2 dropout: 78.59, 77.84
#10x NN, 0.8 Rate, -1-sig dropout, extra layer, extra little layer:78.78, 78.00

t = time.time()
train_data, train_out, test_data_2008, test_data_2012 = load_raw_data()
print('Raw Read Time:', time.time() - t)
num_reps, drop_rate, min_drop_thresh = 1, 0.0, 0.002
t = time.time()
train_in, test_in_2008, test_in_2012, drop_cols = process_inputs(train_data, test_data_2008, test_data_2012, drop_rate, min_drop_thresh)
print('Data Proc Time:', time.time() - t)
# print('Loaded Data')


#No -1 Drop: 1192 cols, Train: 0.799919, Val: 0.778164
#Do -1 Drop: 1140 cols, Train: 0.803611, Val: 0.780484

num_nn, in_frac = 10, 0.8
num_cols = train_in.shape[1]
sel_cols = int(in_frac * num_cols)

skf = StratifiedKFold(n_splits=5)
t_accs, v_accs = [], []
for t_inds, v_inds in skf.split(train_in, train_out):
    sub_train_out, sub_val_out = np.zeros((len(t_inds), num_nn)), np.zeros((len(v_inds), num_nn))
    t_out, v_out = train_out[t_inds], train_out[v_inds]
    train_acc_list = []
    for n in range(num_nn):
        col_sel = np.random.choice(num_cols, sel_cols, replace=False)
        train, val = train_in[t_inds, :][:, col_sel], train_in[v_inds, :][:, col_sel]

        model = Sequential()
        model.add(Dense(80, input_dim=sel_cols, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(80, input_dim=sel_cols, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        fit = model.fit(train, t_out, batch_size=128, epochs=1, verbose=0)
        train_pred = model.predict(train)
        sub_train_out[:, n] = np.reshape(train_pred, (sub_train_out.shape[0]))
        train_acc = sum([(1 if abs(pred - obv) < 0.5 else 0) for (pred, obv) in zip(train_pred, t_out)])/len(t_inds)
        train_acc_list.append(train_acc)
        sub_val_out[:, n] = np.reshape(model.predict(val), (sub_val_out.shape[0]))
        print('Random!')

    acc_mean, acc_std = np.mean(train_acc_list), np.std(train_acc_list)
    acc_thresh = acc_mean - acc_std
    val_cols = np.array([c for c in range(num_nn) if train_acc_list[c] > acc_thresh])
    net_train_out = np.median(sub_train_out[:, val_cols], axis=1)
    net_val_out = np.median(sub_val_out[:, val_cols], axis=1)

    train_acc = sum([(1 if abs(pred - obv) < 0.5 else 0) for (pred, obv) in zip(net_train_out, t_out)])/len(t_inds)
    val_acc = sum([(1 if abs(pred - obv) < 0.5 else 0) for (pred, obv) in zip(net_val_out, v_out)])/len(v_inds)
    print(train_acc, val_acc)
    t_accs.append(train_acc)
    v_accs.append(val_acc)
print(np.median(t_accs), np.median(v_accs))

# pres_accs, miss_accs = [[] for _  in range(num_cols)], [[] for _ in range(num_cols)]
# dts, train_accs, val_accs = [], [], []
# for rep in range(num_reps):
#     layer_size = 10 + 5*rep
#     print(len(train_in[0]))
#     skf = StratifiedKFold(n_splits=5)
#     t_accs, v_accs = [], []
#     for t_inds, v_inds in skf.split(train_in, train_out):
# model = Sequential()
# in_shape = np.shape(train_in)
# model.add(Dense(50, input_dim=in_shape[1], activation='relu'))
# model.add(Dropout(0.15)) #Determined from hyperparameter optimization
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# fit = model.fit(train_in, train_out, batch_size=8, epochs=7, verbose=2)
# out = model.predict(test_in_2008)
# ids = np.reshape(np.arange(0, len(out), 1, dtype=np.int32), (len(out), 1))
# out = np.concatenate((ids, out), axis=1)
# np.savetxt('2008_out.csv', out, delimiter=',', header='id,target',
#            fmt=['%d','%0.6f'], comments='')