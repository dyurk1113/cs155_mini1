from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
import time

from load_input import load_raw_data, process_inputs


t = time.time()
train_data, train_out, test_data_2008, test_data_2012 = load_raw_data()
print('Raw Read Time:', time.time() - t)
num_reps, drop_rate, min_drop_thresh = 1, 0.0, 0.002
t = time.time()
train_in, test_in_2008, test_in_2012, drop_cols = process_inputs(train_data, test_data_2008, test_data_2012, drop_rate, min_drop_thresh)
print('Data Proc Time:', time.time() - t)

num_nn, in_frac = 50, 0.8
num_cols = train_in.shape[1]
sel_cols = int(in_frac * num_cols)
sub_train_out, sub_08_out, sub_12_out = np.zeros((train_in.shape[0], num_nn)), np.zeros((test_data_2008.shape[0], num_nn)),\
                                        np.zeros((test_data_2012.shape[0], num_nn))
val_acc_list = []
for n in range(num_nn):
    col_sel = np.random.choice(num_cols, sel_cols, replace=False)
    train, test_08, test_12 = train_in[:, col_sel], test_in_2008[:, col_sel], test_in_2012[:, col_sel]

    model = Sequential()
    model.add(Dense(80, input_dim=sel_cols, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(80, input_dim=sel_cols, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    skf = StratifiedKFold(n_splits=10)
    t_inds, v_inds = next(skf.split(train, train_out))

    fit = model.fit(train[t_inds], train_out[t_inds], batch_size=32, epochs=5, verbose=0)
    train_pred = model.predict(train)
    sub_train_out[:, n] = np.reshape(train_pred, (sub_train_out.shape[0]))
    test_pred_08 = model.predict(test_08)
    sub_08_out[:, n] = np.reshape(test_pred_08, (sub_08_out.shape[0]))
    test_pred_12 = model.predict(test_12)
    sub_12_out[:, n] = np.reshape(test_pred_12, (sub_12_out.shape[0]))

    val_pred = model.predict(train[v_inds])
    val_acc = sum([(1 if abs(pred - obv) < 0.5 else 0) for (pred, obv) in zip(val_pred, train_out[v_inds])])/len(v_inds)
    val_acc_list.append(val_acc)
    print('Random!', n)

acc_mean, acc_std = np.mean(val_acc_list), np.std(val_acc_list)
acc_thresh = acc_mean + 0.5 * acc_std
val_cols = np.array([c for c in range(num_nn) if val_acc_list[c] > acc_thresh])
net_train_out = np.median(sub_train_out[:, val_cols], axis=1)
net_test_08_out = np.median(sub_08_out[:, val_cols], axis=1)
net_test_12_out = np.median(sub_12_out[:, val_cols], axis=1)

train_acc = sum([(1 if abs(pred - obv) < 0.5 else 0) for (pred, obv) in zip(net_train_out, train_out)])/train_out.shape[0]
print(train_acc)

ids = np.reshape(np.arange(0, len(net_test_08_out), 1, dtype=np.int32), (net_test_08_out.shape[0], 1))
net_test_08_out = np.reshape(net_test_08_out, (net_test_08_out.shape[0], 1))
out = np.concatenate((ids, net_test_08_out), axis=1)
np.savetxt('2008_out.csv', out, delimiter=',', header='id,target',
           fmt=['%d','%0.6f'], comments='')

ids = np.reshape(np.arange(0, len(net_test_12_out), 1, dtype=np.int32), (net_test_12_out.shape[0], 1))
net_test_12_out = np.reshape(net_test_12_out, (net_test_12_out.shape[0], 1))
out = np.concatenate((ids, net_test_12_out), axis=1)
np.savetxt('2012_out.csv', out, delimiter=',', header='id,target',
           fmt=['%d','%0.6f'], comments='')