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
# print('Loaded Data')


#No -1 Drop: 1192 cols, Train: 0.799919, Val: 0.778164
#Do -1 Drop: 1140 cols, Train: 0.803611, Val: 0.780484

num_cols = len(train_data.columns)
# pres_accs, miss_accs = [[] for _  in range(num_cols)], [[] for _ in range(num_cols)]
# dts, train_accs, val_accs = [], [], []
# for rep in range(num_reps):
#     layer_size = 10 + 5*rep
#     print(len(train_in[0]))
#     skf = StratifiedKFold(n_splits=5)
#     t_accs, v_accs = [], []
#     for t_inds, v_inds in skf.split(train_in, train_out):
model = Sequential()
in_shape = np.shape(train_in)
model.add(Dense(100, input_dim=in_shape[1], activation='relu'))
model.add(Dropout(0.2)) #Determined from hyperparameter optimization
model.add(Dense(20, activation='relu'))
# model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
        # print('HERE 1')
        # fit = model.fit(train_in[t_inds], train_out[t_inds], batch_size=32, epochs=3, verbose=0)
        # print('HERE 2')
        # score = model.evaluate(train_in[t_inds], train_out[t_inds], verbose=0)
        # t_accs.append(score[1])
        # score = model.evaluate(train_in[v_inds], train_out[v_inds], verbose=0)
        # v_accs.append(score[1])
    #     print('FOLD')
    # print('Train: %f, Val: %f' % (np.median(t_accs), np.median(v_accs)))
    # dts.append(layer_size)
    # train_accs.append(np.median(t_accs))
    # val_accs.append(np.median(v_accs))
# print('Rep %d, Acc %f' % (rep, val_accs[-1]))
# print(train_accs)
# print(val_accs)
# plt.plot(dts, train_accs, label='Train')
# plt.plot(dts, val_accs, label='Val')
# plt.legend()
# plt.show()
# print('Accuracy:', score[1])
#     print(rep, 'Accuracy:', score[1])
#     for col in range(num_cols):
#         if col in drop_cols:
#             miss_accs[col].append(score[1])
#         else:
#             pres_accs[col].append(score[1])
#
# results = [[col, np.mean(pres_accs[col]), np.std(pres_accs[col]),
#            np.mean(miss_accs[col]), np.std(miss_accs[col])] for col in range(num_cols)]
# np.savetxt('drop_col_res.csv', results, delimiter=',',
#            header='col,present mean,present std,dropped mean,dropped std',
#            comments='', fmt=['%d', '%.6f', '%.6f', '%.6f', '%.6f'])
#     accs.append(score[1])
#
# print(accs)
# plt.plot(regs, accs)
# plt.show()

fit = model.fit(train_in, train_out, batch_size=8, epochs=7, verbose=2)
out = model.predict(test_in_2008)
ids = np.reshape(np.arange(0, len(out), 1, dtype=np.int32), (len(out), 1))
out = np.concatenate((ids, out), axis=1)
np.savetxt('2008_out.csv', out, delimiter=',', header='id,target',
           fmt=['%d','%0.6f'], comments='')