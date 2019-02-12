import numpy as np
import os
import pandas
import time

data_folder = 'data'


def gen_rand_inds(num_inds, data_len):
    rand_inds = []
    while len(rand_inds) < num_inds:
        ind = np.random.randint(0, data_len)
        if not ind in rand_inds:
            rand_inds.append(ind)
    return rand_inds


def get_col_ignore():
    col_ignore = np.genfromtxt(os.path.join(os.getcwd(), 'col_ignore.txt'), dtype='str', comments='#')
    return col_ignore


def load_data_file(file_name):
    data_file = os.path.join(os.getcwd(), data_folder, file_name)
    all_data = pandas.read_csv(data_file)
    all_data.drop(get_col_ignore(), axis=1, inplace=True)
    return all_data


def load_raw_data():
    train_data = load_data_file('train_2008.csv')
    train_out = train_data['target']
    train_out = train_out.astype(np.int8, copy=False)
    del train_data['target']

    test_data_2008 = load_test_data('2008')
    test_data_2012 = load_test_data('2012')
    return train_data, train_out, test_data_2008, test_data_2012


def process_inputs(train_data, test_data_2008, test_data_2012, col_drop_rate = 0.0, min_drop_thresh = 0.002):
    all_data = pandas.concat([train_data, test_data_2008, test_data_2012])
    drop_inds = []
    if col_drop_rate > 0:
        num_cols = len(all_data.columns)
        num_drops = int(col_drop_rate * num_cols)
        drop_inds = gen_rand_inds(num_drops, num_cols)
        col_names = [all_data.columns[ind] for ind in drop_inds]
        all_data.drop(col_names, axis=1)
    t = time.time()
    cat_data = categorize_data(all_data)
    # print('Cat Time:', time.time()-t)

    t = time.time()
    len_train, len_2008 = len(train_data), len(test_data_2008)
    train_cat = cat_data.iloc[:len_train].values.astype(np.int8)
    drop_thresh = len_train * min_drop_thresh
    drop_cols = []
    for col in range(len(cat_data.columns)):#train_cat:
        num_hot = np.count_nonzero(train_cat[:, col])
        if num_hot <= drop_thresh or num_hot >= (len_train - num_hot):
            drop_cols.append(col)
            # del cat_data[col]
    cat_data.drop([cat_data.columns[c] for c in drop_cols], axis=1, inplace=True)
    # print('Thresh Drop Time:', time.time() - t)

    cat_data = cat_data.values.astype(np.int8)
    train_in = cat_data[:len_train]
    test_in_2008 = cat_data[len_train:len_train+len_2008]
    test_in_2012 = cat_data[len_train+len_2008:]
    return train_in, test_in_2008, test_in_2012, drop_inds


def load_test_data(year):
    return load_data_file('test_%s.csv' % year)
    # all_data = load_data_file('test_%s.csv' % year)
    # input = categorize_data(all_data).values
    # input = input.astype(np.int8, copy=False)
    # return input


def categorize_data(data):
    for col in data:
        data[col][data[col] < 0] = -1
    cat_data = pandas.get_dummies(data, columns=data.columns)#, drop_first=True)#, sparse=True)
    neg_cols = [col for col in cat_data if '_-1' in col]
    t = time.time()
    cat_data.drop(neg_cols, axis=1, inplace=True)
    # print('Drop Time:', time.time() - t)
    return cat_data

# load_data()