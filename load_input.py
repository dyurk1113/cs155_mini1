import numpy as np
import os
import pandas

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
    trim_data = all_data.drop(get_col_ignore(), axis=1)
    return trim_data


def load_data(col_drop_rate = 0.0):
    train_data = load_data_file('train_2008.csv')
    train_out = train_data['target']
    train_out = train_out.astype(np.int8, copy=False)
    del train_data['target']

    test_data_2008 = load_test_data('2008')
    test_data_2012 = load_test_data('2012')
    all_data = pandas.concat([train_data, test_data_2008, test_data_2012])
    drop_inds = []
    if col_drop_rate > 0:
        num_cols = len(all_data.columns)
        num_drops = int(col_drop_rate * num_cols)
        drop_inds = gen_rand_inds(num_drops, num_cols)
        col_names = [all_data.columns[ind] for ind in drop_inds]
        all_data.drop(col_names, axis=1)
    cat_data = categorize_data(all_data)

    len_train, len_2008 = len(train_data), len(test_data_2008)
    train_cat = cat_data.iloc[:len_train]
    for col in train_cat:
        if (train_cat[col] == 0).all():
            del cat_data[col]

    cat_data = cat_data.values.astype(np.int8, copy=False)
    train_in = cat_data[:len_train]
    test_in_2008 = cat_data[len_train:len_train+len_2008]
    test_in_2012 = cat_data[len_train+len_2008:]
    return train_in, train_out, test_in_2008, test_in_2012, drop_inds


def load_test_data(year):
    return load_data_file('test_%s.csv' % year)
    # all_data = load_data_file('test_%s.csv' % year)
    # input = categorize_data(all_data).values
    # input = input.astype(np.int8, copy=False)
    # return input


def categorize_data(data):
    for col in data:
        data[col][data[col] < 0] = -1
    return pandas.get_dummies(data, columns=data.columns)

# load_data()