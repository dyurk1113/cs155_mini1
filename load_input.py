import numpy as np
import os
import pandas

data_folder = 'data'


def get_col_ignore():
    col_ignore = np.genfromtxt(os.path.join(os.getcwd(), 'col_ignore.txt'), dtype='str', comments='#')
    return col_ignore


def load_data_file(file_name):
    data_file = os.path.join(os.getcwd(), data_folder, file_name)
    all_data = pandas.read_csv(data_file)
    trim_data = all_data.drop(get_col_ignore(), axis=1)
    return trim_data


def load_train_data():
    all_data = load_data_file('train_2008.csv')
    input = refactor_train_data(all_data)
    output = all_data['target']
    return input, output


def load_test_data(year):
    all_data = load_data_file('test_%s.csv' % year)
    input = refactor_train_data(all_data)
    return input


def refactor_train_data(train_data):
    train_data = load_train_data()
    print('Loaded Data')
    for col in train_data:
        train_data[col][train_data[col] < 0] = -1
    del train_data['target']
    print(np.shape(train_data))
    return pandas.get_dummies(train_data, columns=train_data.columns)
