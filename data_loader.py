import pandas as pd
import numpy as np
from utils import binary_sampler
from keras.datasets import fashion_mnist, mnist

def data_loader(data_name, miss_rate):
    if data_name in ['parkinson', 'gene']:
        file_name = '/kaggle/input/dataset/'+data_name+'.csv/'+data_name+'.csv'
        if data_name == 'parkinson':
            data = pd.read_csv("/kaggle/input/parkinson-dataset/pd_speech_features.csv")
            y = data.loc[:,'class'].to_numpy()
            x = data.drop(['class', 'id'], axis=1).to_numpy()
            data_x, x_test, y_train, y_test = x, None, y, None
        else:
            x = pd.read_csv(file_name)
            x = x.loc[:, ~x.columns.str.contains('^Unnamed')]
            file_labels = "/kaggle/input/impacc-data/TCGA-PANCAN-HiSeq/labels.csv"
            y = pd.read_csv(file_labels)
            y = y.loc[:, ~y.columns.str.contains('^Unnamed')]
            label_list = y['Class'].unique().tolist()
            label_list = {cls: idx for idx, cls in enumerate(label_list)}
            y['Class'] = y['Class'].map(label_list)
            y = y['Class'].to_numpy()
            x = np.reshape(np.asarray(x), [x.shape[0], -1]).astype(float)
            data_x, x_test, y_train, y_test = x, None, y, None

    elif data_name == 'mnist':
        (data_x, y_train), (x_test, y_test)= mnist.load_data()
        data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)
        x_test = np.reshape(np.asarray(x_test), [-1, 28*28]).astype(float)
    else:
        (data_x, y_train), (x_test, y_test) = fashion_mnist.load_data()
        data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)
        x_test = np.reshape(np.asarray(x_test), (-1, 28*28)).astype(float)

    no, dim = data_x.shape
    data_m = binary_sampler(1-miss_rate, no, dim)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan
    data_train = (data_x, y_train)
    data_test = (x_test, y_test)
    return data_train, data_test, miss_data_x, data_m