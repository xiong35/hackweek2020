

from keras import layers
from keras.models import Sequential
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
# np.random.seed(7)


def rand_gen_data(n=None):
    if not n:
        n = np.random.randint(10, 30)
    mean = np.random.uniform(35, 55)
    scaler = np.random.uniform(mean-5, mean+5, n)
    x = np.linspace(-4, 4, n)
    sig = 1/(1+np.exp(-x))
    y = sig*(1-sig)
    y *= scaler
    # plt.plot(x, y)
    # plt.show()
    return y.tolist()


lookback = 4
data_enhance_num = 30
rand_range = 0.07


def data_preprocess(data_seqs, enhance=True):

    train_data = []
    for data_seq in data_seqs:
        if len(data_seq[0]) < lookback + 1:
            continue
        for i in range(0, len(data_seq[0])-lookback):
            one_data = data_seq[:, i:i + lookback + 1]
            if enhance:
                for _ in range(data_enhance_num):
                    temp = one_data * \
                        np.random.normal(1, rand_range, (3, lookback + 1))
                    train_data.append(temp.T)
            else:
                train_data.append(one_data.T)
    train_data = np.array(train_data)
    return train_data


def load_json():
    with open(R'.\world_data.json') as f:
        lines = f.readlines()

    json_str = ''
    for line in lines:
        json_str += line

    world_data_dict = json.loads(json_str)
    return world_data_dict


def dict2array(world_data_dict):
    """
    take a dict:  
    {'US':{'confirm':[1,2,3],'deaths':...}}

    return list of np array(each array for one counrty):  
    array[[1,2,3(confirm)],  
          [2,3,4(deaths)],  
          [4,5,6(recover)]]  
    """
    data_seqs = []
    for country, info_dic in world_data_dict.items():
        country_list = []
        for data_type, data_list in info_dic.items():
            country_list.append(data_list)
        country_array = np.array(country_list)
        data_seqs.append(country_array)
    return data_seqs


world_data_dict = load_json()
data_seqs = dict2array(world_data_dict)

test_num = 8

data_set = data_preprocess(data_seqs[:test_num]+data_seqs[test_num+1:])

num, step, feature_num = data_set.shape
x_train = data_set[:, :step-1, :]
y_train = data_set[:, step-1, :]


model = Sequential()
model.add(layers.LSTM(8,
                      dropout=0.2,
                      recurrent_dropout=0.2,
                      return_sequences=False,
                      input_shape=(None, feature_num)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(3))

model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit(x_train, y_train,
                    batch_size=128, epochs=30, validation_split=0.2, shuffle=True)

test = [data_seqs[test_num]]
test_data = data_preprocess(test, enhance=False)
y_test = test_data[:, step-1, :]
x_test = test_data[:, :step-1, :]

y_pred = model.predict(x_test)

days = range(1, len(y_pred[:, 0])+1)

pred_comfirmed = y_pred[:, 0]
pred_deaths = y_pred[:, 1]
pred_recovered = y_pred[:, 2]

confirmed = y_test[:, 0]
deaths = y_test[:, 1]
recovered = y_test[:, 2]

plt.plot(days, pred_comfirmed, 'bo', alpha=0.7, label='pred comfirmed')
plt.plot(days, y_test, 'b', alpha=0.7, label='real comfirmed')

plt.plot(days, pred_deaths, 'ro', alpha=0.7, label='pred deaths')
plt.plot(days, deaths, 'r', alpha=0.7, label='real deaths')

plt.plot(days, pred_recovered, 'go', alpha=0.7, label='pred recovered')
plt.plot(days, recovered, 'g', alpha=0.7, label='real recovered')

plt.legend()
plt.show()

# model.save('/root/MySource/LSTM0.h5')
