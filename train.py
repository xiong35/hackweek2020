

from keras import layers
from keras.models import Sequential, load_model
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import json


json_file = 'world_confirmed_data.json'

lookback = 4
data_enhance_num = 7
rand_range = 0.07


def data_preprocess(data_seqs, enhance=True):

    train_data = []
    for data_seq in data_seqs:
        if len(data_seq) < lookback + 1:
            continue
        cur_enhance = int(data_enhance_num*np.log(data_seq[-1]))
        for i in range(0, len(data_seq)-lookback):
            one_data = data_seq[i:i + lookback + 1]
            one_data = np.array(one_data)
            if enhance:
                for _ in range(cur_enhance):
                    rand = np.random.uniform(
                        1-rand_range, 1+rand_range, lookback + 1)
                    temp = one_data * rand

                    train_data.append(temp)
            else:
                train_data.append(one_data)
    train_data = np.array(train_data)
    return train_data


def read_json(filename):
    with open(filename) as fr:
        lines = fr.readlines()

    json_str = ''
    for line in lines:
        json_str += line

    world_data_dict = json.loads(json_str)

    world_data = []

    for country, num_list in world_data_dict.items():
        world_data.append(num_list)

    return world_data


data_seqs = read_json(json_file)

data_set = data_preprocess(data_seqs)

num, step = data_set.shape
x_train = data_set[:, :lookback].reshape(num, step-1, 1)
y_train = data_set[:, lookback]


# model = Sequential()
# model.add(layers.LSTM(8,
#                       dropout=0.3,
#                       recurrent_dropout=0.3,
#                       return_sequences=False,
#                       input_shape=(lookback, 1)))
# model.add(layers.Dense(8, activation='relu'))
# model.add(layers.Dense(1))

# model.compile(optimizer=RMSprop(learning_rate=0.003), loss='mae')

# callbacks = [
#     ModelCheckpoint('LSTM.h5', verbose=1, save_best_only=True)
# ]

# history = model.fit(x_train, y_train,
#                     batch_size=128, epochs=35,
#                     validation_split=0.2, shuffle=True,
#                     callbacks=callbacks)

for test_num in range(21, 87, 9):
    model = load_model('LSTM.h5')
    test = [data_seqs[test_num]]
    test_data = data_preprocess(test, enhance=True)
    if test_data.size == 0:
        continue
    num, step = test_data.shape
    x_train = test_data[:, :lookback].reshape(num, step-1, 1)
    y_train = test_data[:, lookback]
    model.fit(x_train, y_train, batch_size=16,
              epochs=int(num/50), shuffle=True)

    test_data = data_preprocess(test, enhance=False)
    num, step = test_data.shape
    y_test = test_data[:, lookback]
    x_test = test_data[:, :lookback].reshape(num, step-1, 1)

    y_pred = model.predict(x_test)

    days = range(1, len(y_pred)+1)

    plt.plot(days, y_pred, c='b', alpha=0.7,
             label='pred'+str(test_num))
    plt.plot(days, y_test, linestyle='--', c='r',
             alpha=0.7, label='real'+str(test_num),)
plt.show()

# model.save('/root/MySource/LSTM0.h5')
