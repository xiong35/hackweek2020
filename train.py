

from keras import layers
from keras.models import Sequential
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
np.random.seed(7)


def rand_gen_data():
    n = np.random.randint(20, 40)
    scaler = np.random.uniform(30, 50)
    x = np.linspace(-4, 4, n)
    sig = 1/(1+np.exp(-x))
    y = sig*(1-sig)
    y *= scaler
    # plt.plot(x, y)
    # plt.show()
    return y.tolist()


lookback = 4
data_enhance_num = 500
rand_range = 0.07


def data_preprocess(data_seqs):

    train_data = []
    for data_seq in data_seqs:
        if len(data_seq) == 0:
            continue
        if len(data_seq) < lookback + 1:
            data_seq = [data_seq[0]]*(lookback + 1 - len(data_seq)) + data_seq
        for i in range(0, len(data_seq)-lookback):
            one_data = data_seq[i:i + lookback + 1]
            one_data = np.array(one_data)
            for _ in range(data_enhance_num):
                temp = one_data * \
                    np.random.normal(1, rand_range, lookback + 1)
                train_data.append(temp)
    train_data = np.array(train_data)
    # np.random.shuffle(train_data)
    return train_data


data_seq = []

for i in range(3):
    data_seq.append(rand_gen_data())

data_set = data_preprocess(data_seq)

num, step = data_set.shape
x_train = data_set[:, :4].reshape(num, step-1, 1)
y_train = data_set[:, 4]


print(data_set)

model = Sequential()
model.add(layers.LSTM(4,
                      dropout=0.1,
                      recurrent_dropout=0.5,
                      return_sequences=False,
                      input_shape=(4, 1)))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mse')

history = model.fit(x_train, y_train,
                    batch_size=16, epochs=5, validation_split=0.2, shuffle=True)

test = [rand_gen_data()]
test_data = data_preprocess(test)
num, step = test_data.shape
y_test = test_data[:, 4]
x_test = test_data[:,:4].reshape(num, step-1, 1)

y_pred = model.predict(x_test)

days = range(1,len(y_pred)+1)

plt.plot(days,y_pred,c='r')
plt.plot(days,y_test,c='b')
plt.show()

# model.save('/root/MySource/LSTM0.h5')
