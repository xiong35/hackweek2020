
import json
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import datetime as dt
import matplotlib.dates as mdates

filename = R'.\world_confirmed_data.json'
font = fm.FontProperties(fname='.\SIMFANG.TTF')
model_name = R'.\LSTM.h5'

lookback = 4
data_enhance_num = 10
rand_range = 0.07
batch_size = 32
epochs = 7
ONEDAY = dt.timedelta(days=1)
tomorrow = dt.date.today()+ONEDAY


class Predict:
    def __init__(self,):
        pass

    def _plot_curve(self, country, data_list):
        if len(data_list) > 10:
            data_list = data_list[-10:]
        dates = [tomorrow]
        for i in range(1, len(data_list)):
            dates.insert(0, tomorrow-i*ONEDAY)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.plot(dates[:-1], data_list[:-1],
                 c='dodgerblue', alpha=1, label='近日感染人数')
        plt.plot(dates[-2:], data_list[-2:],
                 c='cornflowerblue', alpha=1, linestyle='--', label='预计明日人数')
        plt.title(country+"'s confirmed cases")
        plt.legend(prop=font)
        plt.gcf().autofmt_xdate()
        plt.savefig(R'.\ '[:-1]+country+'.png')

    def daily_update(self, filename=filename, model_name=model_name):
        self._load_model(model_name)
        world_data_dict = self._load_data(filename)
        new_data = []
        pred_data = dict()
        for country, num_list in world_data_dict.items():
            if len(num_list) < lookback + 1:
                num_list = [0] * (lookback + 1 - len(num_list)) + num_list

            pred_num = self._predict(num_list)

            pred_data[country] = pred_num

            if num_list[-1] > 1000:
                self._plot_curve(country, num_list+[pred_num])

            new_data.append(num_list[-lookback-1:])

    def _load_data(self, filename):
        with open(filename) as fr:
            lines = fr.readlines()

        json_str = ''
        for line in lines:
            json_str += line

        world_data_dict = json.loads(json_str)

        return world_data_dict

    def _data_preprocess(self, data_list, data_enhance_num=data_enhance_num):
        processed_data = []
        cur_enhance = int(data_enhance_num*np.log(data_list[-1]+10))
        for i in range(0, len(data_list)-lookback):
            one_data = data_list[i:i + lookback + 1]
            one_data = np.array(one_data)
            for _ in range(cur_enhance):
                rand = np.random.uniform(
                    1-rand_range, 1+rand_range, lookback + 1)
                temp = one_data * rand
                processed_data.append(temp)
        return processed_data

    def _load_model(self, filename):
        self.model = load_model(filename)

    def _fine_tune(self, data_array):
        tune_x = data_array[:, :lookback].reshape(-1, lookback, 1)
        tune_y = data_array[:, lookback]
        model = self.model
        model.fit(tune_x, tune_y, batch_size=batch_size,
                  epochs=epochs, shuffle=True)
        return model

    def _predict(self, data_list):

        data_array = np.array(self._data_preprocess(data_list))

        model = self._fine_tune(data_array)

        pred_x = np.array(data_list[-lookback:])
        pred_num = model.predict(pred_x.reshape(-1,lookback, 1))

        return pred_num


p = Predict()
p.daily_update()
