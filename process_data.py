
# 读取文件夹里的每日报告

import pandas as pd
import os
import numpy as np
import json

data_dir = R'C:\Users\xiong35\Desktop\csse_covid_19_daily_reports'

daily_datas = os.listdir(data_dir)

world_data = dict()

for daily_csv in daily_datas:
    csv_path = data_dir+R'\ '[:-1]+daily_csv
    df = pd.read_csv(csv_path, header=0)
    df = df.fillna(0)
    today_data = dict()
    for i in range(df.shape[0]):
        country = df.loc[i, 'Country/Region']
        if country in ['Macau', 'Taiwan', 'Mainland China']:
            country = 'China'
        if country == ' Azerbaijan':
            country = 'Azerbaijan'
        confirmed = df.loc[i, 'Confirmed']
        deaths = df.loc[i, 'Deaths']
        recovered = df.loc[i, 'Recovered']
        if country not in today_data:
            today_data[country] = {'confirmed': 0,
                                   'deaths': 0, 'recovered': 0}

        today_data[country]['confirmed'] += float(confirmed)
        today_data[country]['deaths'] += float(deaths)
        today_data[country]['recovered'] += float(recovered)

    for country, info_dic in today_data.items():
        if country not in world_data:
            world_data[country] = {'confirmed': [],
                                   'deaths': [], 'recovered': []}
        world_data[country]['confirmed'].append(info_dic['confirmed'])
        world_data[country]['deaths'].append(info_dic['deaths'])
        world_data[country]['recovered'].append(info_dic['recovered'])


with open('./world_data.json', 'w') as f:
    f.write(json.dumps(world_data, ensure_ascii=False, indent=4))
