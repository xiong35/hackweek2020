
# 读取fq得到的数据

import pandas as pd
import json

filename = R'C:\Users\xiong35\Desktop\total_cases.csv'

df = pd.read_csv(filename, header=None)

world_data = dict()

for i in range(2, df.shape[1]):
    country = df[i][0]
    data = list(df[i].dropna()[1:].astype(int))
    world_data[country] = data


with open('./world_confirmed_data.json', 'w') as f:
    f.write(json.dumps(world_data, ensure_ascii=False, indent=4))
