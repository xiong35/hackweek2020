
import pandas as pd
import json

filename = R'.\population.csv'

population = pd.read_csv(filename,header=0)
population = pd.DataFrame(list(population['population']),index=population['Country Name'])


with open(R'.\world_data.json') as f:
    lines = f.readlines()

json_str = ''
for line in lines:
    json_str += line

world_data_dict = json.loads(json_str)

for country, info_dic in world_data_dict.items():
    try:
        a= df.loc[country]
    except KeyError:
        print('['+country+']',' not found')
