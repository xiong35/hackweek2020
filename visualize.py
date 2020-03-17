
from pyecharts import Map
import json


with open(R'.\world_confirmed_data.json') as f:
    lines = f.readlines()

json_str = ''
for line in lines:
    json_str += line

world_data_dict = json.loads(json_str)
countries = []
confirmed = []
for country, data_list in world_data_dict.items():
    if data_list:
        countries.append(country)
        confirmed.append(data_list[-1])


map = Map("covid-19疫情地图", width=1000, height=500)
map.add(
    "",
    countries,
    confirmed,
    maptype="world",
    is_visualmap=True,
    is_map_symbol_show=False,
    is_piecewise=True,
    visual_text_color="#000",
    pieces=[
            {"max": 10, "min": 0, "label": "0~10"},
            {"max": 100, "min": 10, "label": "10~100"},
            {"max": 1000, "min": 100, "label": "100~1000"},
            {"max": 10000, "min": 1000, "label": "1000~10000"},
            {"max": 99999999, "min": 10000, "label": "10000+"},
    ]
)
map.render('covid_map.html')
