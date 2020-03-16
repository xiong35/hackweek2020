
from pyecharts import options as opts
from pyecharts.charts import Geo,
from pyecharts import Map
from pyecharts.globals import ChartType, SymbolType
import echarts_countries_pypkg
import json

# pip install pyecharts==0.5.10

value = [95.1, 23.2, 43.3, 66.4, 88.5]
attr = ["China", "Canada", "Brazil", "Russia", "United States"]
map0 = Map("世界地图示例")
map0.add("世界地图", attr, value, maptype="world", is_visualmap=True, visual_text_color='#000')
map0.render(path="世界地图.html")

# with open(R'.\world_data.json') as f:
#     lines = f.readlines()

# json_str = ''
# for line in lines:
#     json_str += line

# world_data_dict = json.loads(json_str)
# countries = []
# confirmed = []
# for country, info_dic in world_data_dict.items():
#     countries.append(country)
#     confirmed.append(info_dic['confirmed'][-1])
# zipped = zip(countries, confirmed)
# input_data = [z for z in zipped]
# print(input_data)
# geo = Geo()
# geo.add_schema(maptype='world')
# geo.add("", input_data)

# geo.render()
