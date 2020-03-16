
import pandas as pd

filename = R'.\population.csv'

df = pd.read_csv(filename,header=0)
df = pd.DataFrame(list(df['population']),index=df['Country Name'])
print(int(df.loc['China']))


