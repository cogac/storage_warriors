
import pandas as pd


df = pd.read_csv('./wanted_data.csv')

df.drop(['barcode', 'weight'], axis=1, inplace=True)

df = df.groupby(['order', 'product_id', 'date', 'year', 'costum'], as_index=False).sum()

df.to_csv('test_wanted_data.csv', index=True)
