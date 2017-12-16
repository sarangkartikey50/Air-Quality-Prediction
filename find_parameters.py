import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sm as sm
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import numpy as np
import math
import itertools

color = sns.color_palette()
print('Please wait. Importing data...')
df = pd.read_csv("data.csv", encoding = "ISO-8859-1")
print('import completed.')

def date_parser(x):
    return datetime.strptime(x, '%Y-%m')


delhi_data = df[df.state == 'Delhi'].sort_values(by='date', ascending=0)
delhi_data_no2 = delhi_data[['date', 'no2']]
delhi_data_no2['no2'] = delhi_data_no2['no2'].map(lambda x: str(x))
delhi_data_no2 = delhi_data_no2[delhi_data_no2.no2 != 'nan']
delhi_data_no2['no2'] = pd.to_numeric(delhi_data_no2['no2'])
delhi_data_no2['date'] = delhi_data_no2['date'].map(lambda x: str(x)[:7])
delhi_data_no2['date'] = delhi_data_no2['date'].map(lambda x: date_parser(x))
delhi_data_no2.index = delhi_data_no2['date']


delhi_data_no2 = delhi_data_no2.fillna(delhi_data_no2.bfill())
delhi_data_no2 = delhi_data_no2['no2'].resample('MS').mean()


# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

warnings.filterwarnings("ignore") # specify to ignore warning messages

df_aic = pd.DataFrame(columns=['aic', 'param', 'seasonal_param'])
i = 0
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(delhi_data_no2,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            df_aic.loc[i] = [results.aic, param, param_seasonal]
            i+=1
        except:
            continue

df_aic = df_aic.sort_values(by='aic', ascending=1)
print(df_aic)