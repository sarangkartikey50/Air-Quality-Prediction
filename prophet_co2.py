import pandas as pd
from fbprophet import Prophet
from pandas import datetime
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

plt.style.use('fivethirtyeight')

print('Please wait. Importing data...')
data = sm.datasets.co2.load_pandas()
co2 = data.data
print('import completed.')
co2['ds'] = co2.index
co2.rename(columns={'co2': 'y'}, inplace=True)

co2.dropna(inplace=True)

co2 = co2['1987-01-01':]

old_dates = pd.DataFrame(co2['ds'])
co2.tail()


#exit(0)

ax = co2.set_index('ds').plot(figsize=(12, 8))
ax.set_ylabel('CO2 EMISSION')
ax.set_xlabel('YEAR')
plt.show()

my_model = Prophet()
my_model.fit(co2)

future_dates = my_model.make_future_dataframe(periods=365*20, include_history=False)


forecast = my_model.predict(future_dates)
old_data_forcast = my_model.predict(old_dates)

rmse = mean_squared_error(co2.y, old_data_forcast.yhat) ** 0.5

print("mean squared error - ", round(rmse, 2))
#print(r2_score(co2.y, old_data_forcast.yhat))

my_model.plot(forecast, uncertainty=True)
plt.show()
my_model.plot_components(forecast)
plt.show()