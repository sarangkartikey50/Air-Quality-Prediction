import pandas as pd
from fbprophet import Prophet
from pandas import datetime
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

plt.style.use('fivethirtyeight')

print('Please wait. Importing data...')
df = pd.read_csv("data.csv", encoding = "ISO-8859-1")
print('import completed.')

def date_parser(x):
    return datetime.strptime(x, '%Y-%m-%d')


delhi_data = df[df.state == 'Delhi'].sort_values(by='date', ascending=0)
delhi_data_no2 = delhi_data[['date', 'no2']]
delhi_data_no2['no2'] = delhi_data_no2['no2'].map(lambda x: str(x))
delhi_data_no2 = delhi_data_no2[delhi_data_no2.no2 != 'nan']
delhi_data_no2['no2'] = pd.to_numeric(delhi_data_no2['no2'])
delhi_data_no2['date'] = delhi_data_no2['date'].map(lambda x: str(x)[:10])
delhi_data_no2['date'] = delhi_data_no2['date'].map(lambda x: date_parser(x))



delhi_data_no2.rename(columns={'date': 'ds', 'no2': 'y'}, inplace=True)


delhi_data_no2.dropna(inplace=True)


old_dates = pd.DataFrame(delhi_data_no2['ds'])

print(old_dates.tail())


#exit(0)

ax = delhi_data_no2.set_index('ds').plot(figsize=(15, 8))
ax.set_ylabel('NO2 CONCENTRATION')
ax.set_xlabel('YEAR')
plt.title("NO2 CONCENTRATION OF DELHI (1987 - 2015)")
plt.show()

my_model = Prophet()
my_model.fit(delhi_data_no2)

future_dates = my_model.make_future_dataframe(periods=365*3, include_history=False)


forecast = my_model.predict(future_dates)
old_data_forcast = my_model.predict(old_dates)

rmse = mean_squared_error(delhi_data_no2.y, old_data_forcast.yhat) ** 0.5

print("mean squared error - ", round(rmse, 2))
#print(r2_score(delhi_data_no2.y, old_data_forcast.yhat))

my_model.plot(forecast, uncertainty=True)
plt.show()
my_model.plot_components(forecast)
plt.show()