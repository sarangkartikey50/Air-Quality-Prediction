import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import datetime
import statsmodels.api as sm

plt.style.use('fivethirtyeight')

color = sns.color_palette()
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
delhi_data_no2.index = delhi_data_no2['date']


delhi_data_no2 = delhi_data_no2.fillna(delhi_data_no2.bfill())
delhi_data_no2 = delhi_data_no2['no2'].resample('MS').mean()

print(delhi_data_no2)

delhi_data_no2.plot(figsize=[15, 8])
plt.xlabel("DATE")
plt.ylabel("NO2 CONCENTRATION")
plt.title("NO2 CONCENTRATION OF DELHI (1987 - 2015)")
plt.show()

mod = sm.tsa.statespace.SARIMAX(delhi_data_no2,
                                order=(1, 1, 1),
                                seasonal_order=(1, 0, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 12))
plt.show()

pred = results.get_prediction(start=pd.to_datetime('1995-01-01'), dynamic=False)
pred_ci = pred.conf_int()

ax = delhi_data_no2['1990':].plot(figsize=[15, 8], label='observed')
pred.predicted_mean.plot(figsize=[15, 8], ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('NO2 Levels')
plt.legend()

delhi_data_no2_forecasted = pred.predicted_mean
delhi_data_no2_truth = delhi_data_no2['1998-01-01':]

# Compute the mean square error
rmse = (((delhi_data_no2_forecasted - delhi_data_no2_truth) ** 2).mean()) ** 0.5
print('The Root Mean Squared Error of our prediction is {}'.format(round(rmse, 2)))

forecast = results.forecast(30)
print(forecast)
forecast.plot(figsize=[15, 8], color='green', label='future predictions')
plt.legend()
plt.show()