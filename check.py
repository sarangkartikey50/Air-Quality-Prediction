no2 = delhi_data_no2.values
no2 = [x for x in no2 if not math.isnan(x)]

size = int(len(no2) * 0.66)
train, test = no2[0:size], no2[size:len(no2)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(2 ,2 ,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat[0])
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

pred = np.array(predictions)

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)



# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
