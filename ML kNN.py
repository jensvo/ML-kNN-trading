
#This code runs a k-Nearest neighbours between two time series and uses the parameters to predict future values.
#Disclosure: Nothing in this repository should be considered investment advice. Past performance is not necessarily indicative of future returns.
# These are general examples about how to import data using pandas for a small sample of financial data across different time intervals.
#Please note that the below code is written with the objective to explore various python features.

from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import quandl
import pandas as pd
from statsmodels import regression
import statsmodels.api as sm
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#Download data from quandl
quandl.ApiConfig.api_key = "yourquandlapikey"
data = quandl.get(["NASDAQOMX/NDX.1","NASDAQOMX/NQUSM.1"], start_date="2015-12-04", end_date=date.today())

#Define length of train and validation data
traindata = int(0.8*(data.index.size))
validdata = int(0.2*(data.index.size))

#split into train and validation data
train = data.iloc[:traindata]
valid = data.iloc[traindata:]

x_train = train.drop('NASDAQOMX/NDX - Index Value', axis=1)
y_train = train['NASDAQOMX/NDX - Index Value']
x_valid = valid.drop('NASDAQOMX/NDX - Index Value', axis=1)
y_valid = valid['NASDAQOMX/NDX - Index Value']


#scaling data
x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_valid_scaled = scaler.fit_transform(x_valid)
x_valid = pd.DataFrame(x_valid_scaled)

#using gridsearch to find the best parameter
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

#fit the model and make predictions
model.fit(x_train,y_train)
predictions = model.predict(x_valid)

#Add predicitions to dataframe
#valid = valid['Predictions']
valid['Predictions'] = predictions

#Plot the result
fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
ax.xaxis.set_tick_params(labelsize=7, labelrotation = 45)
ax.plot(train['NASDAQOMX/NDX - Index Value'], label=train['NASDAQOMX/NDX - Index Value'].name)
ax.plot(valid['NASDAQOMX/NDX - Index Value'], label=valid['NASDAQOMX/NDX - Index Value'].name)
ax.plot(valid['Predictions'], label=valid['Predictions'].name)
ax.legend(loc='upper left', frameon=False)

plt.show()