import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
price_by_person = pd.read_csv('./base.csv', sep=';')
x_people = price_by_person.iloc[:, 0].values
y_prices = price_by_person.iloc[:, 1].values
regression_model = LinearRegression()
x_people_as_matrix = x_people.reshape(-1, 1)
regression_model.fit(x_people_as_matrix, y_prices)
prediction = regression_model.predict(x_people_as_matrix)
print('corelation:', np.corrcoef(x_people, y_prices))
print('intersection:', regression_model.intercept_)
print('coefficient:', regression_model.coef_[0])
print('to 57 people:', regression_model.predict([[57]])[0])
print('to 20 people:', regression_model.predict([[20]])[0])
print('to 25 people:', regression_model.predict([[25]])[0] - 1500)
print('score:', regression_model.score(x_people_as_matrix, y_prices))
print('mean absolute error:', mean_absolute_error(y_prices, prediction))
print('mean squared error:', mean_squared_error(y_prices, prediction))
print('root mean squared error:', np.sqrt(mean_squared_error(y_prices, prediction)))