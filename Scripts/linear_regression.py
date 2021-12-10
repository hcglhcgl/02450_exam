"""
Compute prediction of linear model

"""
import numpy as np
from sklearn.linear_model import LinearRegression

x_data = [1, 3, 4]
y_data = [2, 5, 6]
predict_at = 5

x = np.array(x_data).reshape(-1, 1)
model = LinearRegression()
model.fit(x, y_data)
new_x_data = sorted(x_data + [predict_at])
new_x = np.array(new_x_data).reshape(-1, 1)
prediction = model.predict(new_x)[new_x_data.index(predict_at)]
print(prediction)
