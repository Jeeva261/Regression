import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

x=np.array([1,2,3,4,5,6,7,8,9]).reshape(9,1)
y=np.array([1, 4, 9, 16, 26, 36, 49, 64, 81])

# model=LinearRegression()
# res=model.fit(x,y)
# print(res)

# y_pred=model.predict(x)
# print(y_pred)


# plt.scatter(x,y,color="red")
# plt.plot(x,y_pred,color="blue")
# plt.show()

poly=PolynomialFeatures(degree=2)
x_poly=poly.fit_transform(x)


model=LinearRegression()
res=model.fit(x_poly,y)

y_pred=model.predict(x_poly)
print(y_pred)

plt.scatter(x, y, color='blue', label='Original Data')
plt.plot(x, y_pred, color='red', label='Polynomial Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

mse=mean_squared_error(y,y_pred)
print(f"mean square error :{mse}")

