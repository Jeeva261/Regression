import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

# 1.Implement polynomial regression and compare it with linear regression

x = np.linspace(0, 10, 100).reshape(-1, 1) 
y = 2 * x**2 + 3 * x + 4 + np.random.randn(100, 1) * 10 



# linear regression
model=LinearRegression()
res=model.fit(x,y)

y_pred=model.predict(x)
print(y_pred)


# polynomial regression

poly=PolynomialFeatures(degree=3)
x_poly=poly.fit_transform(x)

model_poly=LinearRegression()
res=model_poly.fit(x_poly,y)

y_pred_poly=model_poly.predict(x_poly)


l_mse=mean_squared_error(y,y_pred)
print(f"linear mean square error:{l_mse}")

p_mse=mean_squared_error(y,y_pred_poly)
print(f"poly mean square error:{p_mse}")


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.scatter(x,y,color="red",label="data point")
plt.plot(x,y_pred,color="blue",label="linear regression")
plt.title("linear regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()


plt.subplot(1,2,2)
plt.scatter(x,y,color="red",label="data point")
plt.plot(x,y_pred_poly,color="blue",label="polynomial regression")
plt.title(" polynamial regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


# 2.Visualize the fit of polynomial regression on the data.


x=np.array([1,2,3,4,5,6,7,8,9,10]).reshape(10,1)
y=np.array([1, 2, 1.5, 3.5, 3, 5, 4.5, 6, 7, 8])

poly=PolynomialFeatures(degree=3)
x_poly=poly.fit_transform(x)

model=LinearRegression()
res=model.fit(x_poly,y)

X_fit = np.linspace(0, 10, 100).reshape(-1, 1)
y_fit = model.predict(poly.transform(X_fit))


plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(X_fit, y_fit, color='red', label='Polynomial Regression Fit')
plt.title('Polynomial Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()