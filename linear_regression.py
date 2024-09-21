import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error


x=np.array([20,25,30,35,40]).reshape(5,1)
y=np.array([13,21,25,35,38])

model=LinearRegression()
model=model.fit(x,y)
print(model)
y_pred=model.predict(x)
print(y_pred)
print(f'slope:{model.coef_}')
print(f'intercept:{model.intercept_}')

r2=r2_score(y,y_pred)
print(f'r_squared:{r2}')
mse=mean_squared_error(y,y_pred)
print(f'mean_square_error:{mse}')


plt.scatter(x,y,color="blue",label="Actual data",marker="o")
plt.plot(x,y_pred,color="red",label="regression line")
plt.title("temperature vs icecream_sales")
plt.xlabel("Temperature")
plt.ylabel("icecream_sales")
plt.legend()
plt.show()


