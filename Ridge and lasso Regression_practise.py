import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,Ridge,LinearRegression
from sklearn.metrics import mean_squared_error



# 1.Implement Ridge and Lasso regression on a dataset

from sklearn.datasets import load_diabetes


diabetes=load_diabetes()
x=diabetes.data
y=diabetes.target
print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# Ridge 
Ridge_reg=Ridge(alpha=1.0)
Ridge_reg.fit(x_train,y_train)

Ridge_model_pridect=Ridge_reg.predict(x_test)

Ridge_mse=mean_squared_error(y_test,Ridge_model_pridect)
print(f'Ridge mean square error:{Ridge_mse:.4f}')


# Lasso
Lasso_las=Lasso(alpha=0.1)
Lasso_las.fit(x_train,y_train)

Lasso_las_pridect=Lasso_las.predict(x_test)

Lasso_mse=mean_squared_error(y_test,Lasso_las_pridect)
print(f"Lasso mean square error:{Lasso_mse:.4f}")

# visulaization

plt.figure(figsize=(10,6))
plt.plot(Ridge_reg.coef_,label="Ridge Coefficients")
plt.plot(Lasso_las.coef_,label="Ridge Coefficients")
plt.title("Implement Ridge and Lasso regression")
plt.legend()
plt.show()


# 2.Compare the performance of models with and without regularization


from sklearn.datasets import load_iris

iris=load_iris()
x=iris.data
y=iris.target
print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# linear regression
lin_model=LinearRegression()
lin_model.fit(x_train,y_train)

model_predict=lin_model.predict(x_test)

linear_mse=mean_squared_error(y_test,model_predict)
print(f'linear regression mse(non regularization):{linear_mse:.4f}')


# Ridge

Ridge_reg=Ridge(alpha=1.0)
Ridge_reg.fit(x_train,y_train)

Ridge_reg_predict=Ridge_reg.predict(x_test)

Ridge_mse=mean_squared_error(y_test,Ridge_reg_predict)
print(f"Ridge regression mse(with regularization):{Ridge_mse:.4f}")


# Lasso

Lasso_reg=Lasso(alpha=0.1)
Lasso_reg.fit(x_train,y_train)

Lasso_reg_predict=Lasso_reg.predict(x_test)

Lasso_mse=mean_squared_error(y_test,Lasso_reg_predict)
print(f"Lasso regression mse;(with regularization):{Lasso_mse:.4f}")


# comparing the allmodels

print(f"comparing the all regression'Mse:")
print(f'linear regression (non regularization):MSE{linear_mse:.4f}')
print(f"Ridge regression (with regularization):MSE{Ridge_mse:.4f}")
print(f"Lasso regression (with regularization):MSE{Lasso_mse:.4f}")





# visulaization

plt.figure(figsize=(15,7))
plt.plot(lin_model.coef_,label="Linear regression (non regularization)",color="red")
plt.plot(Ridge_reg.coef_,label="Ridge regression (with regularization)",color="blue")
plt.plot(Lasso_reg.coef_,label="Lasso regression (with regularization)",color="green")
plt.legend()
plt.title(" models with and without regularization")
plt.show()
