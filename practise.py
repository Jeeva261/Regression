import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

# 1.Implement linear regression on a dataset and interpret the coefficients
# 2.Evaluate the model using R^2 and MSE

file_path=r'C:\Users\ADMIN\OneDrive\Desktop\AIML githup\Regression\exp.csv'
df=pd.read_csv(file_path)
print(df)

x=df[["Salary"]]
y=df["YearsExperience"]
print(x)
print(y)


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2,random_state=0)
print(X_test)
print(X_train)
print(Y_test)
print(Y_train)

model=LinearRegression()
res=model.fit(X_train,Y_train)
print(res)

y_pred=model.predict(X_test)
print(y_pred)

print(f'slope:{model.coef_[0]}')
print(f'intercept:{model.intercept_}')


r2=r2_score(Y_test,y_pred)
print(f'r2:{r2}')

mse=mean_squared_error(Y_test,y_pred)
print(f'mse:{mse}')

YearsExperience=5.5
predicted_salary=model.predict([[YearsExperience]])
print(predicted_salary[0])


plt.scatter(X_train, Y_train, color='blue')
plt.plot(X_train, model.predict(X_train), color='red')
plt.title('YearsExperience vs Salary (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
