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
print(df.head())

x=df[["YearsExperience"]]
y=df["Salary"]
print(x,y)

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(X_test,Y_test)
print(X_train,Y_train)

model=LinearRegression()
res=model.fit(X_train,Y_train)
print(res)

y_pred=model.predict(X_test)
print(y_pred)

slope=model.coef_
print(f'slope:{slope}')

intercept=model.intercept_
print(f'intercept:{intercept}')

r2=r2_score(Y_test,y_pred)
print(f'r square:{r2}')

mse=mean_squared_error(Y_test,y_pred)
print(f'mean square error:{mse}')

plt.scatter(X_train,Y_train,color="blue",label="Traind data")
plt.plot(X_train,model.predict(X_train),color="red",label="regression line")
plt.title("YearsExperience vs Salary")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.legend()
plt.show()

yearsexperience=5.5
new_predict=model.predict([[yearsexperience]])
print(new_predict[0])