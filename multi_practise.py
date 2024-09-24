import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import LabelEncoder,OneHotEncoder


# 1.Implement multiple linear regression on a dataset with multiple features
# 2.Analyze and interpret the results


file_path=r'C:\Users\ADMIN\OneDrive\Desktop\AIML githup\Regression\multi_exp.csv'

df=pd.read_csv(file_path)
print(df)
encoding=LabelEncoder()
df["Education_Level_new"]=encoding.fit_transform(df["Education_Level"])
res=df.drop(columns="Education_Level",axis=1)
print(res)

x=res[["Experience","Education_Level_new","Age"]]
y=res["Salary"]


x_train, x_test, y_train, y_test=train_test_split(x,y)
print(x_test)
print(y_test)


model=LinearRegression()
res=model.fit(x_train,y_train)
print(res)


y_pred=model.predict(x_test)
print(y_pred)

slope=model.coef_
print(f'slope:{slope}')

intercepts=model.intercept_
print(f'intercept:{intercepts}')


r2=r2_score(y_test,y_pred)
print(f'r square:{r2}')

mse=mean_squared_error(y_test,y_pred)
print(f'mean square error:{mse}')



new_predicted=[18,3,50]
new_predicted_value=model.predict([new_predicted])
print(new_predicted_value)