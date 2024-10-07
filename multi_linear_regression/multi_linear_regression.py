import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import LabelEncoder

# loaded dataset
file_path=r'C:\Users\ADMIN\OneDrive\Desktop\AIML githup\Regression\multi_exp.csv'
df=pd.read_csv(file_path)
print(df.head())
encoding=LabelEncoder()
df["Education_Level_New"]=encoding.fit_transform(df["Education_Level"])
res=df.drop(columns="Education_Level",axis=1)
print(res)

x=res[["Experience","Education_Level_New","Age"]]
y=df["Salary"]

# testing and traing
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# multi linear regression
model=LinearRegression()
dm=model.fit(x_train,y_train)


# predicted
y_pred=model.predict(x_test)
print("predicted salaries :",y_pred)

slope=model.coef_
print(f"slope:{slope}")

intercept=model.intercept_
print(f'intercept{intercept}')

r2=r2_score(y_test,y_pred)
print(f'r square:{r2}')

mse=mean_squared_error(y_test,y_pred)
print(f'mean square error:{mse}')



# new_predition
new_pred=[30,0,70]

new_pred_id=model.predict([new_pred])
print(new_pred_id[0])





# visulazation
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)  
plt.title("Actual vs Predicted Salaries")
plt.xlabel("Actual Salaries")
plt.ylabel("Predicted Salaries")
plt.grid()
plt.show() 