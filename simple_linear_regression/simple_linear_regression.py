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
print(f"Coefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

r2=r2_score(y,y_pred)
print(f'r2 squared:{r2}')
mse=mean_squared_error(y,y_pred)
print(f'mean_square_error:{mse}')


plt.scatter(x,y,color="blue",label="Actual data",marker="o")
plt.plot(x,y_pred,color="red",label="regression line")
plt.title("temperature vs icecream_sales")
plt.xlabel("Temperature")
plt.ylabel("icecream_sales")
plt.legend()
plt.show()



import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

file_path=r'C:\Users\ADMIN\OneDrive\Desktop\AIML githup\Regression\simple_exp.csv'
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

print(f"Coefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

r2=r2_score(Y_test,y_pred)
print(f'r2 score:{r2}')

mse = mean_squared_error(Y_test, y_pred)
print(f"Mean Squared Error: {mse}")

years_exp = 5.5
predicted_salary = model.predict([[years_exp]])
print(f"Predicted Salary for {years_exp} years of experience: {predicted_salary[0]}")


plt.scatter(X_train, Y_train, color='blue')
plt.plot(X_train, model.predict(X_train), color='red')
plt.title('YearsExperience vs Salary (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



