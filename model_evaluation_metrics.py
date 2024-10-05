import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import r2_score,mean_squared_error

data={
    "X1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Y": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
}
df=pd.DataFrame(data)
x=df[["X1"]]
y=df["Y"]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


model=LinearRegression()
model.fit(x_train,y_train)

model_predict= model.predict(x_test)


# r2_score
r_square=r2_score(y_test,model_predict)
print(f"r-square:{r_square}")


# adjusted_r2_score
n = x_test.shape[0]  # Number of observations
p = x_test.shape[1]  # Number of predictors
adjusted_r_squared = 1 - ((1 - r_square) * (n - 1) / (n - p ))#- 1))

print(f"adjusted_r_squared:{adjusted_r_squared}")

# mean_square_error
mse=mean_squared_error(y_test,model_predict)
print(f"mean square error:{mse}")

# Root_mean_square_error
rmse=np.sqrt(mse)
print(f"Root mean square error:{rmse}")



