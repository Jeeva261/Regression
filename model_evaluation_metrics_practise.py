# 1.Calculate and interpret different evaluation metrics for regression models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

data={
    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'y': [2.3, 2.9, 3.7, 4.1, 5.4, 5.9, 7.1, 7.9, 8.8,10.1]
}
df=pd.DataFrame(data)
x=df[["x"]]
y=df["y"]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


model=LinearRegression()
model.fit(x_train,y_train)

model_predict= model.predict(x_test)


# r2_score
r_square=r2_score(y_test,model_predict)
print(f"r-square:{r_square}")


# mean_square_error
mse=mean_squared_error(y_test,model_predict)
print(f"mean square error:{mse}")

# Root_mean_square_error
rmse=np.sqrt(mse)
print(f"Root mean square error:{rmse}")


# mean_absolute_error
mae=mean_absolute_error(y_test,model_predict)
print(f'mean absolute error:{mae}')

print("\nInterpretation:")
if r_square < 0:
    print("The model is performing worse than a horizontal line (no predictive power).")
elif r_square <= 0.5:
    print("The model explains a small portion of the variance (weak predictive power).")
elif r_square <= 0.8:
    print("The model explains a moderate portion of the variance (reasonable predictive power).")
else:
    print("The model explains a large portion of the variance (strong predictive power).")

