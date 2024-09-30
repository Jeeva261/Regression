# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Ridge,Lasso
# from sklearn.metrics import mean_squared_error


# x=np.random.rand(100,10)
# y=np.random.rand(100)

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# ridge_model=Ridge(alpha=1.0)
# ridge_model.fit(x_train,y_train)
# ridge_predict=ridge_model.predict(x_test)
# ridge_mse=mean_squared_error(y_test,ridge_predict)
# print(f'ridge_mse:{ridge_mse:.4f}')


# Lasso_model=Lasso(alpha=0.1)
# Lasso_model.fit(x_train,y_train)
# Lasso_predict=Lasso_model.predict(x_test)
# Lasso_mse=mean_squared_error(y_test,Lasso_predict)
# print(f"Lasso_mse:{Lasso_mse:.4f}")


