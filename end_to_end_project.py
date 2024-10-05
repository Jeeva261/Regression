import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder,OneHotEncoder,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

# collection and load data
file_path=r"C:\Users\ADMIN\OneDrive\Desktop\AIML githup\Regression\end_to_end_project.csv"
df=pd.read_csv(file_path)
df.drop(columns=["date","street","city","statezip","country"],inplace=True)
print(df)

# understanding the data
print(df.head())
print(df.tail())
print(df.describe())
print(df.nunique())
print(df.isnull().sum())
print(df.columns)
print(df.duplicated().sum())
print(df.dtypes)
print(df.info())
print(df.corr())
print(df.shape)

# Handle Missing Values
# Check for missing values
print(df.isnull().sum())

# Fill missing values with mean or media


# Data Types Conversion
# Convert 'date' column to datetime


# One-hot encoding for categorical variables



# Exploratory Data Analysis (EDA)
# Visualizations

# Histogram for price 
sns.histplot(df["price"],bins=30)
plt.title("Distribuation of House price")
plt.xlabel("Price")
plt.ylabel("frequancy")
plt.show()

# scatterplot for sqft_living vs price
sns.scatterplot(x="sqft_living",y="price",data=df)
plt.title("Price vs. Square Footage of Living Space")
plt.xlabel("price")
plt.ylabel("Square footage")
plt.show()

# Correlation Analysis

# Correlation matrix

correlation_matrix=df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix,cmap="viridis",annot=True,linecolor="white",linewidths=0.2)
plt.title('Correlation Heatmap')
plt.show()

#  Feature Engineering
# Selecting Features

x=df.drop(columns="price")
y=df["price"]

# Data Splitting

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# Model Building

# Choosing a Model

model=LinearRegression()
model.fit(x_train,y_train)

# Predictions

y_pridect=model.predict(x_test)
print(y_pridect)

#  Model Evaluation

# metrics
r2=r2_score(y_test,y_pridect)
mse=mean_squared_error(y_test,y_pridect)

print(f"r sqaure:{r2}")
print(f"mean_square_error:{mse}")

# Visualization

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pridect)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color="red")
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)


