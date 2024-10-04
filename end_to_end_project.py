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
df.drop(columns=["date","sqft_above","sqft_basement","yr_built","yr_renovated","street","city","statezip","country"],axis=1,inplace=True)

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


# heatmap for describe

plt.figure(figsize=(10,10))
sns.heatmap(df.describe(),cmap="viridis",linecolor="white",linewidths=0.2,annot=True)
plt.title('Describe of House data')
plt.show()


# heatmap for corr

plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),cmap="viridis",linecolor="white",linewidths=0.2,annot=True)
plt.show()

for col in df.columns:
    sns.boxplot(df,y=col)
    plt.title(f"Distribution of {col}in the dataset")
    plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(df)
plt.show()

