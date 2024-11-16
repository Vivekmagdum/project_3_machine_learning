import pandas as pd
import numpy as np

#read the csv file into a pandas dataframe
data=pd.read_csv("data.csv")

#split the data into train and test set with a test size of 20% and random state of 42 to ensure reproducibility of the results
from sklearn.model_selection import train_test_split
train_set, test_set  = train_test_split(data, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")

#split the data into features and target variable
train_set_x=train_set.drop(columns=["T2m"])
train_set_y=train_set["T2m"]

#split the data into features and target variable for the test set
test_set_x=train_set.drop(columns=["T2m"])
test_set_y=train_set["T2m"]

#Add this model which more accurate
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()

# use RandomForestRegressor with n_estimators=100 and random_state=42 to train the model and predict the target variable
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_set_x, train_set_y)
y_pred=model.predict(test_set_x)

# calculate the root mean squared error (RMSE) to evaluate the model's performance
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test_set_y, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

n=model.predict([[202.59,179.50,51.01,201.88,0.25]]) # get the feture if you are manauually want to get feature