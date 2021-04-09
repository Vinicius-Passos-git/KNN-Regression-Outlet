#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import data
data = pd.read_csv('train_cleaned.csv')
print(data.head(5))

#Segregating variables: Independent and Dependent Variables
x = data.drop(['Item_Outlet_Sales'], axis=1)
y = data['Item_Outlet_Sales']
print(x.shape, y.shape)

#Scaling the data (Using MinMax Scaler)
#Importing MinMax Scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
x = pd.DataFrame(x_scaled)

#Importing Train test split
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 56)

#Implementing KNN Regressor
#importing KNN regressor and metric mse
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.metrics import mean_squared_error as mse

#Creating instance of KNN
reg = KNN(n_neighbors = 5)
#Fitting the model
reg.fit(train_x, train_y)
#Predicting over the Train Set and calculating MSE
test_predict = reg.predict(test_x)
k = mse(test_predict, test_y)
print('Test MSE k = 5   ', k )

#Elbow for Classifier
def Elbow(K):
    #initiating empty list\n",
    test_mse = []
    #training model for evey value of K\n",
    for i in K:
        #Instance of KNN\n",
        reg = KNN(n_neighbors = i)
        reg.fit(train_x, train_y)
        #Appending mse value to empty list claculated using the predictions
        tmp = reg.predict(test_x)
        tmp = mse(tmp,test_y)
        test_mse.append(tmp)

    return test_mse

#Defining K range
k = range(1,40)

#calling above defined function
test = Elbow(k)

#plotting the Curves
plt.plot(k, test)
plt.xlabel('K Neighbors')
plt.ylabel('Test Mean Squared Error')
plt.title('Elbow Curve for test')
plt.show()

#Creating instance of KNN\
reg = KNN(n_neighbors = 9)
#Fitting the model
reg.fit(train_x, train_y)
#Predicting over the Train Set and calculating F1
test_predict = reg.predict(test_x)
k = mse(test_predict, test_y)
print('Test MSE  k = 9  ', k )

print('Code successfully executed!')
