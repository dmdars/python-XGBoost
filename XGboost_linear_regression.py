# import the necessary library
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split as tts
import pandas as panda
import xgboost as xg
import numpy as np
from matplotlib import pyplot as plt 
# input the dataset
boston = load_boston()
data = panda.DataFrame(boston.data)

# data.columns = boston.feature_names
# split the dataset to data point and label
x,y = data.iloc[:,:-1], data.iloc[:, -1]

# input the data and the label into xgboost
data_dmatrix = xg.DMatrix(data = x, label = y)

#split the data into train and test
train_x, test_x, train_y, test_y = tts(x, y, test_size= 0.2, random_state=0)

# call the the xgboost algorithm
xgreg = xg.XGBRegressor(objective='reg:linear',learning_rate = 0.1, colsamplebytree = 0.3, max_depth =  5, alpha = 10, n_estimator = 10)

# fit the model into the train data
xgreg.fit(train_x, train_y) 

# predict the data using the test data
predict = xgreg.predict(test_x)

# find the square root of the mean square error
rmse = np.sqrt(mse(test_y, predict))
# print the rmse
# print("RMSE: %f " %(rmse))

# plot the decision tree
# xg.plot_tree(xgreg,num_trees=0)
# plt.rcParams['figure.figsize'] = [50, 10]
# plt.show()




# TEST THE DATASET
# print(data["CRIM"])
# print(data.head())
# print(boston.key())
# print(boston.data.shape)
# print(boston.feature_names)
# print(boston.DESCR)