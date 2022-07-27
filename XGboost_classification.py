# import the necessary packages
from sklearn import metrics
import xgboost as xgb
import sklearn.datasets as dataset
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import matplotlib.pyplot as plt
# import the dataset
data = dataset.load_wine()
# take the data and the labels
X, y = data.data, data.target
# split the data and target
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.1)
# set the XGboost classifier
xgbc = xgb.XGBClassifier(objective= 'binary:logistic', alpha = 5, learning_rate = 0.1, n_estimator = 10)
# fit the model with the training data
xgbc.fit(X_train, y_train)
# predict the output
predict = xgbc.predict(X_test)
# see the classification report and matrix confusion
print(metrics.classification_report(y_test, predict))
print(metrics.confusion_matrix(y_test, predict))
