# This project is for testing out Machine Learning tool in python for an education course.
# Anaconda Env tensorflow name of Env: tensor
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

#diffing dataset and seperators
data = pd.read_csv("student-mat.csv", sep=";")
#filtering dataset
data = data[["G1", "G2", "G3", "studytime"]]

#Which attribute to be predicted
predict = "G3"

#Drops the value in predict from dataset
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

#Splits dataset up into trainig data and test data, so it does not train on the same data as it tests on.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

#makes a liniar regression model
linear = linear_model.LinearRegression()
#trains model
linear.fit(x_train, y_train)
#gets the accuracy of the model
acc = linear.score(x_test, y_test)
print("Accuracy: \n", acc)

#liniar regrssion function: Y = m + n + ... + z * X + b
#gets the coefficient and the intercept values
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

#the model makes predictions for the dataset with liniar regression
predictions = linear.predict(x_test)

#prints the predictions
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
