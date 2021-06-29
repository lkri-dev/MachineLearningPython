# This project is for testing out Machine Learning tool in python for an education course.
# Anaconda Env tensorflow name of Env: tensor
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

# constant bool to determines whether to train new model or load from file.
isTrainingNewModel = False
# diffing dataset and seperators
data = pd.read_csv("student-mat.csv", sep=";")
# filtering dataset
data = data[["G1", "G2", "G3", "studytime", "absences"]]  # Features of the dataset

# Which attribute to be predicted
predict = "G3"  # Taget Value

# Drops the value in predict from dataset
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

if isTrainingNewModel:
    #for loob code is for retraining a new model and saving it to a pickle file if accuracy is better
    bestAcc = 0
    for _ in range(500):
        #Splits dataset up into trainig data and test data, so it does not train on the same data as it tests on.
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

        # makes a liniar regression model
        linear = linear_model.LinearRegression()
        # trains model
        linear.fit(x_train, y_train)

        # gets the accuracy of the model
        acc = linear.score(x_test, y_test)
        print("Accuracy: \n", acc)

        if acc > bestAcc:
            bestAcc = acc
            with open("studentmodel.pickle", "rb") as pickle_in:
                pickle_out = pickle.load(pickle_in)

            if acc > pickle_out["Accuracy"]:
                # saves the model in a pickel file
                with open("studentmodel.pickle", "wb") as file:
                    pickle.dump({"Model": linear, "Accuracy": bestAcc}, file)

    print("Best Accuracy: \n", bestAcc)
else:
    # Splits dataset up into training data and test data, so it does not train on the same data as it tests on.
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # loads the trained model from a pickle file
    with open("studentmodel.pickle", "rb") as pickle_in:
        pickle_out = pickle.load(pickle_in)

    linear = pickle_out["Model"]

    # gets the accuracy of the model
    acc = linear.score(x_test, y_test)
    print("Accuracy: {0} / {1}\n", acc, pickle_out["Accuracy"])

# liniar regrssion function: Y = m + n + ... + z * X + b
# gets the coefficient and the intercept values
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

# the model makes predictions for the dataset with linear regression
predictions = linear.predict(x_test)

# prints the predictions
# for x in range(len(predictions)):
#    print(predictions[x], x_test[x], y_test[x])

p = "G2"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()
