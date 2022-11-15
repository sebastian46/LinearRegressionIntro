import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Import data
data = pd.read_csv("student-mat.csv", sep=";")

# print(data.head())
data = data.replace({'schoolsup': {'yes': 1, 'no': 0}})
# data = data.replace({'famsup': {'yes': 1, 'no': 0}})
# data = data.replace({'romantic': {'yes': 1, 'no': 0}})
# data = data.replace({'famrel': {'yes': 1, 'no': 0}})
# data = data.replace({'activities': {'yes': 1, 'no': 0}})

# trim dataset to include what we want
# this will produce 5-D graph
# data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# G2 and schoolsup have high coe (.95 and .65)
# this variation has fewer variables and better accuracy (.85 to .95)
data = data[["G1", "G2", "G3", "schoolsup", "failures"]]

# more doesn't equal better. seems to give random results from .65 acc to .95 acc
# data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "schoolsup", "famrel", "freetime", "goout", "health"]]

# what we're trying to predict
predict = "G3"

# data minus predict (attributes)
X = np.array(data.drop([predict], 1))
# what we're trying to predict (labels)
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
'''
best = 0
for i in range(1000):

    # We're taking all of our attributes and our labels, split them out into 4 different arrays
    # We have to have different data to train vs test. Otherwise, it'll already have an answer for that data
    # We're splitting off 10% of our data for testing purposes
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    # fit data to find best fit line (linear)
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy > best:
        best = accuracy
        # will save pickle file to directory
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
'''
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# Higher coefficient means more weight attribute has defining prediction
# From my testing, found that G2 has the highest weight of .95+
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

# Final grade prediction, data, actual grade
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'failures'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
