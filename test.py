import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from matplotlib import style
import pickle
import matplotlib.pyplot as plt

data = pd.read_csv("student_mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

'''
best = 0
for i in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)  # acc stands for accuracy
    print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        with open("student_grades.pickle", "wb") as f:
            pickle.dump(linear, f)

    print(best)
'''

# Load model
pickle_in = open("student_grades.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)  # These are each slope value
print('Intercept: \n', linear.intercept_)  # This is the intercept

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# Drawing and plotting model

plot = "G2"
style.use("ggplot")
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()
