import regression, data
from point import Point

from testplot import test_plot

r = regression.LinearRegression([Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 2)])
r.render()

# TODO: Attempt to do it on my own, will probably get stuck
# Then go follow video in email up until I'm stuck and see if it helps

# TODO: Get rid of Point() and clean up code

# x = [i["sepal_length"] for i in virginica]
# y = [i["sepal_width"] for i in virginica]

#test_plot("petal width (cm)", "petal length (cm)")

#regression.LogisticRegression(
#    [point.Point(float(data.IRIS["sepal length (cm)"][i]), float(data.IRIS["sepal width (cm)"][i])) for i in range(len(data.IRIS)) if data.IRIS["target"][i] == "Iris-virginica"]
#    ).render()

# regression.LogisticRegression([point.Point(0, 0), point.Point(1, 1), point.Point(2, 2), point.Point(3, 2)]).render()
# regression.LinearRegression([point.Point(0, 0), point.Point(1, 1), point.Point(2, 2)]).render()
