from data import IRIS
import matplotlib.pyplot as plt

def select_by_target(target):
    return [i for i in IRIS if i['target'] == target]

def get_x_y(target, x_ax, y_ax):
    x = []
    y = []

    for i in select_by_target(target):
        x.append(float(i[x_ax]))
        y.append(float(i[y_ax]))

    return (x, y)

def test_plot(x_axis, y_axis):
    plt.title(x_axis + " :: " + y_axis)

    xy_setosa = get_x_y("Iris-setosa", x_axis, y_axis)
    xy_versicolor = get_x_y("Iris-versicolor", x_axis, y_axis)
    xy_virginica = get_x_y("Iris-virginica", x_axis, y_axis)

    print(xy_setosa)

    plt.scatter(xy_setosa[0], xy_setosa[1], color='red')
    plt.scatter(xy_versicolor[0], xy_versicolor[1], color='green')
    plt.scatter(xy_virginica[0], xy_virginica[1], color='blue')

    plt.show()
