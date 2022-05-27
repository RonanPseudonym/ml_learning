import matplotlib.pyplot as plt

class RegressionBase():
    def __init__(self, data, core):
        self.data = data
        # self.old_data = data
        self.core = core

        self.slope = self.core.do_regression()
        self.expand = lambda slope, x, intercept: \
            list(map(lambda i: slope * i + intercept, x))

    def map_over_x_y(self, x, y, slope, intercept):
        return (self.expand(slope, x, intercept), self.expand(slope, y, intercept))

    def render(self):
        # # Define x and y values
        # x = [i.x for i in self.data]
        # y = [i.x for i in self.data]

        # # Plot a simple line chart without any feature
        # plt.plot(x, y)

        x = [i.x for i in self.data]
        y = [i.y for i in self.data]

        plt.scatter(x, y)

        if "pred" in self.slope:
            mymodel = self.slope["pred"]
        else:
            mymodel = list(map(lambda i: self.slope["slope"] * i + self.slope["intercept"], x))

        plt.plot(x, mymodel)

        plt.show()
