from regression.base import RegressionBase
from scipy import stats
import pandas as pd

class LinearRegression(RegressionBase):
    def __init__(self, data):
        super().__init__(data, self)

    def do_regression(self):
#        x = [j.x for i, j in enumerate(self.data) if i%2]
#        y = [j.y for i, j in enumerate(self.data) if i%2]

        x = [j.x for i, j in enumerate(self.data)]
        y = [j.y for i, j in enumerate(self.data)]

        slope, intercept, _, _, _ = stats.linregress(x, y)

        df = pd.DataFrame({"x":x, "y":y}, columns=["x", "y"])
        matrix = pd.crosstab(df['x'], df['y'], rownames=['X'], colnames=['Predicted'])

        return {"slope":slope, "intercept":intercept, "x":x, "y":y}
