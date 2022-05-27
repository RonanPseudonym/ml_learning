from regression.base import RegressionBase
from sklearn.linear_model import LogisticRegression as _LR
import numpy as np

class LogisticRegression(RegressionBase):
    def __init__(self, data):
        super().__init__(data, self)

    def do_regression(self):
        x = np.array([int(i.x*10) for i in self.data]).reshape((-1, 1))
        y = np.array([int(i.y*10) for i in self.data]).reshape((-1))

        model = _LR(max_iter=1000)
        model.fit(x, y)

        expected = y[::]
        predicted = model.predict(x)

        return {"pred": predicted}
        # return {"slope":slope, "intercept":intercept, "x":x, "y":y}
