from sklearn.linear_model import LinearRegression
from forecast_model.base_model import BasePSOForecaster
class LinearPSOModel(BasePSOForecaster):

    def build_model(self):

        return LinearRegression()