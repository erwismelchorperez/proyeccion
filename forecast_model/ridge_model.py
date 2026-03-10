from sklearn.linear_model import Ridge
from forecast_model.base_model import BasePSOForecaster

class RidgePSOModel(BasePSOForecaster):

    def build_model(self):

        return Ridge(alpha=1.0)