from forecast_model.base_model import BasePSOForecaster
class SVRPSOModel(BasePSOForecaster):

    def build_model(self):

        return SVR(kernel="rbf")