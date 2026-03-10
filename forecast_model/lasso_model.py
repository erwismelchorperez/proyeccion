from forecast_model.base_model import BasePSOForecaster
class LassoPSOModel(BasePSOForecaster):

    def build_model(self):

        return Lasso(alpha=0.01)