from forecast_model.base_model import BasePSOForecaster
class XGBPSOModel(BasePSOForecaster):

    def build_model(self):

        return XGBRegressor()