class ModelRunner:

    def __init__(self, series, lag=12, test_size=12):

        self.series = series
        self.lag = lag
        self.test_size = test_size
        self.models = []

    def add_model(self, model):

        model.series = self.series
        model.max_lag = self.lag
        model.test_size = self.test_size

        self.models.append(model)

    def run(self, forecast_steps=13):

        results = {}

        for model in self.models:

            # 1 evaluar modelo
            res = model.fit()

            # 2 reentrenar con toda la serie
            model.refit_full_series()

            # 3 forecast futuro
            future = model.forecast(forecast_steps)

            results[model.__class__.__name__] = {
                "selected_lags": res["selected_lags"],
                "rmse": res["rmse"],
                "r2": res["r2"],
                "y_test": res["y_test"],
                "y_pred": res["y_pred"],
                "forecast": future
            }

        return results