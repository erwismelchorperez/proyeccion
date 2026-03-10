class ModelRunner:

    def __init__(self, series, lag=12, test_size=12):

        self.series = series
        self.lag = lag
        self.test_size = test_size
        self.models = []

    def add_model(self, model):

        model.series = self.series
        model.lag = self.lag
        model.test_size = self.test_size

        self.models.append(model)

    def run(self):

        results = {}

        for model in self.models:

            res = model.fit()
            """
            y_pred, y_test = model.predict()
            results[model.__class__.__name__] = {
                "y_pred": y_pred,
                "y_test": y_test,
                "rmse": model.rmse(y_test, y_pred)
            }
            """
            results[model.__class__.__name__] = res


        return results