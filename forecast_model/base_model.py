import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import pyswarms as ps


class BasePSOForecasterV1:

    def __init__(self, series, lag=None, max_lag=12, test_size=11,
                 n_particles=40, iters=30):

        self.series = np.array(series)

        # compatibilidad con main
        if lag is not None:
            self.max_lag = lag
        else:
            self.max_lag = max_lag

        self.test_size = test_size
        self.n_particles = n_particles
        self.iters = iters

        self.model = None
        self.selected_lags = None
        self.y_test = None
        self.y_pred = None


    # ---------------------------------
    # crear lags
    # ---------------------------------

    def create_lags(self):

        X, y = [], []

        for i in range(self.max_lag, len(self.series)):
            X.append(self.series[i-self.max_lag:i])
            y.append(self.series[i])

        return np.array(X), np.array(y)


    # ---------------------------------
    # split temporal
    # ---------------------------------

    def split_data(self, X, y):

        split_index = len(X) - self.test_size

        if split_index <= 0:
            raise ValueError("Serie demasiado corta para el test_size")

        X_train = X[:split_index]
        X_test = X[split_index:]

        y_train = y[:split_index]
        y_test = y[split_index:]

        return X_train, X_test, y_train, y_test


    # ---------------------------------
    # PSO feature selection
    # ---------------------------------

    def pso_feature_selection(self, X_train, y_train):

        n_features = X_train.shape[1]

        tscv = TimeSeriesSplit(n_splits=3)

        def objective_function(particles):

            scores = []

            for particle in particles:

                mask = particle > 0.5

                if np.sum(mask) == 0:
                    scores.append(1e6)
                    continue

                rmse_folds = []

                for train_idx, val_idx in tscv.split(X_train):

                    X_tr = X_train[train_idx][:, mask]
                    X_val = X_train[val_idx][:, mask]

                    y_tr = y_train[train_idx]
                    y_val = y_train[val_idx]

                    model = self.build_model()

                    model.fit(X_tr, y_tr)

                    pred = model.predict(X_val)

                    rmse = np.sqrt(mean_squared_error(y_val, pred))

                    rmse_folds.append(rmse)

                scores.append(np.mean(rmse_folds))

            return np.array(scores)


        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=n_features,
            options={'c1':0.5,'c2':0.3,'w':0.9}
        )

        best_cost, best_pos = optimizer.optimize(
            objective_function,
            iters=self.iters
        )

        mask = best_pos > 0.5

        if np.sum(mask) == 0:
            mask[0] = True

        return mask


    # ---------------------------------
    # forecast futuro
    # ---------------------------------

    def forecast(self, steps=13):

        if self.model is None:
            raise ValueError("Modelo no entrenado")

        series = list(self.series)

        predictions = []

        mask = (self.selected_lags - 1)

        for _ in range(steps):

            last_values = series[-self.max_lag:]

            X_input = np.array(last_values)[mask]

            X_input = X_input.reshape(1, -1)

            pred = self.model.predict(X_input)[0]

            predictions.append(pred)

            series.append(pred)

        return np.array(predictions)


    # ---------------------------------
    # entrenamiento completo
    # ---------------------------------

    def fit(self):

        X, y = self.create_lags()

        X_train, X_test, y_train, y_test = self.split_data(X, y)

        mask = self.pso_feature_selection(X_train, y_train)

        self.selected_lags = np.where(mask)[0] + 1

        X_train = X_train[:, mask]
        X_test = X_test[:, mask]

        self.model = self.build_model()

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        self.y_test = y_test
        self.y_pred = y_pred

        return {
            "model": self.model,
            "selected_lags": self.selected_lags,
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "y_test": y_test,
            "y_pred": y_pred
        }


    # ---------------------------------
    # método que cada modelo debe definir
    # ---------------------------------

    def build_model(self):
        raise NotImplementedError

############################################################################
class BasePSOForecaster:

    def __init__(self, series, lag=None, max_lag=12, test_size=11,
                 n_particles=40, iters=30):

        self.series = np.array(series).astype(float)

        # compatibilidad con main
        if lag is not None:
            self.max_lag = lag
        else:
            self.max_lag = max_lag

        self.test_size = test_size
        self.n_particles = n_particles
        self.iters = iters

        self.model = None
        self.selected_lags = None
        self.selected_mask = None

        self.y_test = None
        self.y_pred = None


    # ---------------------------------
    # crear lags + tendencia
    # ---------------------------------

    def create_lags(self):

        X, y = [], []

        for i in range(self.max_lag, len(self.series)):

            lags = self.series[i-self.max_lag:i]

            # feature de tendencia temporal
            trend = i

            row = np.concatenate([lags, [trend]])

            X.append(row)
            y.append(self.series[i])

        return np.array(X), np.array(y)


    # ---------------------------------
    # split temporal
    # ---------------------------------

    def split_data(self, X, y):

        split_index = len(X) - self.test_size

        if split_index <= 0:
            raise ValueError("Serie demasiado corta para el test_size")

        X_train = X[:split_index]
        X_test = X[split_index:]

        y_train = y[:split_index]
        y_test = y[split_index:]

        return X_train, X_test, y_train, y_test


    # ---------------------------------
    # PSO feature selection
    # ---------------------------------

    def pso_feature_selection(self, X_train, y_train):

        n_features = X_train.shape[1]

        tscv = TimeSeriesSplit(n_splits=3)

        def objective_function(particles):

            scores = []

            for particle in particles:

                mask = particle > 0.5

                if np.sum(mask) == 0:
                    scores.append(1e6)
                    continue

                rmse_folds = []

                for train_idx, val_idx in tscv.split(X_train):

                    X_tr = X_train[train_idx][:, mask]
                    X_val = X_train[val_idx][:, mask]

                    y_tr = y_train[train_idx]
                    y_val = y_train[val_idx]

                    model = self.build_model()

                    model.fit(X_tr, y_tr)

                    pred = model.predict(X_val)

                    rmse = np.sqrt(mean_squared_error(y_val, pred))

                    rmse_folds.append(rmse)

                scores.append(np.mean(rmse_folds))

            return np.array(scores)


        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=n_features,
            options={'c1':0.5,'c2':0.3,'w':0.9}
        )

        best_cost, best_pos = optimizer.optimize(
            objective_function,
            iters=self.iters
        )

        mask = best_pos > 0.5

        if np.sum(mask) == 0:
            mask[0] = True

        return mask


    # ---------------------------------
    # entrenamiento completo
    # ---------------------------------

    def fit(self):

        X, y = self.create_lags()

        X_train, X_test, y_train, y_test = self.split_data(X, y)

        mask = self.pso_feature_selection(X_train, y_train)

        self.selected_mask = mask
        self.selected_lags = np.where(mask)[0] + 1

        X_train = X_train[:, mask]
        X_test = X_test[:, mask]

        self.model = self.build_model()

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        self.y_test = y_test
        self.y_pred = y_pred

        return {
            "model": self.model,
            "selected_lags": self.selected_lags,
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "y_test": y_test,
            "y_pred": y_pred
        }


    # ---------------------------------
    # predicción en test
    # ---------------------------------

    def predict(self):

        return self.y_pred, self.y_test


    # ---------------------------------
    # forecast futuro
    # ---------------------------------

    def forecast(self, steps=13):

        if self.model is None:
            raise ValueError("Modelo no entrenado")

        series = list(self.series)

        predictions = []

        for _ in range(steps):

            last_values = series[-self.max_lag:]

            trend = len(series)

            row = np.concatenate([last_values, [trend]])

            X_input = row[self.selected_mask]

            X_input = X_input.reshape(1, -1)

            pred = self.model.predict(X_input)[0]

            predictions.append(pred)

            series.append(pred)

        return np.array(predictions)


    # ---------------------------------
    # método que cada modelo debe definir
    # ---------------------------------

    def build_model(self):
        raise NotImplementedError