import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pyswarms as ps


class BasePSOForecaster:

    def __init__(self, series, lag=None, max_lag=12, test_size=11,
                 n_particles=40, iters=30):

        self.series = np.array(series).astype(float)

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

        # scaler para estabilizar modelos
        self.scaler = StandardScaler()


    # ---------------------------------
    # crear lags + tendencia
    # ---------------------------------

    def create_lags(self):

        # ESCALAR SERIE
        scaled_series = self.scaler.fit_transform(
            self.series.reshape(-1, 1)
        ).flatten()

        X, y = [], []

        for i in range(self.max_lag, len(scaled_series)):

            lags = scaled_series[i-self.max_lag:i]

            trend = i

            row = np.concatenate([lags, [trend]])

            X.append(row)
            y.append(scaled_series[i])

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

                    try:

                        model = self.build_model()

                        model.fit(X_tr, y_tr)

                        pred = model.predict(X_val)

                        rmse = np.sqrt(mean_squared_error(y_val, pred))

                        rmse_folds.append(rmse)

                    except Exception:
                        rmse_folds.append(1e6)

                # penalizar demasiadas features
                penalty = 0.01 * np.sum(mask)

                scores.append(np.mean(rmse_folds) + penalty)

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

        y_pred_scaled = self.model.predict(X_test)

        # DES-ESCALAR
        y_pred = self.scaler.inverse_transform(
            y_pred_scaled.reshape(-1,1)
        ).flatten()

        y_test_real = self.scaler.inverse_transform(
            y_test.reshape(-1,1)
        ).flatten()

        self.y_test = y_test_real
        self.y_pred = y_pred

        return {
            "model": self.model,
            "selected_lags": self.selected_lags,
            "rmse": np.sqrt(mean_squared_error(y_test_real, y_pred)),
            "r2": r2_score(y_test_real, y_pred),
            "y_test": y_test_real,
            "y_pred": y_pred
        }

    # ---------------------------------
    # reentrenar con toda la serie
    # ---------------------------------

    def refit_full_series(self):

        if self.selected_mask is None:
            raise ValueError("Primero debes ejecutar fit()")

        X, y = self.create_lags()

        # usar los mismos lags seleccionados por PSO
        X = X[:, self.selected_mask]

        # construir modelo nuevamente
        self.model = self.build_model()

        # entrenar con TODA la serie
        self.model.fit(X, y)
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

        series_scaled = self.scaler.transform(
            self.series.reshape(-1,1)
        ).flatten()

        series = list(series_scaled)

        predictions = []

        for _ in range(steps):

            last_values = series[-self.max_lag:]

            trend = len(series)

            row = np.concatenate([last_values, [trend]])

            X_input = row[self.selected_mask]

            X_input = X_input.reshape(1, -1)

            pred_scaled = self.model.predict(X_input)[0]

            pred = self.scaler.inverse_transform(
                [[pred_scaled]]
            )[0][0]

            predictions.append(pred)

            # agregar escalado para siguiente paso
            series.append(pred_scaled)

        return np.array(predictions)


    # ---------------------------------
    # método que cada modelo debe definir
    # ---------------------------------

    def build_model(self):
        raise NotImplementedError