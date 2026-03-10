from forecast_model.linear_model import LinearPSOModel
from forecast_model.ridge_model import RidgePSOModel
from forecast_model.runner_model import ModelRunner

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os


# ---------------------------------
# BUSCAR MEJOR LAG
# ---------------------------------

def train_windows(series, windows, test_size=12):

    best_rmse = float("inf")
    best_window = None

    for w in windows:

        try:

            runner = ModelRunner(series, lag=w, test_size=test_size)

            runner.add_model(LinearPSOModel(series, lag=w))
            runner.add_model(RidgePSOModel(series, lag=w))

            results = runner.run()

            rmse_mean = np.mean([r["rmse"] for r in results.values()])

            if rmse_mean < best_rmse:

                best_rmse = rmse_mean
                best_window = w

        except Exception as e:

            print("Error ventana", w, e)

    return best_window


# ---------------------------------
# ENSEMBLE
# ---------------------------------

def ensemble_forecast(linear_model, ridge_model, steps=13):

    f1 = linear_model.forecast(steps)
    f2 = ridge_model.forecast(steps)

    return (np.array(f1) + np.array(f2)) / 2


# ---------------------------------
# CORRECCION DE TENDENCIA
# ---------------------------------

def trend_correction(series, forecast):

    x = np.arange(len(series)).reshape(-1,1)

    model = LinearRegression()
    model.fit(x, series)

    future_x = np.arange(len(series), len(series)+len(forecast)).reshape(-1,1)

    trend = model.predict(future_x)

    return forecast + trend - trend.mean()


# ---------------------------------
# GRAFICOS
# ---------------------------------

def plot_real_vs_pred(codigo, y_test, y_pred, folder="plots"):

    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(8,4))

    plt.plot(y_test, label="Real")
    plt.plot(y_pred, label="Predicción")

    plt.title(f"Real vs Predicción - {codigo}")
    plt.legend()

    path = f"{folder}/{codigo}.png"

    plt.savefig(path)
    plt.close()


# ---------------------------------
# DATASET
# ---------------------------------

df = pd.read_csv("./dataset/proyectar_706101.csv")

windows = [3,6,12,24]

rmse_rows = []
test_rows = []
future_rows = []
best_rows = []

months = [
"Dec_2025","Jan_2026","Feb_2026","Mar_2026","Apr_2026",
"May_2026","Jun_2026","Jul_2026","Aug_2026",
"Sep_2026","Oct_2026","Nov_2026","Dec_2026"
]


# ---------------------------------
# LOOP PRINCIPAL
# ---------------------------------

for idx, row in df.iterrows():

    codigo = row["codigo"]

    serie = row.iloc[1:].values.astype(float)

    print("Procesando:", codigo)

    # -----------------------------
    # BUSCAR MEJOR LAG
    # -----------------------------

    best_lag = train_windows(serie, windows)

    print("Best lag:", best_lag)

    # -----------------------------
    # ENTRENAR MODELOS FINALES
    # -----------------------------

    runner = ModelRunner(serie, lag=best_lag)

    linear_model = LinearPSOModel(serie, lag=best_lag)
    ridge_model = RidgePSOModel(serie, lag=best_lag)

    runner.add_model(linear_model)
    runner.add_model(ridge_model)

    results = runner.run()

    # -----------------------------
    # GUARDAR RESULTADOS
    # -----------------------------

    for model in runner.models:

        name = model.__class__.__name__

        res = results[name]

        rmse_rows.append({
            "Codigo": codigo,
            "Modelo": name,
            "Lag": best_lag,
            "RMSE": res["rmse"]
        })

        row_test = {
            "Codigo": codigo,
            "Modelo": name
        }

        for i,v in enumerate(res["y_pred"]):

            row_test[f"T{i+1}"] = v

        test_rows.append(row_test)

    # -----------------------------
    # MEJOR MODELO
    # -----------------------------

    best_model_name = min(results, key=lambda x: results[x]["rmse"])

    best_result = results[best_model_name]

    best_rows.append({
        "Codigo": codigo,
        "Best_Model": best_model_name,
        "Lag": best_lag,
        "RMSE": best_result["rmse"]
    })

    # -----------------------------
    # GRAFICO
    # -----------------------------

    plot_real_vs_pred(
        codigo,
        best_result["y_test"],
        best_result["y_pred"]
    )

    # -----------------------------
    # FORECAST DE TODOS LOS MODELOS
    # -----------------------------

    for model in runner.models:

        name = model.__class__.__name__

        future = model.forecast(13)

        future = trend_correction(serie, future)

        row_future = {
            "Codigo": codigo,
            "Modelo": name
        }

        for m,v in zip(months, future):

            row_future[m] = float(v)

        future_rows.append(row_future)


    # -----------------------------
    # FORECAST ENSEMBLE
    # -----------------------------

    forecast = ensemble_forecast(linear_model, ridge_model)

    forecast = trend_correction(serie, forecast)

    row_future = {
        "Codigo": codigo,
        "Modelo": "Ensemble"
    }

    for m,v in zip(months, forecast):

        row_future[m] = float(v)

future_rows.append(row_future)


# ---------------------------------
# DATAFRAMES
# ---------------------------------

df_rmse = pd.DataFrame(rmse_rows)

df_test = pd.DataFrame(test_rows)

df_future = pd.DataFrame(future_rows)

df_best = pd.DataFrame(best_rows)


# ---------------------------------
# EXPORTAR EXCEL
# ---------------------------------

os.makedirs("./resultados", exist_ok=True)

output_file = "./resultados/forecast_resultados.xlsx"

with pd.ExcelWriter(output_file) as writer:

    df_rmse.to_excel(writer, sheet_name="RMSE", index=False)

    df_test.to_excel(writer, sheet_name="Test_predictions", index=False)

    df_future.to_excel(writer, sheet_name="Forecast_2026", index=False)

    df_best.to_excel(writer, sheet_name="Best_Model", index=False)


print("Excel generado:", output_file)
print("Gráficos guardados en carpeta: plots/")