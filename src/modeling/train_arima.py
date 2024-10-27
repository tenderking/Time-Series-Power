import pandas as pd
from src.util import bucket_avg
from src.models.arima import ArimaModel
from src.plots import timeseries_plot, config_plot
from src.dataset import preprocess

# Constants
N_ROWS = 15000
PARSE_DATES = [["Date", "Time"]]
FILENAME = "data/raw/household_power_consumption.txt"
BUCKET_SIZE = "30T"
TS_LABEL = "G_power_avg"
PLOT_START = "2010-11-24 00:00:00"
PRED_START = "2010-11-25 14:00:00"
N_STEPS = 100


def main():
    config_plot()

    # we focus on the last 10 days data in Nov 2010
    df = preprocess(N_ROWS, PARSE_DATES, FILENAME)

    G_power = pd.to_numeric(df["Global_active_power"])
    # time series plot of one-minute sampling rate data
    timeseries_plot(G_power, "g", "Global_active_power")

    # we take a 30 minutes bucket average of our time series data to reduce noise.
    G_power_avg = bucket_avg(G_power, BUCKET_SIZE)
    # plot of 30 minutes average.
    timeseries_plot(G_power_avg, "g", TS_LABEL)

    # "Grid search" of seasonal ARIMA model.
    # the seasonal periodicy 24 hours, i.e. S=24*60/30 = 48 samples
    arima_para = {}
    arima_para["p"] = range(2)
    arima_para["d"] = range(2)
    arima_para["q"] = range(2)
    # the seasonal periodicy is  24 hours
    seasonal_para = round(24 * 60 / (float(BUCKET_SIZE[:-1])))
    arima = ArimaModel(arima_para, seasonal_para)

    arima.fit(G_power_avg)

    # Prediction on observed data starting on pred_start
    # observed and prediction starting dates in plots

    # Forecasts to unseen future data
    arima.forecast(G_power_avg, N_STEPS, TS_LABEL)


if __name__ == "__main__":
    main()
