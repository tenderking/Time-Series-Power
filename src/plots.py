from matplotlib import dates
import matplotlib.pyplot as plt
import pandas as pd
import os

# Ensure the directory exists
os.makedirs("reports/figures", exist_ok=True)


def config_plot():
    plt.style.use("seaborn-v0_8-paper")
    #    plt.rcParams.update({'axes.prop_cycle': cycler(color='jet')})
    plt.rcParams.update({"axes.titlesize": 20})
    plt.rcParams["legend.loc"] = "best"
    plt.rcParams.update({"axes.labelsize": 22})
    plt.rcParams.update({"xtick.labelsize": 16})
    plt.rcParams.update({"ytick.labelsize": 16})
    plt.rcParams.update({"figure.figsize": (10, 6)})
    plt.rcParams.update({"legend.fontsize": 20})
    return 1


def timeseries_plot(y, color, y_label):
    # y is Series with index of datetime
    days = dates.DayLocator()
    dfmt_minor = dates.DateFormatter("%m-%d")
    weekday = dates.WeekdayLocator(byweekday=(), interval=1)

    fig, ax = plt.subplots()
    ax.xaxis.set_minor_locator(days)
    ax.xaxis.set_minor_formatter(dfmt_minor)

    ax.xaxis.set_major_locator(weekday)
    ax.xaxis.set_major_formatter(dates.DateFormatter("\n\n%a"))

    ax.set_ylabel(y_label)
    ax.plot(y.index, y, color)
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    plt.savefig(f"reports/figures/{y_label}.png", dpi=300)


def xgb_forecasts_plot(plot_start, Y, Y_test, Y_hat, forecasts, title):
    """
    Plots the observed, predicted, and forecast values.

    Args:
        plot_start (str): The start date for plotting. Should be in a format
                          compatible with pd.to_datetime().
        Y (pd.Series): The observed values for the training period.
        Y_test (pd.Series): The observed values for the testing period.
        Y_hat (pd.Series or pd.DataFrame): The predicted values for the testing period.
        forecasts (pd.Series or pd.DataFrame): The forecast values.
        title (str): The title of the plot.
    """
    Y = pd.concat([Y, Y_test])
    # Convert plot_start to datetime
    plot_start_dt = pd.to_datetime(plot_start)

    # Ensure the index is monotonically increasing
    Y = Y.sort_index()

    # Slice the data from the plot_start_dt
    Y_plot = Y[Y.index >= plot_start_dt]
    ax = Y_plot.plot(label="observed", figsize=(15, 10))
    Y_hat.plot(label="predicted", ax=ax)
    forecasts.plot(label="forecast", ax=ax)

    ax.fill_betweenx(
        ax.get_ylim(),
        pd.to_datetime(Y_test.index[0]),
        Y_test.index[-1],
        alpha=0.1,
        zorder=-1,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Global Active Power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"reports/figures/{title}.png", dpi=300)


def feature_importance_plot(importance_sorted, title):
    df = pd.DataFrame(importance_sorted, columns=["feature", "fscore"])
    df["fscore"] = df["fscore"] / df["fscore"].sum()

    plt.figure()
    # df.plot()
    df.plot(kind="barh", x="feature", y="fscore", legend=False, figsize=(12, 10))
    plt.title("XGBoost Feature Importance")
    plt.xlabel("relative importance")
    plt.tight_layout()
    plt.savefig(f"reports/figures/{title}.png", dpi=300)
