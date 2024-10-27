import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


class ArimaModel:
    def __init__(self, arima_para, seasonal_para):
        # Define the p, d and q parameters in Arima(p,d,q)(P,D,Q) models
        p = arima_para["p"]
        d = arima_para["d"]
        q = arima_para["q"]
        # Generate all different combinations of p, q and q triplets
        self.pdq = list(itertools.product(p, d, q))
        # Generate all different combinations of seasonal p, q and q triplets
        self.seasonal_pdq = [
            (x[0], x[1], x[2], seasonal_para) for x in set(itertools.product(p, d, q))
        ]

    def fit(self, ts):
        warnings.filterwarnings("ignore")
        results_list = []

        # Generate all unique combinations of param and param_seasonal
        all_params = set(itertools.product(self.pdq, self.seasonal_pdq))

        for param, param_seasonal in all_params:
            try:
                mod = sm.tsa.statespace.SARIMAX(
                    ts,
                    order=param,
                    seasonal_order=param_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                results = mod.fit()

                # Ensure results.aic is a scalar value before appending
                if isinstance(results.aic, (int, float, np.float64)):
                    print(
                        f"ARIMA{param}x{param_seasonal}seasonal - AIC:{results.aic:.2f}"
                    )
                    results_list.append([param, param_seasonal, results.aic])
                else:
                    print(
                        f"ARIMA{param}x{param_seasonal}seasonal - AIC has unexpected type: {type(results.aic)}"
                    )
                    results_list.append([param, param_seasonal, np.nan])

            except Exception as e:
                print(f"An error occurred: {e}")
                results_list.append([param, param_seasonal, np.nan])

        # Convert results_list to NumPy array with object dtype to handle mixed types
        results_list = np.array(results_list, dtype=object)
        lowest_AIC = np.argmin(results_list[:, 2])
        print(
            "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        )
        print(
            "ARIMA{}x{}seasonal with lowest_AIC:{}".format(
                results_list[lowest_AIC, 0],
                results_list[lowest_AIC, 1],
                results_list[lowest_AIC, 2],
            )
        )
        print(
            "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        )

        mod = sm.tsa.statespace.SARIMAX(
            ts,
            order=results_list[lowest_AIC, 0],
            seasonal_order=results_list[lowest_AIC, 1],
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.final_result = mod.fit()
        print("Final model summary:")
        print(self.final_result.summary().tables[1])
        print("Final model diagnostics:")
        self.final_result.plot_diagnostics(figsize=(15, 12))
        plt.tight_layout()
        plt.savefig("reports/figures/model_diagnostics.png", dpi=300)
        plt.show()

    def forecast(self, ts, n_steps, ts_label):
        # Get forecast n_steps ahead in future
        pred_uc = self.final_result.get_forecast(steps=n_steps)

        # Get confidence intervals of forecasts
        pred_ci = pred_uc.conf_int()
        ax = ts.plot(label="observed", figsize=(15, 10))
        pred_uc.predicted_mean.plot(ax=ax, label="Forecast in Future")
        ax.fill_between(
            pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color="k", alpha=0.25
        )
        ax.set_xlabel("Time")
        ax.set_ylabel(ts_label)
        plt.tight_layout()
        plt.savefig(f"reports/figures/{ts_label}_forcast.png", dpi=300)
        plt.legend()
        plt.show()
