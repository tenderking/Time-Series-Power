# model/xgb.py
import operator
import pandas as pd
import xgboost as xgb
from src.util import rmse

from src.plots import feature_importance_plot, xgb_forecasts_plot


class XGBoostModel:
    """
    A class for handling XGBoost model training, hyperparameter tuning,
    prediction, and evaluation for time series forecasting.
    """

    def __init__(self, params=None, objective="reg:squarederror"):
        if params is None:
            params = {
                "booster": "gbtree",
                "objective": objective,
                "subsample": 0.80,
                "colsample_bytree": 0.85,
                "eta": 0.1,
                "max_depth": 10,
                "seed": 42,
            }
        self.params = params
        self.model = xgb.XGBRegressor(**self.params)

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        early_stopping_rounds=50,
        eval_metric=rmse,
    ):
        """
        Trains the XGBoost model with early stopping.
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)

        eval_set = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            eval_set.append((dval, "eval"))

        self.model = xgb.train(
            self.params,
            dtrain,
            evals=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=True,
            feval=eval_metric,  # Custom evaluation metric
        )

    def predict(self, X):
        """
        Predicts using the trained XGBoost model.
        """
        X_dmatrix = xgb.DMatrix(X)  # Convert to DMatrix
        return self.model.predict(X_dmatrix)

    def get_feature_importance(self, importance_type="gain"):
        """
        Returns the feature importance as a sorted list of tuples.
        """
        importance = self.model.get_score(importance_type=importance_type)
        return sorted(importance.items(), key=operator.itemgetter(1))

    def plot_feature_importance(
        self, importance_type="gain", title="Feature Importance"
    ):
        """
        Plots the feature importance.
        """
        importance_sorted = self.get_feature_importance(importance_type)
        # save the feature importance to a CSV file
        save_path = "data/processed"
        df = pd.DataFrame(importance_sorted, columns=["feature", "fscore"])
        df["fscore"] = df["fscore"] / df["fscore"].sum()
        df.to_csv(f"{save_path}/{title}.csv", index=False)
        feature_importance_plot(importance_sorted, title)

    def forecast(self, df_unseen, df_test, Y_test, plot_start):
        """
        Generates forecasts, plots the results, and optionally saves the data to CSV.
        """

        # Predictions on test data (use the scikit-learn predict method)
        Y_hat = self.model.predict(df_test.iloc[:, 1:])
        Y_hat_df = pd.DataFrame(Y_hat, index=Y_test.index, columns=["predicted"])

        # Predictions on unseen future data (use the scikit-learn predict method)
        unseen_y = self.model.predict(df_unseen)
        forecasts_df = pd.DataFrame(
            unseen_y, index=df_unseen.index, columns=["forecasts"]
        )
        # Save the predictions and forecasts to CSV if save_path is provided
        save_path = "data/processed/xgb"
        Y_hat_df.to_csv(f"{save_path}_predictions.csv")
        forecasts_df.to_csv(f"{save_path}_forecasts.csv")

        # Plot forecast results
        xgb_forecasts_plot(
            plot_start,
            Y_hat_df["predicted"],  # Use the existing predictions from Y_hat_df
            Y_test,
            Y_hat_df,
            forecasts_df,
            "XGBoost Forecast",
        )
