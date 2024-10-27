# train.py

from sklearn.model_selection import train_test_split
from models.randomsearchcv import RandomSearchCVWrapper
import scipy.stats as st
import pandas as pd

from models.xgb import XGBoostModel
from src.dataset import load_and_process
from src.plots import config_plot

# Constants
N_ROWS = 18000
PARSE_DATES = [["Date", "Time"]]
FILENAME = "data/raw/household_power_consumption.txt"
ENCODE_COLS = ["Month", "DayofWeek", "Hour"]
BUCKET_SIZE = "5min"
VAL_RATIO = 0.3
NTREE = 300
EARLY_STOP = 10
PLOT_START = "2010-11-24 00:00:00"
XGB_PARAMS = {
    "booster": "gbtree",
    "objective": "reg:squarederror",  # regression task
    "subsample": 0.80,  # 80% of data to grow trees and prevent overfitting
    "colsample_bytree": 0.85,  # 85% of features used
    "eta": 0.1,
    "max_depth": 10,
    "seed": 42,  # for reproducible results
}


def main():
    """Main function to load data, train models, and generate forecasts."""
    config_plot()

    df_unseen, df_test, df_train = load_and_process(
        N_ROWS,
        PARSE_DATES,
        FILENAME,
        BUCKET_SIZE,
        steps=200,
        encode_cols=ENCODE_COLS,
    )

    # Display data shapes
    print("\n-----Xgboost on only datetime information---------\n")
    data_shapes = {
        "train and validation data ": df_train.shape,
        "test data ": df_test.shape,
        "forecasting data ": df_unseen.shape,
    }
    print(pd.DataFrame(list(data_shapes.items()), columns=["Data", "dimension"]))

    # Prepare data for model training
    Y = df_train.iloc[:, 0]
    X = df_train.iloc[:, 1:]
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=VAL_RATIO, random_state=42
    )
    Y_test = df_test.iloc[:, 0]

    # --- Train XGBoost Model ---
    model = XGBoostModel(XGB_PARAMS)
    model.fit(X_train, y_train, X_val, y_val)

    # Make predictions
    predictions = model.predict(X_val)
    print(predictions)

    print("---Initial model feature importance---")
    model.plot_feature_importance(title="Feature Importance")

    # --- Hyperparameter Tuning ---
    params_grid = {"max_depth": st.randint(6, 30)}

    # Use the RandomSearchCVWrapper for tuning
    model_for_tuning = XGBoostModel(XGB_PARAMS)

    random_search_wrapper = RandomSearchCVWrapper(
        model=model_for_tuning.model,
        param_distributions=params_grid,
        scoring="neg_mean_squared_error",
    )

    best_model = random_search_wrapper.tune_hyperparameters(X, Y)

    # Update the original XGBoost model with the best parameters
    model.model = best_model

    # --- Evaluation and Forecasting ---
    model.forecast(df_unseen, df_test, Y_test, PLOT_START)


if __name__ == "__main__":
    main()
