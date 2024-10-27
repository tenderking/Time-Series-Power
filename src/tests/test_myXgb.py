import pytest
import pandas as pd
import numpy as npr

# tests/test_myXgb.py
from src.dataset import xgb_data_split
from src.models.xgb import xgb_importance
from src.plots import feature_importance_plot, xgb_forecasts_plot
from src.util import get_unseen_data


@pytest.fixture
def sample_df():
    # Create a sample dataframe for testing
    data = {
        "datetime": pd.date_range(start="2010-01-01", periods=50, freq="h"),
        "Global_active_power": np.random.rand(50),
        "Global_reactive_power": np.random.rand(50),
    }
    df = pd.DataFrame(data).set_index("datetime")
    return df


def test_xgb_data_split(sample_df):
    # Test the xgb_data_split function
    bucket_size = "h"  # Set the frequency as a string for hourly
    unseen_start_date = "2010-01-15"
    steps = 10
    test_start_date = "2010-01-10"
    encode_cols = ["Global_active_power"]

    # Generate unseen data to test
    unseen = get_unseen_data(unseen_start_date, steps, encode_cols, bucket_size)

    # Ensure the sample_df has enough data before this date
    df = pd.concat([sample_df, unseen], axis=0)

    df_unseen, df_test, df_train = xgb_data_split(
        df, bucket_size, unseen_start_date, steps, test_start_date, encode_cols
    )

    assert not df_unseen.empty
    assert df_test.shape[0] > 0  # Ensure test data is not empty
    assert df_train.shape[0] > 0  # Ensure train data is not empty


def test_feature_importance_plot(sample_df):
    # Test the feature_importance_plot function
    importance_sorted = [("Global_active_power", 0.5), ("Global_reactive_power", 0.5)]
    title = "test_importance"

    # Run the plot function, this won't produce an assertion but will check for exceptions
    try:
        feature_importance_plot(importance_sorted, title)
        assert True  # If no exceptions are raised, the test passes
    except Exception as e:
        pytest.fail(f"feature_importance_plot raised an exception: {e}")


def test_xgb_importance(sample_df):
    # Test the xgb_importance function
    test_ratio = 0.2
    xgb_params = {"objective": "reg:squarederror", "max_depth": 2, "eta": 0.1}
    ntree = 10
    early_stop = 5
    plot_title = "test_xgb_importance"

    # This test will check if the function runs without exceptions
    try:
        xgb_importance(sample_df, test_ratio, xgb_params, ntree, early_stop, plot_title)
        assert True  # If no exceptions are raised, the test passes
    except Exception as e:
        pytest.fail(f"xgb_importance raised an exception: {e}")


def test_xgb_forecasts_plot(sample_df):
    # Test the xgb_forecasts_plot function
    plot_start = "2010-01-05"

    # Ensure sample_df has data for the plot_start
    if plot_start not in sample_df.index:
        pytest.skip(f"Plot start date {plot_start} is not in the sample_df index.")

    Y = sample_df["Global_active_power"].copy()
    Y_test = Y[plot_start:]

    # Ensure Y_test is not empty
    if Y_test.empty:
        pytest.fail(f"Y_test is empty. Check the plot_start date: {plot_start}.")

    # Mock predictions and forecasts
    Y_hat = Y_test + np.random.normal(0, 0.1, len(Y_test))
    forecasts = Y_test + np.random.normal(0, 0.1, len(Y_test))

    title = "test_forecast"

    # Check if the function runs without exceptions
    try:
        xgb_forecasts_plot(
            plot_start,
            Y,
            Y_test,
            pd.Series(Y_hat, index=Y_test.index),
            pd.Series(forecasts, index=Y_test.index),
            title,
        )
        assert True  # If no exceptions are raised, the test passes
    except Exception as e:
        pytest.fail(f"xgb_forecasts_plot raised an exception: {e}")
