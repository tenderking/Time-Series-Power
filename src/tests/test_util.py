import pytest
import pandas as pd
from io import StringIO
# tests/test_util.py

from src.dataset import preprocess
from src.plots import config_plot, timeseries_plot
from src.util import bucket_avg, data_add_timesteps, date_transform, get_unseen_data


@pytest.fixture
def sample_csv():
    csv_data = """Date;Time;Global_active_power
    16/12/2006;17:24:00;4.216
    16/12/2006;17:25:00;5.360
    16/12/2006;17:26:00;5.374"""
    return StringIO(csv_data)


@pytest.fixture
def sample_df():
    # A preprocessed dataframe for testing
    data = {
        "datetime": pd.to_datetime(
            ["2006-12-16 17:24:00", "2006-12-16 17:25:00", "2006-12-16 17:26:00"]
        ),
        "Global_active_power": [4.216, 5.360, 5.374],
        "Global_reactive_power": [0.418, 0.436, 0.498],
        "Voltage": [234.840, 233.630, 233.290],
        "Global_intensity": [18.400, 23.000, 23.000],
        "Sub_metering_1": [0.000, 0.000, 0.000],
        "Sub_metering_2": [1.000, 1.000, 2.000],
        "Sub_metering_3": [17.000, 16.000, 17.000],
    }
    return pd.DataFrame(data).set_index("datetime")


# Test for preprocess function
def test_preprocess(sample_csv):
    # Test with parse_dates=False
    df = preprocess(N_rows=3, parse_dates=False, filename=sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 3  # Now it should return 3 rows as expected
    assert "Date" in df.columns
    assert "Time" in df.columns
    assert "Global_active_power" in df.columns

    # Test with parse_dates=[['Date', 'Time']]
    df = preprocess(N_rows=3, parse_dates=[["Date", "Time"]], filename=sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 3
    assert isinstance(df.index, pd.DatetimeIndex)


# Test for bucket_avg
def test_bucket_avg(sample_df):
    resampled_df = bucket_avg(sample_df["Global_active_power"], "min")  # Use 'min'
    assert len(resampled_df) == 3, "Resampled data length mismatch."
    assert (
        resampled_df.index.freq == "min"
    ), "Resampled frequency should be 1 minute."  # Use 'min'


# Test for timeseries_plot
def test_timeseries_plot(sample_df):
    y = sample_df["Global_active_power"]
    # Just test if it runs without errors, since it creates a plot
    try:
        timeseries_plot(y, "b", "Global Active Power")
    except Exception as e:
        pytest.fail(f"Plotting failed: {e}")


# Test for config_plot
def test_config_plot():
    assert config_plot() == 1, "config_plot function should return 1."


# Test for date_transform
def test_date_transform(sample_df):
    df_transformed = date_transform(
        sample_df, encode_cols=["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    )
    assert "Year" in df_transformed.columns, "Year column missing after transformation."
    assert "Sub_metering_1_0.0" in df_transformed.columns, "One-hot encoding failed."


# Test for get_unseen_data
def test_get_unseen_data():
    df_unseen = get_unseen_data(
        "2024-10-01", steps=5, encode_cols=[], bucket_size="min"
    )  # Use 'min'
    assert df_unseen.shape == (5, 1), "Unseen data has incorrect shape."
    assert (
        "Global_active_power" in df_unseen.columns
    ), "Column name mismatch in unseen data."


# Test for data_add_timesteps
def test_data_add_timesteps(sample_df):
    lag = 4
    original_columns = sample_df.shape[1]

    # Apply the function to add timesteps
    df_lagged = data_add_timesteps(sample_df, "Global_active_power", lag=lag)

    # The number of added columns should be (lag // 2)
    expected_columns = original_columns + (lag // 2)

    assert (
        df_lagged.shape[1] == expected_columns
    ), f"Incorrect number of columns after adding timesteps. Expected {expected_columns}, but got {df_lagged.shape[1]}."

    df_lagged.dropna(inplace=True)  # Drop rows with NaN values

    assert (
        not df_lagged.isna().any().any()
    ), "There should be no NaN values after adding timesteps."
