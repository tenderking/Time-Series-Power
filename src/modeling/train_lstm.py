#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from src.models.lstm import LSTMModel
from src.util import WindowGenerator
from src.dataset import preprocess

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

VAL_RATIO = 0.3
TIMESTEPS = 100
N_ROWS = 18000
BUCKET_SIZE = "15min"
NUM_UNITS = 1024
NUM_FEATURES = 1
EPOCHS = 30
BATCH_SIZE = 32
PLOT_FIGSIZE = (15, 8)
PLOT_DPI = 300
PLOT_FILENAME = "reports/figures/lstm_predict_result.png"
DATA = "data/raw/household_power_consumption.txt"


# get LSTM data
def get_rnn_data(N_rows):
    parse_dates = [["Date", "Time"]]
    filename = DATA
    df = preprocess(N_rows, parse_dates, filename)

    # Calculate mean and std on the training set ONLY to avoid data leakage
    train_df = df[0 : int(len(df) * 0.7)]
    train_mean = train_df.mean()
    train_std = train_df.std()

    # Now split the data
    n = len(df)
    train_df = df[0 : int(n * 0.7)]
    val_df = df[int(n * 0.7) : int(n * 0.9)]
    test_df = df[int(n * 0.9) :]

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    num_features = df.shape[1]

    return train_df, val_df, test_df, num_features


if __name__ == "__main__":
    train_df, val_df, test_df, num_features = get_rnn_data(N_ROWS)

    wide_window = WindowGenerator(
        input_width=24,
        label_width=24,
        shift=1,
        label_columns=["Global_active_power"],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    # Create the model
    model = LSTMModel(num_units=NUM_UNITS, num_features=num_features)
    print("Input shape:", wide_window.example[0].shape)
    print("Output shape:", model(wide_window.example[0]).shape)

    # Train the model
    model.fit(epochs=EPOCHS, batch_size=BATCH_SIZE, window=wide_window)

    val_performance = {}
    performance = {}

    val_performance["LSTM"] = model.evaluate(wide_window.val, return_dict=True)
    performance["LSTM"] = model.evaluate(wide_window.test, verbose=0, return_dict=True)

plt.figure(figsize=PLOT_FIGSIZE)

wide_window.plot(model, plot_col="Global_active_power")  # Corrected line
plt.savefig(PLOT_FILENAME, dpi=PLOT_DPI)
