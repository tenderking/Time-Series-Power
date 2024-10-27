import pandas as pd

from src.util import bucket_avg, date_transform, get_unseen_data


def preprocess(N_rows, parse_dates, filename):
    try:
        if isinstance(filename, str):
            # If filename is a string, open the file
            with open(filename) as f:
                total_rows = sum(1 for line in f)
        else:
            # If filename is a StringIO object, reset it to the beginning
            filename.seek(0)
            total_rows = sum(1 for line in filename)
            filename.seek(0)  # Reset again after counting lines

        print("total rows in the file: ", total_rows)

        # Only skip rows if total_rows is greater than N_rows

        # Read the data with specified columns and parse dates if specified
        df = pd.read_csv(
            filename,
            header=0,
            delimiter=";",
            nrows=N_rows,
        )

        if parse_dates:
            # Strip whitespace from Date and Time columns
            df["Date"] = df["Date"].str.strip()
            df["Time"] = df["Time"].str.strip()

            # Create a single datetime column from Date and Time
            df["datetime"] = pd.to_datetime(
                df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
            )
            df.drop(
                columns=["Date", "Time"], inplace=True
            )  # Drop original Date and Time columns
            df.set_index("datetime", inplace=True)  # Set datetime as index
        # Convert all columns except 'datetime' to numeric, forcing errors to NaN
        for col in df.columns:
            if col != "datetime":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def xgb_data_split(
    df, bucket_size, unseen_start_date, steps, test_start_date, encode_cols
):
    # Generate unseen data
    unseen = get_unseen_data(unseen_start_date, steps, encode_cols, bucket_size)
    df = pd.concat([df, unseen], axis=0)

    # Sort the DataFrame by the index to avoid KeyError
    df = df.sort_index()

    df = date_transform(df, encode_cols)
    print("df shape before split:", df.shape)

    # Data for forecast, skip the connecting point
    df_unseen = df[unseen_start_date:].iloc[:, 1:]
    df_test = df[test_start_date:unseen_start_date].iloc[:-1, :]
    df_train = df[:test_start_date]
    print("df_test shape:", df_test.shape)  # Print the shape of df_test
    print("df_test head:", df_test.head())
    return df_unseen, df_test, df_train


def load_and_process(n_rows, parse_dates, filename, bucket_size, steps, encode_cols):
    """Loads, preprocesses, and splits the data."""
    df = preprocess(n_rows, parse_dates, filename)
    G_power = df["Global_active_power"]
    df = pd.DataFrame(bucket_avg(G_power, bucket_size))
    df.dropna(inplace=True)

    last_timestamp = df.index[-1]

    # Define test and unseen start dates
    test_start_date = last_timestamp - pd.Timedelta(hours=2)
    unseen_start_date = last_timestamp - pd.Timedelta(hours=1)

    # Split data
    df_unseen, df_test, df_train = xgb_data_split(
        df, bucket_size, unseen_start_date, steps, test_start_date, encode_cols
    )

    return df_unseen, df_test, df_train
