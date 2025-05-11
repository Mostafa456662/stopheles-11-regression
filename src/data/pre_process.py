### ~~~ GLOBAL IMPORTS ~~~ ###
import pandas as pd
import tqdm
import warnings

### ~~~ LOCAL IMPORTS ~~~ ###
from dates import *

### ~~~ STATE DEFINITION ~~~ ###
warnings.filterwarnings("ignore")


def load_data(input_path: str) -> pd.DataFrame:
    """
    This functuion loads the data from the input path
    and returns it as a pandas dataframe.
    args:
        input_path: str, the path to the input file
    returns:
        df: pd.DataFrame, the dataframe of the data
    """
    return pd.read_csv(input_path)


def set_data_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set the data type of the dataframe
    args:
        df: pd.DataFrame
            it is the dataframe of the csv file
    return:
        df: pd.DataFrame
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["spain"] = df["spain"].astype("float")
    return df


def combine_date_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine the date and time column into one column timestamp
    args:
        df: pd.DataFrame, it is the dataframe of the csv file
    return:
        df: pd.DataFrame, it is the dataframe of the csv file
    """
    df["hour"] = df["hour"].str.split(" - ").str[0]
    df["timestamp"] = df["date"] + " " + df["hour"]
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all columns except the timestamp and spain
    args:
        df: pd.DataFrame, it is the dataframe of the csv file
    return:
        df: pd.DataFrame, it is the dataframe of the csv file
    """
    return df[["timestamp", "spain"]]


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    split the data in two diffrent dataset one for train and the other
    for test. i want the split ratio to 0.8 for train and 0.2 for test
    args:
        df: pd.DataFrame
    return:
        tr_df: pd.DataFrame, the train dataframes
        te_df: pd.DataFrame, the test dataframes
    """
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:].reset_index(drop=True)
    return train_df, test_df


def calculate_holiday_distances(df: pd.DataFrame, holidays: list) -> pd.DataFrame:
    """
    Calculate days to next holiday and days since last holiday.
    Holidays are given as (month, day) tuples.
    """

    # Create a list of holidays as day-of-year
    holiday_days = [
        pd.Timestamp(month=month, day=day, year=2000).dayofyear
        for month, day in holidays
    ]

    days_to_next = []
    days_since_last = []

    for doy in df["day_of_year"]:
        future_holidays = [h - doy for h in holiday_days if h - doy >= 0]
        past_holidays = [doy - h for h in holiday_days if doy - h >= 0]

        days_to_next.append(
            min(future_holidays) if future_holidays else (365 - doy + min(holiday_days))
        )
        days_since_last.append(
            min(past_holidays) if past_holidays else (doy + (365 - max(holiday_days)))
        )

    # Assign back to dataframe
    df["days_to_next_holiday"] = days_to_next
    df["days_since_last_holiday"] = days_since_last

    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add date features to the dataframe
        - day
        - month
        - year
        - hour
        - day of week
        - week of year
        - is weekend
        - is holiday
        - which season
        - which quarter
        - is month begining
        - is month ending
        - day of the year
        - days to next holidays
        - days since last holidays
        - is good weather
        - is bad weather
        - 2_peak 1_shoulder 0_low season
        - day or night
        ...
    drop timestamp column
    args:
        df: pd.DataFrame, it is the dataframe of the csv file
    return:
        df: pd.DataFrame, it is the dataframe of the csv file
    """

    # Extract date and time components
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.apply(int)
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x > 5 else 0)
    df["is_month_beginning"] = df["timestamp"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["timestamp"].dt.is_month_end.astype(int)

    # Add seasons based on month
    # represent each season occurring as 1, and 0 otherwise
    df["summer"] = df["month"].apply(lambda x: 1 if season_map[x] == "Summer" else 0)
    df["spring"] = df["month"].apply(lambda x: 1 if season_map[x] == "Spring" else 0)
    df["winter"] = df["month"].apply(lambda x: 1 if season_map[x] == "Winter" else 0)
    df["fall"] = df["month"].apply(lambda x: 1 if season_map[x] == "Fall" else 0)

    # check if the day contained a popular football match
    df["is_matchday"] = df.apply(
        lambda row: 1 if (row["month"], row["day"]) in matchdays else 0, axis=1
    )
    # Add quarters
    df["quarter"] = df["timestamp"].dt.quarter

    # Determine 'day or night' based on the hour, 1 for day, 0 for night
    df["day_or_night"] = df["hour"].apply(lambda x: 1 if 6 <= x < 18 else 0)

    # Seasonal categorization for '2_peak 1_shoulder 0_low season'
    df["season_category"] = df["month"].apply(
        lambda x: (
            2
            if season_map[x] == "Summer" or season_map[x] == "Spring"
            else (1 if season_map[x] == "Fall" else 0)
        )
    )

    # assign good and bad weather

    df["is_bad_weather"] = df.apply(
        lambda row: int(
            (row["month"], row["day"]) in bad_days or row["month"] in bad_full_months
        ),
        axis=1,
    )
    df["is_good_weather"] = 1 - df["is_bad_weather"]

    # Add holiday column (1 if holiday, 0 if not)
    df["is_holiday"] = df.apply(
        lambda row: 1 if (row["month"], row["day"]) in holidays else 0, axis=1
    )

    df = calculate_holiday_distances(df, holidays)
    # Drop the 'timestamp' column
    df.drop(columns=["timestamp"], inplace=True)

    return df


def add_target_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add target value features to the dataframe
        - lags [1, 3, 6, 12, 24, 48, 96, 192, 384]
        - moving average [1, 3, 6, 12, 24, 48, 96, 192, 384]
        - exponential moving average [1, 3, 6, 12, 24, 48, 96, 192, 384]
        - Rolling Standard Deviation features [1, 3, 6, 12, 24, 48, 96, 192, 384]
        - lagged difference features [1, 3, 6, 12, 24, 48, 96, 192, 384]
        - cumulative sum
        - cumulative mean
    args:
        df: pd.DataFrame, it is the dataframe of the csv file
    return:
        df: pd.DataFrame, it is the dataframe of the csv file
    """
    # Add lag features
    for lag in [1, 3, 6, 12, 24, 48, 96, 192, 384]:
        df[f"lag_{lag}"] = df["spain"].shift(lag)
    # Add moving average features
    for window in [1, 3, 6, 12, 24, 48, 96, 192, 384]:
        df[f"moving_avg_{window}"] = df["spain"].rolling(window=window).mean()
    # Add exponential moving average features
    for window in [1, 3, 6, 12, 24, 48, 96, 192, 384]:
        df[f"exp_moving_avg_{window}"] = (
            df["spain"].ewm(span=window, adjust=False).mean()
        )
    # Add rolling standard deviation features
    for window in [3, 6, 12, 24, 48, 96, 192, 384]:
        df[f"rolling_std_{window}"] = (
            df["spain"].rolling(window=window, min_periods=1).std()
        )
    # Add lagged difference features
    for lag in [1, 3, 6, 12, 24, 48, 96, 192, 384]:
        df[f"lag_diff_{lag}"] = df["spain"].diff(lag)
    # Add cumulative sum feature
    df["cumulative_sum"] = df["spain"].cumsum()
    # Add cumulative mean feature
    df["cumulative_mean"] = df["spain"].expanding().mean()
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in the dataframe. The method of imputation
    is up to the developer to decide. Therefore the developer is Incentivised

    args:
        df: pd.DataFrame, a dataframe with missing values
    return:
        df: pd.DataFrame, a dataframe with no missing values
    """
    df["spain"] = df["spain"].ffill()
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


def save_data(df: pd.DataFrame, path: str) -> int:
    """
    Save data to a csv file.

    args:
        df: pd.DataFrame
            it is the dataframe of the csv file
        path: str
            it is the path where the csv file will be saved

    return:
        0: int, if the data is saved successfully
        1: int, if the data is not saved successfully
    """
    try:
        # Save the dataframe to the provided path
        df.to_csv(path, index=False)  # index=False prevents writing row indices
        return 0  # Return 0 for success
    except Exception as e:
        print(f"Error saving file: {e}")
        return 1  # Return 1 for failure


def main() -> int:
    """"""
    ### init tqdm ###
    pbar = tqdm.tqdm(total=8)

    ### init some variables ###
    pbar.set_description("init variables")
    input_path: str = "dbs/raw/df.csv"
    train_path: str = "dbs/preprocessing/tr_df.csv"
    test_path: str = "dbs/preprocessing/te_df.csv"
    pbar.update(1)

    ### load the data ###
    pbar.set_description("load data")
    df: pd.DataFrame = load_data(input_path)
    df = combine_date_time(df)
    df = set_data_type(df)
    pbar.update(1)

    ### clean data ###
    pbar.set_description("clean data")
    df = drop_columns(df)
    df = impute_missing_values(df)
    pbar.update(1)

    ### split the data ###
    pbar.set_description("split data")
    df_tr: pd.DataFrame
    df_te: pd.DataFrame
    df_tr, df_te = split_train_test(df)
    pbar.update(1)

    ### add features to the data ###
    pbar.set_description("add date features")
    df_tr = add_date_features(df_tr)
    df_te = add_date_features(df_te)
    pbar.update(1)
    pbar.set_description("add target value features")
    df_tr = add_target_value_features(df_tr)
    df_te = add_target_value_features(df_te)
    pbar.update(1)

    ### impute missing values ###
    pbar.set_description("impute missing values")
    df_tr = impute_missing_values(df_tr)
    df_te = impute_missing_values(df_te)
    pbar.update(1)

    ### save the data ###
    pbar.set_description("save data")
    save_data(df_tr, train_path)
    save_data(df_te, test_path)
    pbar.update(1)

    ### clean up ###
    pbar.close()

    return 0


if __name__ == "__main__":
    main()
