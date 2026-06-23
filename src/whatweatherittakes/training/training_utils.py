import pandas as pd
import datetime as dt

def train_test_split(df: pd.DataFrame, strategy: str = "monthly_based"):
    """
    Splits the DataFrame into training and testing sets based on the specified strategy.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to split.
    - strategy (str): The splitting strategy to use. Options are "random", "time_based", or "monthly_based".

    Returns:
    - train_df (pd.DataFrame): The training set.
    - test_df (pd.DataFrame): The testing set.
    """
    
    if strategy == "random":
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        yield train_df, test_df
        
    elif strategy == "time_based":
        # Sort by time and split based on the specified test size
        date_splits = [dt.date(2026,1,1), dt.date(2026,2,1), dt.date(2026,3,1), dt.date(2026,4,1)]
        for date in date_splits:
            train_df = df[df["timestamp"] < date].copy()
            test_df = df.drop(train_df.index).copy()
            yield train_df, test_df
    elif strategy == "monthly_based":
        day_splits = [
            ((1, 21)),
            ((8, 31)),
            ((1, 7), (15, 31)),
            ((1, 14), (22, 31)),
        ]
        for day_split in day_splits:
            train_dfs = []
            for day_range in day_split:
                train_df = df[df['timestamp'].dt.day.between(day_range[0], day_range[1])].copy()
                train_dfs.append(train_df)
            train_df = pd.concat(train_dfs)
            test_df = df.drop(train_df.index).copy()
            yield train_df, test_df
        
    else:
        raise ValueError("Invalid strategy. Choose 'random', 'time_based', or 'monthly_based'.")