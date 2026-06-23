import polars as pl
import datetime as dt
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


import logging

import pandas as pd
from sklearn.preprocessing import FunctionTransformer

LOGGER: logging.Logger = logging.getLogger(__name__)


class SelectSubset(FunctionTransformer):
    """Custom transformation class that allows to select a subset of features."""

    def __init__(self, subset_features: list[str]) -> None:
        """Selects subset features specified in subset_features."""
        self.subset_features = subset_features

        # TODO: put this back in place once we have trained and deployed a model with python 3.12
        # And remove transform method.

        # def _subset_feature_names_callable(
        #     self: SelectSubset,
        #     input_names: list[str],  # pylint: disable=unused-argument
        # ) -> list[str]:
        #     return self.subset_features

        # super().__init__(
        #     func=self._select_subset,
        #     feature_names_out=_subset_feature_names_callable,
        # )
        super().__init__()

    def _select_subset(self, x: pd.DataFrame) -> pd.DataFrame:
        LOGGER.info("Columns before transform: %s", x.columns)
        y = x[self.subset_features]
        LOGGER.info("Columns after transform: %s", y.columns)
        return y

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms."""
        return self._select_subset(X)



class LgbmEarlyStoppingPipeline(Pipeline):
    """Pipeline that forwards transformed eval_set to final LGBM step."""

    def fit(self, X, y=None, *, eval_set=None, callbacks=None, **fit_params):
        Xt = X

        for name, transformer in self.steps[:-1]:
            if transformer == "passthrough":
                continue
            Xt = transformer.fit_transform(Xt, y)

        final_name, final_estimator = self.steps[-1]

        final_fit_params = {}
        for key, value in fit_params.items():
            prefix = f"{final_name}__"
            if key.startswith(prefix):
                final_fit_params[key.split("__", 1)[1]] = value

        if eval_set is not None:
            transformed_eval_set = []
            for X_val, y_val in eval_set:
                X_val_t = X_val
                for _, transformer in self.steps[:-1]:
                    if transformer == "passthrough":
                        continue
                    X_val_t = transformer.transform(X_val_t)
                transformed_eval_set.append((X_val_t, y_val))
            final_fit_params["eval_set"] = transformed_eval_set

        if callbacks is not None:
            final_fit_params["callbacks"] = callbacks

        final_estimator.fit(Xt, y, **final_fit_params)
        return self

def run_cross_validation(
    model: LgbmEarlyStoppingPipeline,
    df: pl.DataFrame,
    target_col: str,
    strategy: str = "monthly_based",
    early_stopping_rounds: int = 500,
    verbose: bool = False,
):
    """
    Run cross-validation on the provided DataFrame using the specified strategy.

    Parameters:
    - model: A fitted-or-unfitted LgbmEarlyStoppingPipeline instance.
    - df (pl.DataFrame): The input DataFrame containing features and target.
    - target_col (str): Name of the target column.
    - strategy (str): The splitting strategy to use. Options are "random", "time_based", or "monthly_based".
    - early_stopping_rounds (int): Patience for LightGBM early stopping.
    - verbose (bool): Whether to print LightGBM training logs.

    Returns:
    - metrics (dict): Per-fold r2, mse and mae scores.
    """
    metrics = {}
    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=verbose)]
    if verbose:
        callbacks.append(lgb.log_evaluation(early_stopping_rounds))

    for i, (train_df, test_df) in enumerate(train_test_split(df, strategy=strategy)):
        X_train = train_df.drop(target_col).to_pandas()
        y_train = train_df[target_col].to_pandas()
        X_test = test_df.drop(target_col).to_pandas()
        y_test = test_df[target_col].to_pandas()

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks)
        preds = model.predict(X_test)

        metrics[i] = {
            "r2": r2_score(y_test, preds),
            "mse": mean_squared_error(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
        }
        print(metrics[i])
    return metrics
        

def train_test_split(df: pl.DataFrame, strategy: str = "monthly_based"):
    """
    Splits the DataFrame into training and testing sets based on the specified strategy.

    Parameters:
    - df (pl.DataFrame): The input DataFrame to split.
    - strategy (str): The splitting strategy to use. Options are "random", "time_based", or "monthly_based".

    Returns:
    - train_df (pl.DataFrame): The training set.
    - test_df (pl.DataFrame): The testing set.
    """
    
    if strategy == "random":
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        yield train_df, test_df
        
    elif strategy == "time_based":
        # Sort by time and split based on the specified test size
        date_splits = [dt.date(2026,1,1), dt.date(2026,2,1), dt.date(2026,3,1), dt.date(2026,4,1)]
        for date in date_splits:
            train_df = df.filter(pl.col("Abfahrtsdatum") < date)
            test_df = df.drop(train_df.index)
            yield train_df, test_df
    elif strategy == "monthly_based":
        day_splits = [
            ((1, 21),),
            ((8, 31),),
            ((1, 7), (15, 31)),
            ((1, 14), (22, 31)),
        ]
        for day_split in day_splits:
            train_dfs = []
            train_masks = []
            for day_range in day_split:
                mask = pl.col("Abfahrtsdatum").dt.day().is_between(day_range[0], day_range[1])
                train_masks.append(mask)
            train_mask = train_masks[0]
            for m in train_masks[1:]:
                train_mask = train_mask | m
            # train = rows matching the mask
            train_df = df.filter(train_mask)
            test_df = df.filter(~train_mask)
            yield train_df, test_df
        
    else:
        raise ValueError("Invalid strategy. Choose 'random', 'time_based', or 'monthly_based'.")