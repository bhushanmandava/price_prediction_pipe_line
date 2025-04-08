from typing import Tuple

import pandas as pd
from src.data_spliting import MainDataSplitter, SimpleTrainTestSplit
from zenml import step
# from zenml.artifacts import Output
# from zenml.artifacts import ArtifactType

@step
def data_splitter_step(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the data into training and testing sets using DataSplitter and a chosen strategy."""
    splitter = MainDataSplitter(splitter=SimpleTrainTestSplit())
    X_train, X_test, y_train, y_test = splitter.split_data(df, target_column)
    return X_train, X_test, y_train, y_test
