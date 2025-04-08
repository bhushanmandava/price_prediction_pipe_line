from src.handle_missing import MainMissingValue, DropMissingValues, FillMissingValuesStrategy
import pandas as pd
from zenml import step

@step 
def handle_missing_values_step( df:pd.DataFrame,strategy:str = "mean")->pd.DataFrame:

    if strategy == "drop":
        handler = MainMissingValue(strategy=DropMissingValues(axis=0))
    else:
        handler = MainMissingValue(strategy=FillMissingValuesStrategy(method=strategy))
    
    df_cleaned = handler.handle_stratagy(df)
    return df_cleaned