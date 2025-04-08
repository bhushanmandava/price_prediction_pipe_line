from src.feature_engineering import MainFeatureEngineering, LogTransformation, StandardScaler, MinMaxScaler, OneHotEncoding
import pandas as pd
import logging
from zenml import step

@step

def feature_engineering_step(df:pd.DataFrame,strategy:str ="log",features:list = None)->pd.DataFrame:
    if features is None:
        features=[]
    if strategy == "log":
        engineer = MainFeatureEngineering(LogTransformation(features))
    elif strategy == "standard_scaling":
        engineer = MainFeatureEngineering(StandardScaler(features))
    elif strategy == "minmax_scaling":
        engineer = MainFeatureEngineering(MinMaxScaler(features))
    elif strategy == "onehot_encoding":
        engineer = MainFeatureEngineering(OneHotEncoding(features))
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")

    transformed_df = engineer.apply_Feature_enginnering(df)
    return transformed_df
