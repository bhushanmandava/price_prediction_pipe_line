import logging
from abc import ABC, abstractmethod

import pandas as pd

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MissingValues(ABC):
    @abstractmethod
    def handle(self, df:pd.DataFrame)->pd.DataFrame:
        pass
class DropMissingValues(MissingValues):
    def __init__(self,axis:0,thresh=None):
        self.axis = axis
        self.thresh = thresh
    def handle(self , df:pd.DataFrame)->pd.DataFrame:
        logging.info("Dropping missing values")
        df = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values dropped.")
        return df
class FillMissingValuesStrategy(MissingValues):
    def __init__(self, method="mean", fill_value=None):
        """
        Initializes the FillMissingValuesStrategy with a specific method or fill value.

        Parameters:
        method (str): The method to fill missing values ('mean', 'median', 'mode', or 'constant').
        fill_value (any): The constant value to fill missing values when method='constant'.
        """
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Filling missing values using method: {self.method}")

        df_cleaned = df.copy()
        if self.method == "mean":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )
        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].median()
            )
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(f"Unknown method '{self.method}'. No missing values handled.")

        logging.info("Missing values filled.")
        return df_cleaned
class MainMissingValue:
    def __init__(self, strategy:MissingValues):
        self._strategy = strategy
    def set_strategy(self, strategy:MissingValues):
        self._strategy = strategy
    def handle_stratagy(self, df:pd.DataFrame)->pd.DataFrame:
        logging.info("Handling missing values using the selected strategy.")
        return self._strategy.handle(df)
if __name__ == "__main__":
    pass
