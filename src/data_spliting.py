import logging
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataSplitter(ABC):
    @abstractmethod
    def split(self, df:pd.DataFrame , target_column:str )->Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        pass

class SimpleTrainTestSplit(DataSplitter):
    def __init__(self, test_size:float = 0.2, random_state:int = 42):
        self.test_size = test_size
        self.random_state = random_state
    def split(self, df:pd.DataFrame, target_column:str):
        logging.info("Splitting data into train and test sets")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        logging.info("Train-test split completed.")
        return X_train, X_test, y_train, y_test

class MainDataSplitter:
    def __init__(self,splitter:DataSplitter):
        self._splitter = splitter
    def set_strategy(self,splitter:DataSplitter):
        self._splitter = splitter
    def split_data(self, df:pd.DataFrame, target_column:str):
        logging.info("Splitting data using the selected strategy.")

        return self._splitter.split(df, target_column)
if __name__ == "__main__":
    pass