from abc import ABC, abstractmethod

import pandas as pd

class DataInspection(ABC):
    @abstractmethod
    def inspect(self,df: pd.DataFrame):
        pass


class DataTypesInspection(DataInspection):
    def inspect(self, df: pd.DataFrame):
        print("\n data types and non-null counts")
        print(df.info())

class SymmaryStatisticsInspection(DataInspection):
    def inspect(self, df: pd.DataFrame):
        print("\n summary statistics")
        print(df.describe())
        print("\n summary for the statics of a catogorical variable")
        print(df.describe(include=["O"]))
    

class DataInspector:
    def __init__(self, inspection: DataInspection):
        self._inspection = inspection
    def set_inspection(self, inspection: DataInspection):
        self._inspection = inspection
    def execute_inspection(self, df: pd.DataFrame):
        self._inspection.inspect(df)

if __name__ =="__main__":
    pass