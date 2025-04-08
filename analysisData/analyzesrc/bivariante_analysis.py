from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class BivariateAnalysis(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1:str,feature2:str):
        pass


class NumericalVsNumerical(BivariateAnalysis):
    def analyze(self, df, feature1, feature2):
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

class CategoricalVsNumerical(BivariateAnalysis):
    def analyze(self, df, feature1, feature2):
        plt.figure(figsize=(10,6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

class MainBivariateAnalysis:
    def __init__(self,strategy:BivariateAnalysis):
        self._strategy = strategy
    def set_strategy(self,strategy:BivariateAnalysis):
        self._strategy = strategy
    def execute_analysis(self,df:pd.DataFrame,feature1:str,feature2:str):
        self._strategy.analyze(df,feature1,feature2)

if __name__ == "__main__":
    pass