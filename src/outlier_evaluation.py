from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
class OutlierDetection(ABC):
    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class ZScoreOutlierDetection(OutlierDetection):
    def __init__(self, threshold:3):
        self.threshold =threshold
    def detect(self, df:pd.DataFrame)->pd.DataFrame:
        logging.info("Detecting outliers using the Z-score method.")
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = z_scores > self.threshold
        logging.info(f"Outliers detected with Z-score threshold: {self.threshold}.")
        return outliers

class IQROutlierDetection(OutlierDetection):
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using the IQR method.")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
        logging.info("Outliers detected using the IQR method.")
        return outliers
class MainOutlierDetection:
    def __init__(self, strategy:OutlierDetection):
        self._strategy = strategy

    def set_strategy(self, strategy:OutlierDetection):
        self._strategy = strategy

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using the selected strategy.")
        return self._strategy.detect(df)
    def handle_outliers(self,df:pd.DataFrame,method="remove",**kwargs)->pd.DataFrame:
        outliers = self.detect(df)
        if method == "remove":
            df_cleaned = df[~outliers.any(axis=1)]
            logging.info("Outliers removed from the DataFrame.")
        elif method == "replace":
            for column in df.columns:
                if outliers[column].any():
                    df[column] = np.where(outliers[column], kwargs.get("replacement_value", 0), df[column])
            logging.info("Outliers replaced in the DataFrame.")
        else:
            raise ValueError("Unsupported method. Use 'remove' or 'replace'.")
        return df_cleaned
    def visualize_outliers(self, df: pd.DataFrame, features:list):
        logging.info("Visualizing Outliers")
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Outlier Visuvalization completed")
if __name__ == "__main__":
    pass