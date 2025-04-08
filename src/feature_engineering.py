import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class FeatureEngineering(ABC):
    @abstractmethod
    def apply_transform(self,df:pd.DataFrame)->pd.DataFrame:
        pass
class LogTransformation(FeatureEngineering):
    def __init__(self,features):
        self.features = features
    def apply_transform(self, df:pd.DataFrame)->pd.DataFrame:
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed=df.copy()
        for feature in self.features:
            if feature in df_transformed.columns:
                df_transformed[feature]=np.log1p(df_transformed[feature])
            else:
                logging.warning(f"Feature {feature} not found in DataFrame columns.")
        return df_transformed
class StandardScaler(FeatureEngineering):
    def __init__(self,features):
        self.features = features
    def apply_transform(self, df:pd.DataFrame)-> pd.DataFrame:
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard scaling completed.")
        return df_transformed
class MinMaxScaler(FeatureEngineering):
    def __init__(self,features):
        self.features = features
    def apply_transform(self, df:pd.DataFrame)-> pd.DataFrame:
        logging.info(f"Applying Min-Max scaling to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Min-Max scaling completed.")
        return df_transformed
class OneHotEncoding(FeatureEngineering):
    def __init__(self,features):
        self.features = features
    def apply_transform(self, df:pd.DataFrame)-> pd.DataFrame:
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One-hot encoding completed.")
        return df_transformed
class MainFeatureEngineering:
    def __init__(self,strategy: FeatureEngineering):
        self._startegy = strategy
    def set_stratagy(self,strategy: FeatureEngineering):
        self._startegy = strategy
    def apply_Feature_enginnering(self, df:pd.DataFrame)-> pd.DataFrame:
        logging.info("Applying feature engineering strategy.")
        return self._startegy.apply_transform(df)

if __name__ == "__main__":
    pass