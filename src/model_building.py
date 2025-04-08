import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelBuilding(ABC):
    @abstractmethod
    def build_train_model(self, X: pd.DataFrame, y: pd.Series) -> RegressorMixin:
        pass

class LinearRegressionModel(ModelBuilding):
    def build_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing Linear Regression model with scaling.")
        pipeline = Pipeline(
            [
                ("Scaler", StandardScaler()),
                ("model",LinearRegression()),
            ]
        )
        logging.info("Fitting Linear Regression model.")
        pipeline.fit(X_train, y_train)
        logging.info("Linear Regression model built successfully.")
        return pipeline
class MainModelBuilding:
    def __init__(self, model: ModelBuilding):
        self._model = model

    def set_model(self, model: ModelBuilding):
        self._model = model

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        logging.info("Building model using the selected strategy.")
        return self._model.build_train_model(X_train, y_train)

if __name__ == "__main__":
    pass