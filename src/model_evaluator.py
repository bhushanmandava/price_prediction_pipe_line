import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.sparse  # Important for checking sparse matrix type
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelEvaluator(ABC):
    @abstractmethod
    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        pass


class RegressionModelEvaluator(ModelEvaluator):
    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        # Ensure X_test is a dense pandas DataFrame
        if isinstance(X_test, scipy.sparse.spmatrix):
            X_test = pd.DataFrame(X_test.toarray())
        elif not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)

        # Ensure y_test is a pandas Series
        if not isinstance(y_test, pd.Series):
            y_test = pd.Series(y_test)

        logging.info("Evaluating regression model.")

        # Ensure the dimensions of X_test and y_test match
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError(f"Mismatch between X_test ({X_test.shape[0]}) and y_test ({y_test.shape[0]}) samples.")

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        logging.info(f"Model evaluation completed with MSE: {mse}, R^2: {r2}.")
        return {"MSE": mse, "R^2": r2}


class MainModelEvaluator:
    def __init__(self, evaluator: ModelEvaluator):
        self._evaluator = evaluator

    def set_evaluator(self, evaluator: ModelEvaluator):
        self._evaluator = evaluator

    def evaluate_model(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        logging.info("Evaluating model using the selected strategy.")
        return self._evaluator.evaluate(model, X_test, y_test)


if __name__ == "__main__":
    pass
