import logging
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml import ArtifactConfig, step
from zenml.client import Client

# Create experiment tracker from ZenML client
experiment_tracker = Client().active_stack.experiment_tracker
from zenml import Model

# Define the model metadata
model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Price prediction model for houses.",
)

@step
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:

    # Ensure X_train is a DataFrame and y_train is a Series
    if not isinstance(X_train, pd.DataFrame):
        logging.error(f"X_train is NOT a DataFrame, type: {type(X_train)}")
        raise TypeError(f"X_train must be a pandas DataFrame, but got {type(X_train)}")
    else:
        logging.info("X_train is a valid DataFrame!")

    if not isinstance(y_train, pd.Series):
        logging.error(f"y_train is NOT a pandas Series, type: {type(y_train)}")
        raise TypeError(f"y_train must be a pandas Series, but got {type(y_train)}")
    
    # Extract categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")

    # Define preprocessing pipelines for numerical and categorical features
    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Column transformer that applies the correct preprocessing steps to each column
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Define the full pipeline with preprocessing and a Linear Regression model
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])

    # Start a new MLflow run if none is active
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        # Enable autologging for scikit-learn to capture metrics, parameters, and artifacts
        mlflow.sklearn.autolog()

        logging.info("Building and training the Linear Regression model.")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")

        # Log the expected columns based on the preprocessing steps
        onehot_encoder = (
            pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
        )
        onehot_encoder.fit(X_train[categorical_cols])
        expected_columns = numerical_cols.tolist() + list(
            onehot_encoder.get_feature_names_out(categorical_cols)
        )
        logging.info(f"Model expects the following columns: {expected_columns}")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        # Ensure the MLflow run is properly ended
        mlflow.end_run()

    return pipeline
