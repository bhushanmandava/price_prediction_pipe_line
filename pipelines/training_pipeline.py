from PipeLineSteps.steps.outlier_detection_step import  outlier_detection_step
from PipeLineSteps.steps.handle_missing_values_step import handle_missing_values_step
from PipeLineSteps.steps.data_ingestion_step import data_ingestion_step
from PipeLineSteps.steps.data_spliter import data_splitter_step
from PipeLineSteps.steps.model_building_step import model_building_step
from PipeLineSteps.steps.model_evaluator_step import model_evaluator_step
from zenml import Model, pipeline, step

from PipeLineSteps.steps.feature_engineeering_steps import feature_engineering_step
@pipeline(
    model=Model(
        name="prices_predictor"
    ),
)
def ml_pipeline():
    raw_data = data_ingestion_step(file_path = "/Users/bhushanchowdary/Documents/house_predict_system/ZipData/archive.zip")
    filled_data = handle_missing_values_step(raw_data)
    engineered_data = feature_engineering_step(filled_data, strategy="log",features =["Gr Liv Area", "SalePrice"])
    clean_data = outlier_detection_step(engineered_data, column_name="SalePrice")
    X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column="SalePrice")
    model = model_building_step(X_train=X_train, y_train=y_train)
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )
    return model

if __name__ == "__main__":
    run = ml_pipeline()