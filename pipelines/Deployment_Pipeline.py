import os
import click
from pipelines.training_pipeline import ml_pipeline
from PipeLineSteps.steps.dynamic_importer import dynamic_importer
from PipeLineSteps.steps.model_loader_step import model_loader
from PipeLineSteps.steps.prediction_service_loader import prediction_service_loader
from PipeLineSteps.steps.predictor import predictor
from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from rich import print


requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")


@pipeline
def continuous_deployment_pipeline():
    """Pipeline for continuous deployment"""
    trained_model = ml_pipeline()  # Ensure this outputs a model object
    print(f"Deploying model: {trained_model}")  # Debugging output
    mlflow_model_deployer_step(workers=3, deploy_decision=True, model=trained_model)


@pipeline(enable_cache=False)
def inference_pipeline():
    """Run a batch inference job with data loaded from an API."""
    # Load batch data for inference
    batch_data = dynamic_importer()

    # Load the deployed model service
    model_deployment_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step",
    )

    # Run predictions on the batch data
    predictor(service=model_deployment_service, input_data=batch_data)


@click.command()
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
def run_main(stop_service: bool):
    """Run the prices predictor deployment pipeline"""
    model_name = "prices_predictor"

    if stop_service:
        # Get the MLflow model deployer stack component
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()

        # Fetch existing services with same pipeline name, step name, and model name
        existing_services = model_deployer.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            model_name=model_name,
            running=True,
        )

        if existing_services:
            existing_services[0].stop(timeout=10)
        else:
            print("No running services found.")
        return

    # Run the continuous deployment pipeline
    continuous_deployment_pipeline()

    # Get the active model deployer
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # Run the inference pipeline
    inference_pipeline()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs."
    )

    # Fetch existing services with the same pipeline name, step name, and model name
    service = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
    )
    if not service:
        print("No prediction service is currently running.")
    else:
        print(f"Prediction service is running at: {service[0].prediction_url}")


    if service[0]:
        print(
            f"The MLflow prediction server is running locally as a daemon "
            f"process and accepts inference requests at:\n"
            f"    {service[0].prediction_url}\n"
            f"To stop the service, re-run the same command and supply the "
            f"`--stop-service` argument."
        )


if __name__ == "__main__":
    run_main()
