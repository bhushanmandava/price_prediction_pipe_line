import time
from zenml.steps import step
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService

@step(enable_cache=False)
@step(enable_cache=False)
def prediction_service_loader(pipeline_name: str, step_name: str) -> MLFlowDeploymentService:
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
    )
    retries = 3
    while not existing_services and retries > 0:
        print(f"No service found, retrying... {retries} attempts remaining.")
        time.sleep(5)  # Wait for a few seconds before retrying
        existing_services = model_deployer.find_model_server(
            pipeline_name=pipeline_name,
            pipeline_step_name=step_name,
        )
        retries -= 1
        
    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{step_name} step in the {pipeline_name} "
            f"pipeline is currently "
            f"running."
        )

    print(f"Prediction service loaded: {existing_services[0].prediction_url}")
    return existing_services[0]
