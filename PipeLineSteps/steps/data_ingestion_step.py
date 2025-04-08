import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.ingest_data import MainDataIngestor
from zenml import step

@step
def data_ingestion_step(file_path:str) -> pd.DataFrame:
    ingestor = MainDataIngestor()
    df = ingestor.get_data_ingestor(file_path)
    return df