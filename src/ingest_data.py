import os
import pandas as pd
from abc import ABC, abstractmethod
import zipfile

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        pass

class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        if not file_path.endswith(".zip"):
            raise ValueError("Given file is not a zip file")
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("Unzipped_data")
        extracted_files = os.listdir("Unzipped_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]
        if len(csv_files) == 0:
            raise ValueError("No CSV file found in the zip")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found, what to do?")
        csv_file_path = os.path.join("Unzipped_data", csv_files[0])
        df = pd.read_csv(csv_file_path)
        return df

class MainDataIngestor:
    @staticmethod
    def get_data_ingestor(file_path: str) -> pd.DataFrame:
        if file_path.endswith("zip"):
            ingestor = ZipDataIngestor()
            return ingestor.ingest(file_path)  # Ensure the DataFrame is returned
        else:
            raise ValueError("Unsupported file type")

if __name__ == "__main__":
    pass
