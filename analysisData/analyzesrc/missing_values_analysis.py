from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class MissingValuesAnalysis(ABC):
    def analyze(self, df: pd.DataFrame):
        self.identify_missing_value(df)
        self.visualize_missing_value(df)
    
    @abstractmethod
    def identify_missing_value(self, df: pd.DataFrame):
        pass
    
    @abstractmethod
    def visualize_missing_value(self, df: pd.DataFrame):
        pass


class SimpleMissingValuesAnalysis(MissingValuesAnalysis):
    def identify_missing_value(self, df: pd.DataFrame):
        print("\nMissing values in the data:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])
    
    def visualize_missing_value(self, df: pd.DataFrame):
        print("\nVisualizing a heat map:")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Value Heat Map")
        plt.show()

# Example usage
if __name__ == "__main__":
    pass