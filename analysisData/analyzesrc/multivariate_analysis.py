from abc import ABC, abstractmethod
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class MultivariateAnalysis(ABC):
    def analyze(self,df:pd.DataFrame):
        self.generate_correation_matrix(df)
        self.generate_pair_plot(df)
    @abstractmethod
    def generate_correation_matrix(self,df:pd.DataFrame):
        pass
    @abstractmethod
    def generate_pair_plot(self,df:pd.DataFrame):
        pass
class SimpleMultivariateAnalysis(MultivariateAnalysis):
    def generate_correation_matrix(self,df:pd.DataFrame):
        print("\nCorrelation Matrix:")
        corr = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Matrix")
        plt.show()

    def generate_pair_plot(self,df:pd.DataFrame):
        print("\nPair Plot:")
        sns.pairplot(df)
        plt.title("Pair Plot")
        plt.show()
 
if __name__ == "__main__":
    pass