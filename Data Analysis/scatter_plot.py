import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def DataPreprocessing(dataframe : pd.DataFrame):
    df = dataframe.dropna()
    numbered_df = df.select_dtypes(include="number")
    return numbered_df

def cleanSortCorrelation(matrix: pd.DataFrame):
    new_df = matrix.abs().unstack().sort_values(ascending=False)
    feat1_names = new_df.index.get_level_values(0)
    feat2_names = new_df.index.get_level_values(1)
    sorted_pairs = new_df[feat1_names != feat2_names]
    sorted_pairs = sorted_pairs[::2]
    return sorted_pairs

def scatterPlot(df: pd.DataFrame, xFeatureName: str, yFeatureName: str):
    plt.scatter(df[xFeatureName], df[yFeatureName])
    plt.xlabel(xlabel=xFeatureName)
    plt.ylabel(ylabel=yFeatureName)
    plt.title(f"pair plot for {xFeatureName} and {yFeatureName}")
    plt.show()

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df = pd.read_csv("Data CSV/dataset_train.csv")
    df = df.set_index('Index')
    new_df = DataPreprocessing(df)
    corr_matrix = cleanSortCorrelation(new_df.corr())
    print(corr_matrix)
    scatterPlot(new_df, corr_matrix.head(1).index[0][0], corr_matrix.head(1).index[0][1])
