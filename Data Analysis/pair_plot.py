import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def pairPlot(df: pd.DataFrame):
    sns.pairplot(df, hue='Hogwarts House')
    plt.savefig("my_huge_pairplot.png")

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df = pd.read_csv("Data CSV/dataset_train.csv")
    df = df.set_index('Index')
    df = df.dropna()

    pairPlot(df)
