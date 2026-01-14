import json
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import json

# def my_mean(column: pd.Series):
#     mean = sum(value for value in column if pd.notna(value)) / my_count(column)
#     return mean

def dataPreprocessing(test_df : pd.DataFrame):
    df = test_df.set_index('Index')
    new_df = df[["Herbology", "Ancient Runes", "Astronomy"]].copy()
    train_mean = new_df.mean()
    print(train_mean)
    new_df = new_df.fillna(train_mean)
    scaler = StandardScaler()
    scaler.fit(new_df)

    new_df[["Herbology", "Ancient Runes", "Astronomy"]] = scaler.transform(new_df)
    print(new_df)
    return new_df

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def calculateZ(row, theta0, theta1, theta2, b):
    z = row.iloc[0] * theta0 + row.iloc[1] * theta1 + row.iloc[2] * theta2 + b
    return z

def getPred(housesWeigths, row):
    housesPred = {}
    for houseName, weights in housesWeigths.items():
        z = calculateZ(row, weights["Herbology"], weights["Ancient Runes"], weights["Astronomy"], weights["bias"])
        sig = sigmoid(z)
        housesPred[houseName] = sig
    bestHouse = max(housesPred, key= housesPred.get)
    return bestHouse


def main():
    if len(sys.argv) == 3:
        fileName = sys.argv[1]
        weights = sys.argv[2]
    else:
        print("Error: No csv fileName was given")
        exit(1)
    df = pd.read_csv(fileName)
    new_df = dataPreprocessing(df)
    with open(weights, 'r') as f:
        loaded_weights = json.load(f)
    predictions = []
    for index , row in new_df.iterrows():
        prediction = getPred(loaded_weights, row)
        predictions.append(prediction)
    output_df = pd.DataFrame()
    output_df["Index"] = new_df.index
    output_df["Hogwarts House"] = predictions
    # true_df = pd.read_csv("./dataset_truth.csv")
    output_df.to_csv('house.csv', index=False)
    # print(accuracy_score(true_df["Hogwarts House"], output_df["Hogwarts House"]))
    return

if __name__ == "__main__":
    main()