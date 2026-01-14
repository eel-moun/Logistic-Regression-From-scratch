import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

def dataPreprocessing(dataframe : pd.DataFrame):
    df = dataframe.set_index('Index')
    new_df = df[["Hogwarts House", "Herbology", "Ancient Runes", "Astronomy"]]
    new_df = new_df.dropna()
    scaler = StandardScaler()
    new_df[["Herbology", "Ancient Runes", "Astronomy"]] = scaler.fit_transform(new_df[["Herbology", "Ancient Runes", "Astronomy"]])
    houses_names = new_df["Hogwarts House"].unique()
    dataframe_dict = {}
    for house in houses_names:
        house_df = new_df.copy()
        house_df["Hogwarts House"] = (house_df["Hogwarts House"] == house).astype(int)
        dataframe_dict[house] = house_df

    return dataframe_dict

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))


def calculatePartialDerivative(z, x, y, m):
    return ((x * (sigmoid(z) - y)) / m)

def calculateZ(row, theta0, theta1, theta2, b):
    z = row.iloc[1] * theta0 + row.iloc[2] * theta1 + row.iloc[3] * theta2 + b
    return z

def gradientDescent(dataframe: pd.DataFrame):
    theta0 = 0
    theta1 = 0
    theta2 = 0
    b = 0
    m = 5
    learning_rate = 0.001
    epoch = 500
    for i in range(epoch):
        print("we are in epoch", i)
        print("current  theta:", theta0, theta1, theta2)
        tmp_theta0 = 0
        tmp_theta1 = 0
        tmp_theta2 = 0
        tmp_b = 0
        i = 0 
        for _ , row in dataframe.iterrows():
            z = calculateZ(row, theta0, theta1, theta2, b)
            tmp_theta0 += calculatePartialDerivative(z, row.iloc[1], row.iloc[0], m)
            tmp_theta1 += calculatePartialDerivative(z, row.iloc[2], row.iloc[0], m)
            tmp_theta2 += calculatePartialDerivative(z, row.iloc[3], row.iloc[0], m)
            tmp_b += calculatePartialDerivative(z, 1, row.iloc[0], m)
            i += 1
            if i % m == 0 and i != 0:
                theta0 = theta0 - (learning_rate * tmp_theta0)
                theta1 = theta1 - (learning_rate * tmp_theta1)
                theta2 = theta2 - (learning_rate * tmp_theta2)
                b = b - (learning_rate * tmp_b)

    weights = {}
    weights[dataframe.columns[1]] = theta0
    weights[dataframe.columns[2]] = theta1
    weights[dataframe.columns[3]] = theta2
    weights["bias"] = b

    return weights

def main():
    if len(sys.argv) == 2:
        fileName = sys.argv[1]
    else:
        print("Error: No csv fileName was given")
        exit(1)
    df = pd.read_csv(fileName)
    houses_df = dataPreprocessing(df)
    ravenClawWeights = gradientDescent(houses_df["Ravenclaw"])
    slytherinWeights = gradientDescent(houses_df["Slytherin"])
    gryffindorWeights = gradientDescent(houses_df["Gryffindor"])
    hufflepuffWeights = gradientDescent(houses_df["Hufflepuff"])
    housesWeights = {
        "Ravenclaw" : ravenClawWeights,
        "Slytherin" : slytherinWeights,
        "Gryffindor": gryffindorWeights,
        "Hufflepuff": hufflepuffWeights
    }

    with open('my_weights.json', 'w') as f:
        json.dump(housesWeights, f, indent=4)
    
    return

if __name__ == "__main__":
    main()