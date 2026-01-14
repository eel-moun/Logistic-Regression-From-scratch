import pandas as pd
import numpy as np

def find_mean(values):
    total = 0
    n = 0

    for value in values:
        if not np.isnan(value):
            total += value
            n += 1

    return total / n if n > 0 else float("NaN")


def find_std(values):

    mean = find_mean(values)
    total = 0
    n = 0

    for value in values:

        if not np.isnan(value):

            # Subtract the mean from the value
            find_std = value - mean

            # Square it
            Square_std = find_std * find_std

            # Sum it to average it in the next step
            total += Square_std

            n += 1
    
    if n > 1:
        # Average the squared deviations (variance)
        variance = total / (n - 1)
    else:
        return float("NaN")

    # Square root it to get out STD final value
    return variance ** 0.5 if n > 1 else float("NaN")


