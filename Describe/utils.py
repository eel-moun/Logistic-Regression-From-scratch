import numpy as np
import pandas as pd

def my_count(column: pd.Series):
    return column.notna().sum()

def my_mean(column: pd.Series):
    mean = sum(value for value in column if pd.notna(value)) / my_count(column)
    return mean

def my_std(column: pd.Series):
    column_size = my_count(column)
    if(column_size < 2):
        return np.nan
    mean = my_mean(column)
    sum_squarred_diff = 0
    for value in column:
        if pd.notna(value):
            temp = value - mean
            squarred_diff =  temp**2
            sum_squarred_diff = sum_squarred_diff + squarred_diff
    variance = sum_squarred_diff / (column_size - 1)
    res = variance ** 0.5
    return res


def my_min(column: pd.Series):
    column_size = my_count(column)
    if(column_size):
        min = column[0]
        for value in column.values:
            if value < min:
                min = value
        return min
    return np.nan

def my_max(column: pd.Series):
    column_size = my_count(column)
    if(column_size):
        max =column[0]
        for value in column.values:
            if value > max:
                max = value
        return max
    return np.nan

def my_percentile(column: pd.Series, p: float):
    sorted_list = sorted([value for value in column if pd.notna(value)])
    if not sorted_list or (p > 1) or (p < 0):
        return np.nan
    pos = (len(sorted_list) - 1) * p
    lower = int(np.floor(pos))
    upper = int(np.ceil(pos))
    if lower == upper:
        return sorted_list[lower]
    return ((sorted_list[lower] * (upper - pos)) + (sorted_list[upper] * (pos - lower)))
    
    
        
