import pandas as pd
import sys
from utils import my_count, my_mean, my_max, my_min, my_percentile, my_std

def checkArgsLength(numberOfArgs:int):
    if len(sys.argv) != numberOfArgs:
        raise Exception("didn't recieve the right number of argvs")
        
def checkFileExtension(fileName: str, extension: str):
    extLength = len(extension)
    if(fileName[-extLength:] == extension):
        return True
    else:
        raise Exception("bad file extention")
    
def describe(dataframe: pd.DataFrame):
    num_df = dataframe.select_dtypes(include="number")
    filler = {}
    for column in num_df.columns:
        column_serie = num_df[column]
        temp_filler = {
            'count': my_count(column_serie),
            'mean': my_mean(column_serie),
            'std': my_std(column_serie),
            'min': my_min(column_serie),
            '25%' : my_percentile(column_serie, 0.25),
            '50%' : my_percentile(column_serie, 0.50),
            '75%' : my_percentile(column_serie, 0.75),
            'max': my_max(column_serie)
        }
        filler[column] = temp_filler
    
    return pd.DataFrame(filler)
    
def main():
    try:
        checkArgsLength(2)
        filePath = sys.argv[1]
        checkFileExtension(filePath, ".csv")
        df = pd.read_csv(filePath)
        print(describe(df))
        # print(df.describe())

    except Exception as e:
        print(e)
        exit(1)

if __name__ == "__main__":
    main()
