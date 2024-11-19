import random
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

def to_df(array):   # convert to dataframe (optional)
    X = [i[:-1] for i in array]
    y = [i[-1] for i in array]
    transpose = [list(x) for x in zip(*X)]

    dataframe = pd.DataFrame({f'Column{i+1}': lst for i, lst in enumerate(transpose)})
    dataframe['Target'] = y
    return dataframe

def transform_to_multivariate(dataframe, gap_size): # convert univaritate data to multivariate
    lst_multi_var = []
    for i in range(len(dataframe) - gap_size):
        row = dataframe[i : i+gap_size+1]
        lst_multi_var.append(row)
    return np.array(lst_multi_var)

def create_continuous_missing_values(dataframe, column_name, num_missing_values): # randomly create missing data follow by gap_size(num_missing_values)
    modified_df = dataframe.copy()
    random_index = random.randint(0, len(dataframe) - num_missing_values - 1)
    y_truth = modified_df.loc[random_index:random_index + num_missing_values - 1, column_name].values.copy()
    modified_df.loc[random_index:random_index + num_missing_values - 1, column_name] = np.nan
    return modified_df, y_truth

def check_cases_of_missing_data(df, r):
    start = 0
    while start < len(df):
        if pd.isna(df.iloc[start, 0]):
            end = start
            while end < len(df) and pd.isna(df.iloc[end, 0]):
                end += 1
            
            # Check if size of missing <= 3 then apply interpolation
            if end - start <= r:
                df.iloc[start:end] = df.iloc[start-1:end].interpolate(method='linear').iloc[1:]  # interpolate linearly

            start = end
        else:
            start += 1
            
    return df

def visualize_for_impute_real_missing_dataset(df, imputed_values, target_col):
    plt.figure(figsize=(12, 6))
    plt.plot(df[target_col].values.tolist(), label='Initial Value', linestyle='-', color='black') 
    plt.plot(imputed_values, label='Predicted Value', linestyle='-', color='orange') 
    plt.title('Wave Heights Over Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)
    plt.close()