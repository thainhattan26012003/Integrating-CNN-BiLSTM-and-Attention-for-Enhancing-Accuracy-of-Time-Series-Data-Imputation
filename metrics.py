import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import defaultdict

def similarity(value_lst_after, value_lst_before):
        T = len(value_lst_after)  # Number of missing values
        similarity_sum = 0

        for i in range(T):
            yi = value_lst_after[i]
            xi = value_lst_before[i]
            similarity_sum += 1 / (1 + abs(yi - xi) / (max(value_lst_before) - min(value_lst_before)))

        similarity = similarity_sum / T
        return similarity

def MAE(value_lst_missing, value_lst_after):
        return mean_absolute_error(value_lst_missing, value_lst_after)

def RMSE(value_lst_missing, value_lst_after):
    return np.sqrt(mean_squared_error(value_lst_missing, value_lst_after))

def FB(value_lst_missing, value_lst_after):
    return 2 * abs((np.mean(value_lst_after) - np.mean(value_lst_missing)) / (np.mean(value_lst_after) + np.mean(value_lst_missing)))

def FSD(value_lst_missing, value_lst_after):
    std_dev_Y = np.std(value_lst_after)
    std_dev_X = np.std(value_lst_missing)

    if std_dev_X == 0:
        return None
    
    FSD = 2 * abs((std_dev_Y - std_dev_X) / (std_dev_X + std_dev_Y))
    
    return FSD

def r_score(value_lst_missing, value_lst_after):

    correlation_matrix = np.corrcoef(value_lst_missing, value_lst_after)
    r_score = correlation_matrix[0, 1]
    return r_score

def nse(value_lst_missing, value_lst_after):

    value_lst_missing = np.array(value_lst_missing)
    value_lst_after = np.array(value_lst_after)

    numerator = np.sum((value_lst_missing - value_lst_after)**2)
    denominator = np.sum((value_lst_missing - np.mean(value_lst_missing))**2)

    nse = 1 - (numerator / denominator)
    
    return nse



# Function to calculate metrics and save results
def calculate_metrics(file_path, value_lst_after, model_name, nan_index, size_of_gap, model_dict):
    # Load original data with missing values
    df_before_missing = pd.read_csv(file_path)
    value_lst_missing = df_before_missing['Waterlevel'].values.tolist()[nan_index:nan_index+size_of_gap]

    # Calculate metrics
    similarity_score = similarity(value_lst_after, value_lst_missing)
    MAE_score = MAE(value_lst_missing, value_lst_after)
    RMSE_score = RMSE(value_lst_missing, value_lst_after)
    FSD_score = FSD(value_lst_missing, value_lst_after)
    NSE_score = nse(value_lst_missing, value_lst_after)
    
    model_lst = model_dict[model_name]

    # Add metrics to the model-specific lists (for tracking)
    model_lst['sim'].append(similarity_score)
    model_lst['mae'].append(MAE_score)
    model_lst['rmse'].append(RMSE_score)
    model_lst['fsd'].append(FSD_score)
    model_lst['nse'].append(NSE_score)

    # Output results
    print('\nOri_data:', value_lst_missing)
    print('\nvalue_data:', value_lst_after)
    print('\nSimilarity_score:', similarity_score)
    print('\nMean Absolute Error (MAE):', MAE_score)
    print('\nRoot Mean Squared Error (RMSE):', RMSE_score)
    print('\nFraction of Standard Deviation Score:', FSD_score)
    print('\nThe Nash Sutcliffe efficiency (NSE):', NSE_score)

    return similarity_score, MAE_score, RMSE_score, FSD_score, NSE_score
    
# Calculate mean of all run times for each metrics 
def calculate_mean_metrics(model_dict):
    mean_metrics = {}
    for model_name, metrics in model_dict.items():
        mean_metrics[model_name] = {key: sum(values) / len(values) if values else 0 for key, values in metrics.items()}
    return mean_metrics