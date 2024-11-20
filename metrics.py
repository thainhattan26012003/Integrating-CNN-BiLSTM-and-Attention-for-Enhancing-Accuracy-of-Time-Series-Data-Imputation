import numpy as np
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

def evaluate(lst_y_truth, lst_y_pred):
    return {
        'Similarity' : similarity(lst_y_truth, lst_y_pred),
        'MAE' : MAE(lst_y_truth, lst_y_pred),
        'RMSE' : RMSE(lst_y_truth, lst_y_pred),
        'Fractional Bias' : FB(lst_y_truth, lst_y_pred),
        'Fractional Standard Deviation' : FSD(lst_y_truth, lst_y_pred)
    }
    
def average_performance_metrics(list_of_dicts):
    totals = defaultdict(float)
    counts = defaultdict(int)
    
    for d in list_of_dicts:
        for key, value in d.items():
            totals[key] += value
            counts[key] += 1

    averages = {key: totals[key] / counts[key] for key in totals}
    return averages