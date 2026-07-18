'''
    This contains method for data preprocessing.
    Following data preprocessing techniques that are implemented
    * One-Hot encoding
'''
import numpy as np
from typing import Any

def OneHotEncoder(data : np.ndarray, dtype : Any = np.float32):
    # Categories data.
    Unique_categories = np.unique(data)
        
    # Num_Samples - Number of data (Rows)
    # Num_Categories - Number of feature columns (Columns)
    Num_Samples = len(data)
    Num_Categories = len(Unique_categories)

    # Initializing matrix with all values 0
    one_hot_matrix = np.zeros((Num_Samples, Num_Categories), dtype=dtype)

    # Setting 1 where the data matches the categories
    for index, category in enumerate(Unique_categories):
        one_hot_matrix[data == category, index] = 1

    return one_hot_matrix
