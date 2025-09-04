from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import config as common_config
from preprocess.impute import add_missing

def split_data(index, stratify, train_path, val_path, test_path):

    # First, split into train_val and test indices (80:20 split)
    train_val_index, test_index = train_test_split(
        index, test_size=0.2, random_state=common_config.RANDOM_SEED, stratify = stratify)

    # Then split the train_val indices into train and validation (80:10 split)
    train_index, val_index = train_test_split(
        train_val_index, test_size=0.125, random_state=common_config.RANDOM_SEED, stratify = stratify[train_val_index])
    
    np.save(train_path, train_index)
    np.save(val_path, val_index)
    np.save(test_path, test_index)

def columns_to_int(df, columns):
    for col in columns:
        df[col] = df[col].astype(np.int64)
    return df
