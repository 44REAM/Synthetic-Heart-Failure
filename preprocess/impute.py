import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.impute import SimpleImputer
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import AddMissingIndicator
from hyperimpute.plugins.imputers import Imputers
from preprocess.min_max import var

def impute_data(df, train_index, val_index, test_index, method = 'median', impute_feature_only = True, impute_test_set = True):
    train_index = np.concatenate([train_index, val_index])

    if impute_feature_only:
        col_use = [col for col in df.columns if col not in ['dead', 'Days']]
    else:
        col_use = df.columns.to_list()
    #! Note: May change this, or impute using only median from training set
    if method == 'median':
        imputer = SimpleImputer(strategy='median')
        df.loc[train_index, col_use] = imputer.fit_transform(df.loc[train_index, col_use].copy())
        if impute_test_set:

            df.loc[test_index, col_use] = imputer.transform(df.loc[test_index, col_use].copy())
        return df

    elif method == 'add_missing':
        ami = AddMissingIndicator()
        df = ami.fit_transform(df)


        plugin = Imputers().get('mice')
        out = plugin.fit_transform(df.loc[train_index, col_use].copy())
        out.columns = df[col_use].columns
        out.index = df.loc[train_index].index
        df.loc[train_index, col_use] = out

        if impute_test_set:
            out = plugin.transform(df.loc[test_index, col_use].copy())
            out.columns = df[col_use].columns
            out.index = df.loc[test_index].index
            df.loc[test_index, col_use] = out

        for key in var:
            df[key] = df[key].clip(lower=var[key][0], upper=var[key][1])

        return df
    else:

        plugin = Imputers().get(method, random_state = 1561561) 
        out = plugin.fit_transform(df.loc[train_index, col_use].copy())
        out.columns = df[col_use].columns
        out.index = df.loc[train_index].index
        df.loc[train_index, col_use] = out

        if impute_test_set:

            out = plugin.transform(df.loc[test_index, col_use].copy())
            out.columns = df[col_use].columns
            out.index = df.loc[test_index].index
            df.loc[test_index, col_use] = out

        for key in var:
            df[key] = df[key].clip(lower=var[key][0], upper=var[key][1])

        return df

    
def add_missing(synthetic_data):
    for col in synthetic_data.columns:
        if col.endswith('_na'):
            target_col = col.replace('_na', '')
            if target_col in synthetic_data.columns:
                # Convert 0/1 to boolean mask
                mask = synthetic_data[col].astype(bool)
                synthetic_data.loc[mask, target_col] = np.nan  # or None

    syndf = synthetic_data[[col for col in synthetic_data.columns if "_na" not in col]]
    return syndf

