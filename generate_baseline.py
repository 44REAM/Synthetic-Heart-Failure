
# stdlib
import yaml
import sys
import warnings
import numpy as np
# synthcity absolute
import synthcity.logger as log
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader, SurvivalAnalysisDataLoader

log.add(sink=sys.stderr, level="DEBUG")
warnings.filterwarnings("ignore")

import pandas as pd
import config as common_config
from preprocess.impute import impute_data
from preprocess.min_max import var
from utils import *

real_df = pd.read_parquet(common_config.BASELINE_COMBINE_FILE)
real_df = real_df.drop(columns=['admit'])

with open(common_config.COLUMNS_CLASS, 'rb') as yaml_file:
    column_class = yaml.load(yaml_file, Loader=yaml.FullLoader)

real_train_idx = np.load(common_config.INDEX_TRAIN)
real_val_idx = np.load(common_config.INDEX_VAL)
test_idx = np.load(common_config.INDEX_TEST)

impute_method = 'add_missing'  
impute_feature_only = True  # True, False
impute_test_set = True  # True, False
real_df = impute_data(real_df, real_train_idx, real_val_idx, test_idx, method=impute_method, impute_feature_only=impute_feature_only, impute_test_set=impute_test_set)
df_train = real_df.loc[np.concatenate([real_train_idx, real_val_idx])]
# Note: preprocessing data with OneHotEncoder or StandardScaler is not needed or recommended. Synthcity handles feature encoding and standardization internally.


def drop_rows(df):
    # Create a boolean mask for rows to keep
    mask = pd.Series([True] * len(df), index=df.index)

    for col, (min_val, max_val) in var.items():
        if col in df.columns:
            # Only apply condition to non-NaN values
            if min_val is not None:
                mask &= df[col].ge(min_val) | df[col].isna()
            if max_val is not None and max_val != np.inf:
                mask &= df[col].le(max_val) | df[col].isna()
        else:
            print(f"Column '{col}' not found in DataFrame.")
    
    mask &= (df['SBP'] > df['DBP']) | df['DBP'].isna()

    
    return df[mask]

def generate_data_synthcity(syn_model, count, max_attempts=2):
    start_count = 100
    syn_count = start_count
    invalid = 0
    data_gen = start_count

    synthetic_data = syn_model.generate(count=start_count).dataframe()
    synthetic_data  = add_missing(synthetic_data)
    trial = 0
    while syn_count < count and trial < max_attempts:

        tmp = syn_model.generate(count=count).dataframe()
        tmp= add_missing(tmp)
        data_gen+=len(tmp)
        synthetic_data = pd.concat([synthetic_data, tmp])
        synthetic_data = synthetic_data.reset_index(drop=True)

        synthetic_data = drop_rows(synthetic_data).reset_index(drop=True)
  
        syn_count = len(synthetic_data)
        invalid += count-syn_count

        synthetic_data = synthetic_data[:count]
        trial += 1
        if trial > max_attempts:
            raise ValueError(f"Failed to generate enough valid data after {max_attempts} attempts. Generated {syn_count} rows, expected {count}.")

    print(f'percent invalid = {(data_gen-count)/data_gen*100}')
    synthetic_data = add_missing(synthetic_data).reset_index(drop=True)
    return synthetic_data

def generate_synthcity(model, real_df, loader, column_class, baseline_path, train_path, val_path, test_path):


    syn_model = Plugins().get(model)

    syn_model.fit(loader)

    count = len(real_df)
    synthetic_data = generate_data_synthcity(syn_model, count)


    synthetic_data['ENC_HN'] = list(range(len(synthetic_data)))
    synthetic_data.columns = [col + "_1" if col in column_class['discrete'] else col for col in synthetic_data.columns]

    synthetic_data.to_csv(baseline_path, index = False)

    split_data(
            synthetic_data['ENC_HN'].tolist(), 
            synthetic_data['dead_1'],
            train_path,
            val_path,
            test_path)
    

models = {
    'adsgan': {
        'baseline': common_config.ADGAN_SYNTHETIC_BASELINE + impute_method + '.csv',
        'index_train': common_config.ADGAN_INDEX_TRAIN + impute_method + '.npy',
        'index_val': common_config.ADGAN_INDEX_VAL + impute_method + '.npy',
        'index_test': common_config.ADGAN_INDEX_TEST + impute_method + '.npy',
        'library': 'synthcity',
        'type': 'generic'
    },
    'survival_gan':{
        'baseline': common_config.SURVADGAN_SYNTHETIC_BASELINE + impute_method + '.csv',
        'index_train': common_config.SURVADGAN_INDEX_TRAIN + impute_method + '.npy',
        'index_val': common_config.SURVADGAN_INDEX_VAL + impute_method + '.npy',
        'index_test': common_config.SURVADGAN_INDEX_TEST + impute_method + '.npy',
        'library': 'synthcity',
        'type': 'survival'
    },
    'dpgan':{
        'baseline': common_config.DPGAN_SYNTH_SYNTHETIC_BASELINE + impute_method + '.csv',
        'index_train': common_config.DPGAN_SYNTH_INDEX_TRAIN + impute_method + '.npy',
        'index_val': common_config.DPGAN_SYNTH_INDEX_VAL + impute_method + '.npy',
        'index_test': common_config.DPGAN_SYNTH_INDEX_TEST + impute_method + '.npy',
        'library': 'synthcity',
        'type': 'generic'
    },
    'tvae':{
        'baseline': common_config.TVAE_SYNTHETIC_BASELINE + impute_method + '.csv',
        'index_train': common_config.TVAE_INDEX_TRAIN + impute_method + '.npy',
        'index_val': common_config.TVAE_INDEX_VAL + impute_method + '.npy',
        'index_test': common_config.TVAE_INDEX_TEST + impute_method + '.npy',
        'library': 'synthcity',
        'type': 'generic'
    },
    'survae':{
        'baseline': common_config.SURVTVAE_SYNTHETIC_BASELINE + impute_method + '.csv',
        'index_train': common_config.SURVTVAE_INDEX_TRAIN + impute_method + '.npy',
        'index_val': common_config.SURVTVAE_INDEX_VAL + impute_method + '.npy',
        'index_test': common_config.SURVTVAE_INDEX_TEST + impute_method + '.npy',
        'library': 'synthcity',
        'type': 'generic'
    },
    'ctgan':{
        'baseline': common_config.CGAN_SYNTHETIC_BASELINE + impute_method + '.csv',
        'index_train': common_config.CGAN_INDEX_TRAIN + impute_method + '.npy',
        'index_val': common_config.CGAN_INDEX_VAL + impute_method + '.npy',
        'index_test': common_config.CGAN_INDEX_TEST + impute_method + '.npy',
        'library': 'synthcity',
        'type': 'generic'
    },
    'ddpm':{
        'baseline': common_config.DDPM_SYNTHETIC_BASELINE + impute_method + '.csv',
        'index_train': common_config.DDPM_INDEX_TRAIN + impute_method + '.npy',
        'index_val': common_config.DDPM_INDEX_VAL + impute_method + '.npy',
        'index_test': common_config.DDPM_INDEX_TEST + impute_method + '.npy',
        'library': 'synthcity',
        'type': 'generic'
    },
    'nflow':{
        'baseline': common_config.NFLOW_SYNTHETIC_BASELINE + impute_method + '.csv',
        'index_train': common_config.NFLOW_INDEX_TRAIN + impute_method + '.npy',
        'index_val': common_config.NFLOW_INDEX_VAL + impute_method + '.npy',
        'index_test': common_config.NFLOW_INDEX_TEST + impute_method + '.npy',
        'library': 'synthcity',
        'type': 'generic'
    },
}

models_use = [ 'ddpm', 'adsgan','tvae', 'ctgan', 'survival_gan', 'nflow']
models_use = [ 'adsgan']
for model in models_use:
    print(f'Generating {model} baseline...')
    if models[model]['type'] == 'survival':
        loader = SurvivalAnalysisDataLoader(
            df_train,
            target_column="dead",
            time_to_event_column="Days",
            train_size = 0.875
        )
    else:
        loader = GenericDataLoader(
            df_train,
            train_size = 0.875
        )
    try:
        if models[model]['library'] == 'synthcity':
            generate_synthcity(model, real_df, loader, column_class, 
                    models[model]['baseline'], 
                    models[model]['index_train'], 
                    models[model]['index_val'],
                    models[model]['index_test'])
        else:
            raise ValueError("not support")
    except Exception as e:
        print(f"Error generating baseline for {model}: {e}")
        continue