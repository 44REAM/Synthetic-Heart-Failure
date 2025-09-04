import os

COMBINE_FILE='data/data_combine.csv'

BASELINE_COMBINE_FILE = 'data/baseline_data_combine.parquet'

COLUMNS_CLASS = 'data/columns_class.yaml'


SURVADGAN_SYNTHETIC_BASELINE = 'data/surv_adgan/data_baseline'
SURVADGAN_INDEX_TRAIN = 'data/surv_adgan/index_train'
SURVADGAN_INDEX_VAL = 'data/surv_adgan/index_val'
SURVADGAN_INDEX_TEST = 'data/surv_adgan/index_test'

CGAN_SYNTHETIC_BASELINE = 'data/cgan/data_baseline'
CGAN_INDEX_TRAIN = 'data/cgan/index_train'
CGAN_INDEX_VAL = 'data/cgan/index_val'
CGAN_INDEX_TEST = 'data/cgan/index_test'

DPGAN_SYNTH_SYNTHETIC_BASELINE = 'data/dpgan_synth/data_baseline'
DPGAN_SYNTH_INDEX_TRAIN = 'data/dpgan_synth/index_train'
DPGAN_SYNTH_INDEX_VAL = 'data/dpgan_synth/index_val'
DPGAN_SYNTH_INDEX_TEST = 'data/dpgan_synth/index_test'

NFLOW_SYNTHETIC_BASELINE = 'data/nflow/data_baseline'
NFLOW_INDEX_TRAIN = 'data/nflow/index_train'
NFLOW_INDEX_VAL = 'data/nflow/index_val'
NFLOW_INDEX_TEST = 'data/nflow/index_test'

ADGAN_SYNTHETIC_BASELINE = 'data/adgan/data_baseline'
ADGAN_INDEX_TRAIN = 'data/adgan/index_train'
ADGAN_INDEX_VAL = 'data/adgan/index_val'
ADGAN_INDEX_TEST = 'data/adgan/index_test'

SURVCGAN_SYNTHETIC_BASELINE = 'data/surv_cgan/data_baseline'
SURVCGAN_INDEX_TRAIN = 'data/surv_cgan/index_train'
SURVCGAN_INDEX_VAL = 'data/surv_cgan/index_val'
SURVCGAN_INDEX_TEST = 'data/surv_cgan/index_test'

TVAE_SYNTHETIC_BASELINE = 'data/tvae/data_baseline'
TVAE_INDEX_TRAIN = 'data/tvae/index_train'
TVAE_INDEX_VAL = 'data/tvae/index_val'
TVAE_INDEX_TEST = 'data/tvae/index_test'

DDPM_SYNTHETIC_BASELINE = 'data/ddpm/data_baseline'
DDPM_INDEX_TRAIN = 'data/ddpm/index_train'
DDPM_INDEX_VAL = 'data/ddpm/index_val'
DDPM_INDEX_TEST = 'data/ddpm/index_test'

SURVTVAE_SYNTHETIC_BASELINE = 'data/surv_tvae/data_baseline'
SURVTVAE_INDEX_TRAIN = 'data/surv_tvae/index_train'
SURVTVAE_INDEX_VAL = 'data/surv_tvae/index_val'
SURVTVAE_INDEX_TEST = 'data/surv_tvae/index_test'

INDEX_TRAIN = 'data/index_train.npy'
INDEX_VAL = 'data/index_val.npy'
INDEX_TEST = 'data/index_test.npy'
RANDOM_SEED = 42

file_paths = {

    'COMBINE_FILE': COMBINE_FILE,

    'COLUMNS_CLASS': COLUMNS_CLASS,

    'SURVADGAN_SYNTHETIC_BASELINE': SURVADGAN_SYNTHETIC_BASELINE,
    'ADGAN_SYNTHETIC_BASELINE': ADGAN_SYNTHETIC_BASELINE,

    'CGAN_SYNTHETIC_BASELINE': CGAN_SYNTHETIC_BASELINE,

    'SURVCGAN_SYNTHETIC_BASELINE': SURVCGAN_SYNTHETIC_BASELINE,

    'DPGAN_SYNTH_SYNTHETIC_BASELINE': DPGAN_SYNTH_SYNTHETIC_BASELINE,

    'DDPM_SYNTHETIC_BASELINE': DDPM_SYNTHETIC_BASELINE,

    'TVAE_SYNTHETIC_BASELINE': TVAE_SYNTHETIC_BASELINE,
    'TVAE_INDEX_TRAIN': TVAE_INDEX_TRAIN,
    'TVAE_INDEX_VAL': TVAE_INDEX_VAL,
    'TVAE_INDEX_TEST': TVAE_INDEX_TEST,

    'TVAE_SYNTHETIC_BASELINE': SURVTVAE_SYNTHETIC_BASELINE,

    'NFLOW_SYNTHETIC_BASELINE': NFLOW_SYNTHETIC_BASELINE,

    'INDEX_TRAIN': INDEX_TRAIN,
    'INDEX_VAL': INDEX_VAL,
    'INDEX_TEST': INDEX_TEST,



}

def create_required_directories():

    directories = set()

    # Extract directory paths
    for key in file_paths.keys():
        path = file_paths[key]
        directory = os.path.dirname(path)
        if directory:  # Only add non-empty directory paths
            directories.add(directory)
    
    # Create directories if they don't exist
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created (if it didn't exist): {directory}")