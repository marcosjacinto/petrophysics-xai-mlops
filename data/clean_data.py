import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer


DATA_PATH = "data/raw/"

train_dataset = pd.read_csv(DATA_PATH + 'sonic_data_train.csv')
test_dataset = pd.read_csv(DATA_PATH + 'sonic_data_test.csv')
y_test_results = pd.read_csv(DATA_PATH + 'sonic_test_results.csv')

# Replaces -999 with NaNs (null values) and cleans it
train_dataset = train_dataset.replace(-999, np.nan)
train_dataset = train_dataset.dropna(how = 'any')

#Establishes data limits
density_lower_limit = 1.75
density_upper_limit = 3

neutron_lower_limit = -0.2
neutron_upper_limit = 1

sonic_lower_limit = 40
sonic_upper_limit = 160

gamma_lower_limit = 0
gamma_upper_limit = 300

hrm_lower_limit = 0
hrm_upper_limit = np.percentile(train_dataset.HRM, 99.9)

# Cleans data outside of ZDEN, CNC and DTC limits
train_dataset = train_dataset[
    ((train_dataset.ZDEN > density_lower_limit) 
    & (train_dataset.ZDEN < density_upper_limit))
    & ((train_dataset.CNC > neutron_lower_limit) 
    & (train_dataset.CNC < neutron_upper_limit))
    & ((train_dataset.DTC > sonic_lower_limit) 
    & (train_dataset.DTC < sonic_upper_limit))
]

# Cleans data outside of GR and HRM limits
train_dataset = train_dataset[
    ((train_dataset.GR > gamma_lower_limit) 
    & (train_dataset.GR < gamma_upper_limit))
    & ((train_dataset.HRM > hrm_lower_limit) 
    & (train_dataset.HRM < hrm_upper_limit))
]

yeo_johnson_transformer = PowerTransformer()

x_train = yeo_johnson_transformer.fit_transform(train_dataset.iloc[:, :-2])
x_test = yeo_johnson_transformer.transform(test_dataset)

y_train_dtc = train_dataset.loc[:, 'DTC'].to_numpy()
y_train_dts = train_dataset.loc[:, 'DTS'].to_numpy()

y_test_dtc = y_test_results.iloc[:, 0].to_numpy()
y_test_dts = y_test_results.iloc[:, 1].to_numpy()

EXPORT_PATH = 'data/processed/'

np.save(EXPORT_PATH + 'x_train.npy', x_train)
np.save(EXPORT_PATH + 'x_test.npy', x_test)
np.save(EXPORT_PATH + 'y_train_dtc.npy', y_train_dtc)
np.save(EXPORT_PATH + 'y_train_dts.npy', y_train_dts)
np.save(EXPORT_PATH + 'y_test_dtc.npy', y_test_dtc)
np.save(EXPORT_PATH + 'y_test_dts.npy', y_test_dts)