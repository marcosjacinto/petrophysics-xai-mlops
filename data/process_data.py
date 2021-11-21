from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer


def load_data() -> Tuple[np.ndarray]:
    """Retrieves the dataset containing the features and the target, both for
    the training and test dataset.

    Returns:
        Tuple[np.ndarray]: training dataset containing both features and the
        target; test_dataset containing only the features; and the target of
        the test dataset.
    """

    DATA_PATH = "data/raw/"

    train_dataset = pd.read_csv(DATA_PATH + "sonic_data_train.csv")
    test_dataset = pd.read_csv(DATA_PATH + "sonic_data_test.csv")
    y_test_results = pd.read_csv(DATA_PATH + "sonic_test_results.csv")

    return train_dataset, test_dataset, y_test_results


def remove_nans(train_dataset: pd.DataFrame) -> pd.DataFrame:
    """This function first replaces the null values (-999) with a NaN. Then it
    removes any row of the data which has a NaN.

    Args:
        train_dataset (pd.DataFrame): dataset containing the features and/or
        the target (DTC).

    Returns:
        pd.DataFrame: clean dataset after removing null values.
    """

    # Replaces -999 with NaNs (null values) and cleans it
    train_dataset = train_dataset.replace(-999, np.nan)
    train_dataset = train_dataset.dropna(how="any")
    return train_dataset


def clean_zden_cnc_dtc(train_dataset: pd.DataFrame) -> pd.DataFrame:
    """This function removes any data that falls outside of the upper and lower
    limits. We use the following values:

    Density lower limit = 1.75; Density upper limit = 3.00;
    Neutron lower limit = -0.2; Neutron upper limit = 1;
    Sonic lower limit = 40; Sonic upper limit = 160;

    Args:
        train_dataset (pd.DataFrame): dataset containing the features and/or
        the target (DTC).

    Returns:
        pd.DataFrame: clean dataset after removing rows which values outside
        the boundaries established.
    """

    # Establishes data limits
    density_lower_limit = 1.75
    density_upper_limit = 3

    neutron_lower_limit = -0.2
    neutron_upper_limit = 1

    sonic_lower_limit = 40
    sonic_upper_limit = 160

    # Cleans data outside of ZDEN, CNC and DTC limits
    train_dataset = train_dataset[
        (
            (train_dataset.ZDEN > density_lower_limit)
            & (train_dataset.ZDEN < density_upper_limit)
        )
        & (
            (train_dataset.CNC > neutron_lower_limit)
            & (train_dataset.CNC < neutron_upper_limit)
        )
        & (
            (train_dataset.DTC > sonic_lower_limit)
            & (train_dataset.DTC < sonic_upper_limit)
        )
    ]

    return train_dataset


def clean_gr_hrm(train_dataset):
    """This function removes any data that falls outside of the upper and lower
    limits. We use the following values:

    Gamma ray lower limit = 0;
    Gamma ray upper limit = 300;
    Medium Resistivity lower limit = 0;
    Medium Resistivity upper limit = 99.9 percentile of the data distribution;


    Args:
        train_dataset (pd.DataFrame): dataset containing the features and/or
        the target (DTC).

    Returns:
        pd.DataFrame: clean dataset after removing rows which values outside
        the boundaries established.
    """

    gamma_lower_limit = 0
    gamma_upper_limit = 300

    hrm_lower_limit = 0
    hrm_upper_limit = np.percentile(train_dataset.HRM, 99.9)

    # Cleans data outside of GR and HRM limits
    train_dataset = train_dataset[
        (
            (train_dataset.GR > gamma_lower_limit)
            & (train_dataset.GR < gamma_upper_limit)
        )
        & (
            (train_dataset.HRM > hrm_lower_limit)
            & (train_dataset.HRM < hrm_upper_limit)
        )
    ]

    return train_dataset


def power_transformation(
    train_dataset: pd.DataFrame, test_dataset: pd.DataFrame
) -> Tuple[np.ndarray]:
    """This function applies Yeo-Johnson's power transformation. This reduces
    data skewness as well as normalizes the data.

    Args:
        train_dataset (pd.DataFrame): training features.
        test_dataset (pd.DataFrame): test features.

    Returns:
        Tuple[np.ndarray]: training and test features, respectively, as
        np.ndarrays
    """

    yeo_johnson_transformer = PowerTransformer()

    x_train = yeo_johnson_transformer.fit_transform(train_dataset.iloc[:, :-2])
    x_test = yeo_johnson_transformer.transform(test_dataset)

    return x_train, x_test


if __name__ == "__main__":

    train_dataset, test_dataset, y_test_results = load_data()

    train_dataset = remove_nans(train_dataset)

    train_dataset = clean_zden_cnc_dtc(train_dataset)

    train_dataset = clean_gr_hrm(train_dataset)

    x_train, x_test = power_transformation(train_dataset, test_dataset)

    # Changes y arrays to numpy format
    y_train_dtc = train_dataset.loc[:, "DTC"].to_numpy()
    y_train_dts = train_dataset.loc[:, "DTS"].to_numpy()

    y_test_dtc = y_test_results.iloc[:, 0].to_numpy()
    y_test_dts = y_test_results.iloc[:, 1].to_numpy()

    # Export processed data to a proper folder
    EXPORT_PATH = "data/processed/"
    np.save(EXPORT_PATH + "x_train.npy", x_train)
    np.save(EXPORT_PATH + "x_test.npy", x_test)
    np.save(EXPORT_PATH + "y_train_dtc.npy", y_train_dtc)
    np.save(EXPORT_PATH + "y_train_dts.npy", y_train_dts)
    np.save(EXPORT_PATH + "y_test_dtc.npy", y_test_dtc)
    np.save(EXPORT_PATH + "y_test_dts.npy", y_test_dts)
