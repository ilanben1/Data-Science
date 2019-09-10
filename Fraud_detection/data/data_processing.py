import os
import pandas as pd
import numpy as np # linear algebra
import seaborn as sns
import matplotlib.pyplot as plt
from os import chdir


def load_training_data(path):
    df_train_transaction = pd.read_csv(path+'/train_transaction.csv')
    df_train_identity = pd.read_csv(path+'/train_identity.csv')

    return pd.merge(df_train_identity, df_train_transaction, on='TransactionID', how ='left')


def find_categorical_numerical_vars(raw_training_data):
    """
    identify categories of variables
    :param raw_training_data:
    :return: na_colums,
    """
    na_columns = raw_training_data.isna().sum() # numbers of na by features
    data_columns = raw_training_data.columns
    numeric_cols = raw_training_data._get_numeric_data().columns # get numeric variables

    categorical_cols = list(set(data_columns) - set(numeric_cols))

    return na_columns, data_columns, numeric_cols, categorical_cols


def transform_na(raw_training_data, categorical_cols, numeric_cols):
    raw_training_data[categorical_cols] = raw_training_data[categorical_cols].replace({np.nan: 'missing'})
    raw_training_data[numeric_cols] = raw_training_data[numeric_cols].replace({np.nan: -1})
    return raw_training_data


def correlation_plot(numeric_cols,raw_training_data):
    variables = list(numeric_cols)
    variables.remove('isFraud')
    correlation_matrix = raw_training_data.loc[:, variables].corr().abs()
    plt.figure(figsize=(20, 20))
    heat = sns.heatmap(data=correlation_matrix)
    plt.title('Heatmap of Correlation')

