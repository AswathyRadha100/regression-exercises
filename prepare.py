# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd


# Perform one-hot encoding for the "county" column
def zillow_county_data_as_int(df):
    '''
    Perform data preparation for Zillow county data.

    Args:
        df (DataFrame): The input DataFrame containing Zillow county data.

    Returns:
        DataFrame: The processed DataFrame with one-hot encoding and boolean columns converted to integers.
    '''
    
    # Perform one-hot encoding for the "county" column with 'LA' included
    df = pd.get_dummies(df, columns=['county'], prefix='county')

    # Convert boolean columns to integers (0 or 1)
    df['county_LA'] = df['county_LA'].astype(int)
    df['county_Orange'] = df['county_Orange'].astype(int)
    df['county_Ventura'] = df['county_Ventura'].astype(int)

    return df


def min_max_scaler(df, cols):
    """
    Scale specified columns in a DataFrame using MinMaxScaler.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    cols (list): List of column names to scale.
    
    Returns:
    DataFrame: A new DataFrame with specified columns scaled.
    """
    from sklearn.preprocessing import MinMaxScaler

    # Create a MinMaxScaler object
    scaler = MinMaxScaler()

    # Fit the scaler to the specified columns and transform them
    df[cols] = scaler.fit_transform(df[cols])

    return df



# +
from sklearn.preprocessing import StandardScaler

def standard_scaler(df, cols):
    """
    Scales the specified columns in a dataframe using StandardScaler.
    
    Args:
        df (pd.DataFrame): The dataframe containing the columns to be scaled.
        cols (list): A list of column names to be scaled.
        
    Returns:
        pd.DataFrame: The dataframe with the specified columns scaled using StandardScaler.
    """
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df



# +
# cols=None means- if a list of column names to scale is not given
# when calling the function, it will assume that we want to scale all numeric columns in the DataFrame

def robust_scaler(df, cols=None):
    """
    Apply RobustScaler to specified or all numeric columns in a pandas DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    cols (list, optional): List of column names to scale. If None, scales all numeric columns.

    Returns:
    pandas.DataFrame: The input DataFrame with scaled numeric columns.
    
    How to call:
    robust_scale_columns = ['column1', 'column2', 'column3']
    scaled_df = robust_Scaler(df, cols=robust_scale_columns)

    """
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()

    if cols is None:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[cols] = scaler.fit_transform(df[cols])

    return df

# -




