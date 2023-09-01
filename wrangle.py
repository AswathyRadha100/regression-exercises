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

# +
############################ IMPORTS #########################

#standard ds imports
import pandas as pd
import numpy as np
import os

#visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

#import custom modules
from env import user, host, password

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

############################## AQUIRE ZILLOW FUNCTION ##############################

def get_zillow_data():
    '''
    This function acquires zillow.csv it is available
    otherwise, it makes the SQL connection and uses the query provided
    to read in the dataframe from SQL.
    If they csv is not present, it will write one.
    '''
    filename = "zillow.csv"

    if os.path.isfile(filename):

        return pd.read_csv(filename, index_col=0)
    else:
        # Create the url
        url = env.get_db_url('zillow')
        
        sql_query = '''
            SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
            FROM properties_2017
            WHERE propertylandusetypeid = 261'''

        # Read the SQL query into a dataframe
        df = pd.read_sql(zillow_query, url)

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df

############################## AQUIRE ZILLOW FUNCTION ##############################

def acquire_zillow():
    '''
    This function checks to see if zillow.csv already exists, 
    if it does not, one is created
    '''
    #check to see if telco_churn.csv already exist
    if os.path.isfile('zillow.csv'):
        df = pd.read_csv('zillow.csv', index_col=0)
    
    else:

        #creates new csv if one does not already exist
        df = get_zillow_data()
        df.to_csv('zillow.csv')

    return df

############################## PREPARE ZILLOW FUNCTION ##############################

def prep_zillow(df):
    '''
    This function takes in a dataframe
    renames the columns and drops nulls values
    Additionally it changes datatypes for appropriate columns
    and renames fips to actual county names.
    Then returns a cleaned dataframe
    '''
    df = df.rename(columns = {'bedroomcnt':'bedrooms',
                     'bathroomcnt':'bathrooms',
                     'calculatedfinishedsquarefeet':'area',
                     'taxvaluedollarcnt':'taxvalue',
                     'fips':'county'})
    
    df = df.dropna()
    
    make_ints = ['bedrooms','area','taxvalue','yearbuilt']

    for col in make_ints:
        df[col] = df[col].astype(int)
        
    df.county = df.county.map({6037:'LA',6059:'Orange',6111:'Ventura'})
    
    return df

############################ WRANGLE ZILLOW FUNCTION ############################

def wrangle_zillow():
    '''
    This function acquires and prepares our Zillow data
    and returns the clean dataframe
    '''
    df = prep_zillow(acquire_zillow())
    return df
