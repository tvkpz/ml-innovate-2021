import logging
import argparse
import os
import sys
import json
import warnings
import random

#import boto3
#import io
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import pickle as pkl
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
#from shift_detect import rulsif
#import xgboost as xgb

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    #from prettytable import PrettyTable
    #import autogluon as ag
    #from autogluon import TabularPrediction as task
    #from autogluon.task.tabular_prediction import TabularDataset
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action="ignore", category=DataConversionWarning)

def fix_data_imbalance(df_training):

    y = np.array(df_training['Target'].values.tolist())
    X = np.array(df_training.drop(['Target'], axis=1).values.tolist())

    #oversample = SMOTE(sampling_strategy=0.6, k_neighbors=7)
    oversample = SMOTE()
    #undersample = RandomUnderSampler(sampling_strategy=0.3)
    #X_temp, y_temp = oversample.fit_resample(X, y)
    X_resampled, y_resampled = oversample.fit_resample(X, y)
    #X_resampled, y_resampled = undersample.fit_resample(X_temp, y_temp)
    
    columns=list(df_training.columns)
    columns.remove('Target')
    f_balanced = pd.DataFrame(X_resampled.tolist(), columns=columns)
    target_df = pd.DataFrame(y_resampled.tolist(), columns=['Target'])
    df_training_balanced = pd.concat([f_balanced, target_df], axis=1)
    
    df_training_balanced = pd.concat([df_training_balanced['Target'], df_training_balanced.drop(['Target'], axis=1)], axis=1)
    
    return df_training_balanced

def fix_class_imbalance(df_training):

    y = np.array(df_training['Gender'].values.tolist())
    X = np.array(df_training.drop(['Gender'], axis=1).values.tolist())

    oversample = SMOTE(sampling_strategy=0.7) #, k_neighbors=7)
    #oversample = SMOTE()
    undersample = RandomUnderSampler(sampling_strategy=0.8)
    X_temp, y_temp = oversample.fit_resample(X, y)
    #X_resampled, y_resampled = oversample.fit_resample(X, y)
    X_resampled, y_resampled = undersample.fit_resample(X_temp, y_temp)
    
    columns=list(df_training.columns)
    columns.remove('Gender')
    f_balanced = pd.DataFrame(X_resampled.tolist(), columns=columns)
    target_df = pd.DataFrame(y_resampled.tolist(), columns=['Gender'])
    df_training_balanced = pd.concat([f_balanced, target_df], axis=1)
    
    df_training_balanced = pd.concat([df_training_balanced['Target'], df_training_balanced.drop(['Target'], axis=1)], axis=1)
    
    return df_training_balanced


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.2)
    parser.add_argument("--etl-pipeline", type=str, default='')
    args = parser.parse_args()

    print("Entered Processing....")
    print("Received arguments {}".format(args))
    
    return args
    
def main():
    
    args = parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_path="/opt/ml/processing/input/data"
    input_files = [ os.path.join(file) for file in os.listdir(input_path) ]
    print('filename:', input_files)
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format("/opt/ml/processing/input"))

    raw_data = [ pd.read_csv(os.path.join(input_path,file)) for file in input_files ]
    if len(raw_data) == 1:
        concat_data = raw_data[0]
    else:
        concat_data = pd.concat(raw_data, axis=1)
    
    # Preprocessing training data
    #dtrain = fix_data_imbalance(concat_data)
    print('Before Target: ',Counter(dtrain['Target']))
    print('Before Gender: ',Counter(dtrain['Gender']))
    dtrain = fix_class_imbalance(concat_data)
    
    print('After Target: ',Counter(dtrain['Target']))
    print('After Gender: ',Counter(dtrain['Gender']))
    
    train_dir="/opt/ml/processing/train"
    train_output_path = os.path.join(train_dir, "train_balanced.csv")
    
    dtrain.to_csv(train_output_path, index=False)
    print('Training files created.....')

if __name__ == '__main__':
    main()
