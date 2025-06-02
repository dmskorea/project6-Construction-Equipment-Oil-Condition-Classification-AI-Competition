import joblib
import numpy as np
import pandas as pd
import gc
import time
import os
import sys
from contextlib import contextmanager
from lightgbm import LGBMClassifier
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from skopt import BayesSearchCV 
from optbinning import OptimalBinning
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_rows', 500)

from DMS_202211.seed_everything import seed_everything

def make_group_features(train, test, numeric_features, group):
    
    print('group_feature:',group)

    for numeric_var in numeric_features:

        train = train.copy()
        test = test.copy()

        a1 = train.groupby([group])[numeric_var].median().to_dict()

        train[numeric_var+'_'+group+'_'+'median'] = train[group].map(a1)
        test[numeric_var+'_'+group+'_'+'median'] = test[group].map(a1)

        train[numeric_var+'_minus_'+numeric_var+'_'+group+'_'+'median'] = train[numeric_var] - train[numeric_var+'_'+group+'_'+'median']
        test[numeric_var+'_minus_'+numeric_var+'_'+group+'_'+'median'] = test[numeric_var] - test[numeric_var+'_'+group+'_'+'median']
        
        train[numeric_var+'_divide_'+numeric_var+'_'+group+'_'+'median'] = np.log(train[numeric_var]+0.00001) / np.log(train[numeric_var+'_'+group+'_'+'median']+0.00001) 
        test[numeric_var+'_divide_'+numeric_var+'_'+group+'_'+'median'] = np.log(test[numeric_var]+0.00001) / np.log(test[numeric_var+'_'+group+'_'+'median']+0.00001)


        a1 = train.groupby([group])[numeric_var].max().to_dict()

        train[numeric_var+'_'+group+'_'+'max'] = train[group].map(a1)
        test[numeric_var+'_'+group+'_'+'max'] = test[group].map(a1)

        train[numeric_var+'_minus_'+numeric_var+'_'+group+'_'+'max'] = train[numeric_var] - train[numeric_var+'_'+group+'_'+'max']
        test[numeric_var+'_minus_'+numeric_var+'_'+group+'_'+'max'] = test[numeric_var] - test[numeric_var+'_'+group+'_'+'max']

        train[numeric_var+'_divide_'+numeric_var+'_'+group+'_'+'max'] = np.log(train[numeric_var]+0.00001) / np.log(train[numeric_var+'_'+group+'_'+'max']+0.00001)
        test[numeric_var+'_divide_'+numeric_var+'_'+group+'_'+'max'] = np.log(test[numeric_var]+0.00001) / np.log(test[numeric_var+'_'+group+'_'+'max']+0.00001)


        a1 = train.groupby([group])[numeric_var].min().to_dict()

        train[numeric_var+'_'+group+'_'+'min'] = train[group].map(a1)
        test[numeric_var+'_'+group+'_'+'min'] = test[group].map(a1)

        train[numeric_var+'_minus_'+numeric_var+'_'+group+'_'+'min'] = train[numeric_var] - train[numeric_var+'_'+group+'_'+'min']
        test[numeric_var+'_minus_'+numeric_var+'_'+group+'_'+'min'] = test[numeric_var] - test[numeric_var+'_'+group+'_'+'min']

        train[numeric_var+'_divide_'+numeric_var+'_'+group+'_'+'min'] = np.log(train[numeric_var]+0.00001) / np.log(train[numeric_var+'_'+group+'_'+'min']+0.00001) 
        test[numeric_var+'_divide_'+numeric_var+'_'+group+'_'+'min'] = np.log(test[numeric_var]+0.00001) / np.log(test[numeric_var+'_'+group+'_'+'min']+0.00001) 


        a1 = train.groupby([group])[numeric_var].sum().to_dict()

        train[numeric_var+'_'+group+'_'+'sum'] = train[group].map(a1)
        test[numeric_var+'_'+group+'_'+'sum'] = test[group].map(a1)

        train[numeric_var+'_minus_'+numeric_var+'_'+group+'_'+'sum'] = train[numeric_var] - train[numeric_var+'_'+group+'_'+'sum']
        test[numeric_var+'_minus_'+numeric_var+'_'+group+'_'+'sum'] = test[numeric_var] - test[numeric_var+'_'+group+'_'+'sum']

        train[numeric_var+'_divide_'+numeric_var+'_'+group+'_'+'sum'] = np.log(train[numeric_var]+0.00001) / np.log(train[numeric_var+'_'+group+'_'+'sum']+0.00001)
        test[numeric_var+'_divide_'+numeric_var+'_'+group+'_'+'sum'] = np.log(test[numeric_var]+0.00001) / np.log(test[numeric_var+'_'+group+'_'+'sum']+0.00001)
            
    return train, test
