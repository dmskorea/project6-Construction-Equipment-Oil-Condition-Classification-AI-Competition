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
import eli5
import matplotlib.gridspec as gridspec
from IPython.display import display
from eli5.permutation_importance import get_score_importances
from eli5.sklearn import PermutationImportance
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_rows', 500)
from DMS_202211.seed_everything import seed_everything

def train_model_lgb_regressor(train,test,params,stratified,num_folds,drop_features,seed_num):
    
    # start log 
    print('-'*50)
    print('>> seed_num:',seed_num)   
    print('>> drop_features:',len(drop_features))
    
    seed_everything(1)
    
    # Divide in training/validation and test data
    train_df = train.copy()
    test_df = test.copy()
    
    train_df['AL'] = np.log(train_df['AL']+1)

    # label encoding 
    encoder = LabelEncoder()
    categorical_features = [i for i in train_df.select_dtypes(include=['object','category']).columns.tolist() if i not in ['ID']]
    categorical_features = [i for i in categorical_features if i in train_df.columns.tolist()]
    for each in categorical_features:
        train_df[each] = encoder.fit_transform(train_df[each])
        test_df[each] = encoder.fit_transform(test_df[each])

    # set training options
    stratified = stratified
    num_folds = num_folds

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1)

    # Create arrays and dataframes to store results
    oof_preds_lgb = np.zeros(train_df.shape[0])
    sub_preds_lgb = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['Y_LABEL','ID','SAMPLE_TRANSFER_DAY','AL']+drop_features]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['YEAR'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['AL'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['AL'].iloc[valid_idx]

        """        
        learning_rate=0.02,
        num_leaves=34,
        colsample_bytree=0.2,
        subsample=0.8715623,
        max_depth=4,
        reg_alpha=0.041545473,
        reg_lambda=0.0735294,
        min_split_gain=0.0222415,
        min_child_weight=39.3259775,
        min_child_samples = 10,
        """

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMRegressor(
            
            learning_rate = params['learning_rate'],
            num_leaves = int(round(params['num_leaves'])),
            colsample_bytree = params['colsample_bytree'],
            subsample = params['subsample'],
            max_depth = int(round(params['max_depth'])),
            reg_alpha = params['reg_alpha'],
            reg_lambda = params['reg_lambda'],
            min_split_gain = params['min_split_gain'],
            min_child_weight = params['min_child_weight'],
            min_child_samples = int(round(params['min_child_samples'])),    
            
            n_jobs = -1,
            n_estimators = 10000,            
            random_state = seed_num,
            silent=-1,
            deterministic=True,
            verbose=-1
        )
        
        with warnings.catch_warnings():
            
            warnings.filterwarnings('ignore')

            clf.fit(
                  train_x
                , train_y
                , eval_set=[(train_x, train_y), (valid_x, valid_y)]
                , eval_metric= 'rmse'
                , verbose= -1
                , early_stopping_rounds= 500
            )

        oof_preds_lgb[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration_)
        sub_preds_lgb += clf.predict(test_df[feats], num_iteration=clf.best_iteration_) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        # print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds_lgb[valid_idx])))
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, mean_squared_error(valid_y, oof_preds_lgb[valid_idx])**0.5))
        
        

    print('Full RMSE score %.6f' % mean_squared_error(train_df['Y_LABEL'], oof_preds_lgb)**0.5)

    # Write submission file and plot feature importance
    train_df['Y_LABEL_lgb'] = oof_preds_lgb
    test_df['Y_LABEL_lgb'] = sub_preds_lgb   
    

    # vi
    # print('-'*50)
    display(feature_importance_df.groupby(['feature'])['importance'].sum().sort_values(ascending=False).head(30))
    # print('-'*50)
    # display_importances(feature_importance_df)   
    
    return train_df,test_df
