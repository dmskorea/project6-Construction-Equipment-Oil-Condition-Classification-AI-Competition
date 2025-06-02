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

def permutation_vi(train, best_params, n_iter=60):
    
    params = {'application':'binary', 'metric':'auc'}
    params['learning_rate'] = max(min(best_params['learning_rate'], 1), 0)
    params["num_leaves"] = int(round(best_params['num_leaves']))
    params['colsample_bytree'] = max(min(best_params['colsample_bytree'], 1), 0)
    params['subsample'] = max(min(best_params['subsample'], 1), 0)
    params['max_depth'] = int(round(best_params['max_depth']))        
    params['reg_alpha'] = best_params['reg_alpha']
    params['reg_lambda'] = best_params['reg_lambda']        
    params['min_split_gain'] = best_params['min_split_gain']
    params['min_child_weight'] = best_params['min_child_weight']
    params['min_child_samples'] = int(round(best_params['min_child_samples']))

    train_df = train.copy()

    train_df = train_df.replace(np.Inf,0)

    # label encoding 
    encoder = LabelEncoder()
    categorical_features = [i for i in train_df.select_dtypes(include=['object','category']).columns.tolist() if i not in ['ID']]
    categorical_features = [i for i in categorical_features if i in train_df.columns.tolist()]
    for each in categorical_features:
        train_df[each] = encoder.fit_transform(train_df[each])

    stratified = True

    if stratified:
        folds = StratifiedKFold(n_splits= 5, shuffle=True, random_state=1)
    else:
        folds = KFold(n_splits= 5, shuffle=True, random_state=1)

    FEATURES = [f for f in train_df.columns if f not in ['Y_LABEL','ID','SAMPLE_TRANSFER_DAY']]
    TARGET_COL = 'Y_LABEL'
    feature_importance_df = pd.DataFrame()
    for fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df[TARGET_COL])):
        clf = LGBMClassifier(
            **params, 
            random_state=1, 
            n_jobs = -1,
            silent=True,
            n_estimators=10000
        )

        with warnings.catch_warnings():

            warnings.filterwarnings('ignore')

            clf.fit(
                train_df.loc[train_idx, FEATURES], 
                train_df.loc[train_idx, TARGET_COL], 
                eval_metric="auc", 
                verbose=False,
                early_stopping_rounds=500,
                eval_set=[(train_df.loc[valid_idx, FEATURES],train_df.loc[valid_idx, TARGET_COL])]
            )    
            
        # scoring = 'f1'
        perm = PermutationImportance(clf, n_iter=n_iter, scoring='roc_auc',random_state=1).fit(train_df.loc[valid_idx, FEATURES],train_df.loc[valid_idx, TARGET_COL])
        fold_importance_df = eli5.explain_weights_df(perm, top = len(FEATURES), feature_names = FEATURES)
        fold_importance_df["fold"] = fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print(f"Permutation importance for fold {fold}")
    
    return feature_importance_df
