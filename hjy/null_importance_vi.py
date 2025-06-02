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

def null_importance_vi(train,nb_runs):
    
    seed_everything(1)
    
    train_df = train.copy()

    # label encoding 
    encoder = LabelEncoder()
    categorical_features = [i for i in train_df.select_dtypes(include=['object','category']).columns.tolist() if i not in ['ID']]
    categorical_features = [i for i in categorical_features if i in train_df.columns.tolist()]
    for each in categorical_features:
        train_df[each] = encoder.fit_transform(train_df[each])
        
    data = train_df.copy()
    
    def get_feature_importances(data, shuffle, categorical_features, seed=None):
        
        train_df = data.copy()
        
        # Shuffle target if required
        if shuffle:
            # Here you could as well use a binomial distribution
            train_df['Y_LABEL'] = train_df['Y_LABEL'].copy().sample(frac=1.0)        

        # Gather real features
        feats = [f for f in train_df.columns if f not in ['Y_LABEL','ID','SAMPLE_TRANSFER_DAY']]
        
        stratified = True
        
        # Cross validation model
        if stratified:
            folds = StratifiedKFold(n_splits= 5, shuffle=True, random_state=1)
        else:
            folds = KFold(n_splits= 5, shuffle=True, random_state=1)
        
        params =  {
            'learning_rate': 0.02,
            'max_depth': 4,
            'colsample_bytree': 0.2,
            'subsample': 0.8715623,
            'min_child_samples': 10,
            'min_child_weight': 39.3259775,
            'min_split_gain': 0.0222415,
            'num_leaves': 34,
            'reg_alpha': 0.041545473,
            'reg_lambda': 0.0735294
        }
        
        feature_importance_df = pd.DataFrame()
        oof_preds_lgb = np.zeros(data.shape[0])

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['Y_LABEL'])):
            train_x, train_y = train_df[feats].iloc[train_idx], train_df['Y_LABEL'].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['Y_LABEL'].iloc[valid_idx]

            # LightGBM parameters found by Bayesian optimization
            clf = LGBMClassifier(

                **params,

                n_jobs = -1,
                n_estimators = 10000,            
                random_state = 1,
                silent=True,
                deterministic=True,
                verbose=-100
            )

            with warnings.catch_warnings():

                warnings.filterwarnings('ignore')

                clf.fit(
                      train_x
                    , train_y
                    , eval_set=[(train_x, train_y), (valid_x, valid_y)]
                    , eval_metric= 'auc'
                    , verbose= False
                    , early_stopping_rounds= 500
                )
                
            oof_preds_lgb[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

            # Get feature importances
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = feats
            fold_importance_df["importance_gain"] = clf.booster_.feature_importance(importance_type='gain')
            fold_importance_df["importance_split"] = clf.booster_.feature_importance(importance_type='split')    
            fold_importance_df["fold"] = n_fold + 1
            fold_importance_df['trn_score'] = roc_auc_score(valid_y, oof_preds_lgb[valid_idx])
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            
        # imp_df
        imp_df = pd.DataFrame()
        imp_df["feature"] = feats
        imp_df['importance_gain'] = imp_df["feature"].map(feature_importance_df.groupby(['feature'])['importance_gain'].sum().to_dict())
        imp_df['importance_split'] = imp_df["feature"].map(feature_importance_df.groupby(['feature'])['importance_split'].sum().to_dict())
        imp_df['trn_score'] = imp_df["feature"].map(feature_importance_df.groupby(['feature'])['trn_score'].mean().to_dict())
        
        return imp_df
    
    # Seed the unexpected randomness of this world
    np.random.seed(1)
    # Get the actual importance, i.e. without shuffling
    actual_imp_df = get_feature_importances(data=data, shuffle=False, categorical_features=categorical_features)
    
    null_imp_df = pd.DataFrame()
    nb_runs = nb_runs # 80
    import time
    start = time.time()
    dsp = ''
    for i in range(nb_runs):
        # Get current run importances
        imp_df = get_feature_importances(data=data, shuffle=True, categorical_features=categorical_features)
        imp_df['run'] = i + 1 
        # Concat the latest importances with the old ones
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
        # Erase previous message
        for l in range(len(dsp)):
            print('\b', end='', flush=False)
        # Display current run and time used
        spent = (time.time() - start) / 60
        dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
        print(dsp, end='', flush=True)
        
    return actual_imp_df, null_imp_df
