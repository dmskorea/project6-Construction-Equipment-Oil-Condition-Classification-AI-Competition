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

def bayes_parameter_opt_xgb(
    train, 
    opt_params, 
    init_round=15, 
    opt_round=25, 
    n_folds=3, 
    random_seed=1, 
    n_estimators=10000, 
    output_process=False, 
    drop_features=[]
    ):   
    
    seed_everything(1)
    
    train_df = train.copy()

    # label encoding 
    encoder = LabelEncoder()
    categorical_features = [i for i in train_df.select_dtypes(include=['object','category']).columns.tolist() if i not in ['ID']]
    categorical_features = [i for i in categorical_features if i in train_df.columns.tolist()]
    for each in categorical_features:
        train_df[each] = encoder.fit_transform(train_df[each])
        
    # feats = [f for f in train_df.columns if f not in ['Y_LABEL','ID','SAMPLE_TRANSFER_DAY']]    
    # X = train_df[feats].copy()
    # y = train_df['Y_LABEL'].copy()
    
    # prepare data
    # train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)
    
    # parameters
    def lgb_eval(
          learning_rate
        , max_leaves
        , colsample_bytree
        , subsample
        , max_depth
        , reg_alpha
        , reg_lambda
        , gamma
        , min_child_weight

    ):
        
        params = {'objective':'binary:logistic'}
        params['learning_rate'] = max(min(learning_rate, 1), 0)
        params["max_leaves"] = int(round(max_leaves))
        params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
        params['subsample'] = max(min(subsample, 1), 0)
        params['max_depth'] = int(round(max_depth))        
        params['reg_alpha'] = reg_alpha
        params['reg_lambda'] = reg_lambda        
        params['gamma'] = gamma
        params['min_child_weight'] = min_child_weight
        
        # cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
        
        # -----        
        
        stratified = True
        
        # Cross validation model
        if stratified:
            folds = StratifiedKFold(n_splits= 5, shuffle=True, random_state=1)
        else:
            folds = KFold(n_splits= 5, shuffle=True, random_state=1)

        # Create arrays and dataframes to store results
        oof_preds_lgb = np.zeros(train_df.shape[0])

        feats = [f for f in train_df.columns if f not in ['Y_LABEL','ID','SAMPLE_TRANSFER_DAY']+drop_features]

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['Y_LABEL'])):
            train_x, train_y = train_df[feats].iloc[train_idx], train_df['Y_LABEL'].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['Y_LABEL'].iloc[valid_idx]

            # LightGBM parameters found by Bayesian optimization
            clf = XGBClassifier(

                **params,

                n_jobs = -1,
                tree_method='gpu_hist',  # THE MAGICAL PARAMETER
                n_estimators = 10000,            
                random_state = 1
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

            oof_preds_lgb[valid_idx] = clf.predict_proba(valid_x)[:, 1]

        cv_result = roc_auc_score(train_df['Y_LABEL'], oof_preds_lgb)        
       
        # ----

        return cv_result

    lgbBO = BayesianOptimization(lgb_eval, opt_params, random_state=1)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        lgbBO.maximize(init_points=init_round, n_iter=opt_round, acq='ucb')
    
    model_auc=[]
    for model in range(len( lgbBO.res)):
        model_auc.append(lgbBO.res[model]['target'])
    
    a1 = lgbBO.res[pd.Series(model_auc).idxmax()]['target'],lgbBO.res[pd.Series(model_auc).idxmax()]['params']
    file_name = 'res_tune_'+str(len(drop_features))+'_'+str(a1[0])+'.joblib'    
    joblib.dump(a1[1],file_name)
    
    return a1
