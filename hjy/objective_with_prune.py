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
from DMS_202211.seed_everything import seed_everything

pd.set_option('display.max_rows', 500)

from tqdm import tqdm_notebook as tqdm
import optuna
from optuna import Trial

# study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=100),direction="maximize")
# study.optimize(objective_with_prune, n_trials=200)

def objective_with_prune(trial:Trial):  
       
    global seed_num, drop_features, categorical_feature
       
    print('seed_num',seed_num)
       
    def f1_eval(yhat,data):

        y = data.get_label()
        thred = 0.1636 # 0.125~0.13
        pred = np.where(yhat>thred,1,0)

        return 'f1', f1_score(y, pred, average='macro'),True 
    
    
    def fit_lgbm_with_pruning(trial, train_x, train_y, valid_x, valid_y, drop_features, categorical_feature, seed_num):

        lgb_train = lgb.Dataset(data=train_x, label=train_y, categorical_feature=categorical_feature)
        lgb_valid = lgb.Dataset(data=valid_x, label=valid_y, reference=lgb_train, categorical_feature=categorical_feature)

        """
        opt_params = {
        'learning_rate': (0.02, 0.08),
        'num_leaves': (24, 80),
        'colsample_bytree': (0.2, 0.8),
        'subsample': (0.2, 0.8),
        'max_depth': (3, 8),
        'reg_alpha': (0.01, 10),
        'reg_lambda': (1, 10),
        'min_split_gain': (0.01, 1),
        'min_child_weight': (0.001,1),  
        'min_child_samples':(10, 50)
        }    
        """      
        params = {

            'objective': 'binary',
            "boosting": "gbdt",
            "verbose": -1,

            'learning_rate': trial.suggest_uniform('colsample_bytree', 0.02, 0.08),
            'num_leaves': trial.suggest_int('num_leaves', 24, 80),
            "colsample_bytree": trial.suggest_uniform('colsample_bytree', 0.2, 0.8),
            "subsample": trial.suggest_uniform('subsample', 0.2, 0.8),
            'max_depth': trial.suggest_int('max_depth', 3, 8),  
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.01, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1, 10.0),
            "min_split_gain": trial.suggest_uniform('min_split_gain', 0.01, 1.0),
            "min_child_weight": trial.suggest_uniform('min_child_weight', 0.001, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50)
        }

        params['random_state'] = seed_num

        # callback for pruning.
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'f1', valid_name='valid_1')

        # train 
        model = lgb.train(
            params,
            train_set=lgb_train,
            num_boost_round=10000,
            feval = f1_eval,
            valid_sets=[lgb_train, lgb_valid],
            verbose_eval=False,
            early_stopping_rounds=200,
            callbacks=[pruning_callback]
        )

        # predictions
        y_pred_valid = model.predict(valid_x, num_iteration=model.best_iteration)    
        print('best_score(valid/f1):', model.best_score['valid_1']['f1'])
        log = {'train/f1': model.best_score['training']['f1'],'valid/f1': model.best_score['valid_1']['f1']}

        return model, y_pred_valid, log
    
    # ----
    
    seed_everything(seed_num)    
    
    train_df = train.copy()  
    
    # label encoding 
    encoder = LabelEncoder()
    categorical_features = [i for i in train_df.select_dtypes(include=['object','category']).columns.tolist() if i not in ['ID']]
    categorical_features = [i for i in categorical_features if i in train_df.columns.tolist()]
    for each in categorical_features:
        train_df[each] = encoder.fit_transform(train_df[each])
        
    # split 
    folds = StratifiedKFold(n_splits= 5, shuffle=True, random_state=seed_num) 
    feats = [f for f in train_df.columns if f not in ['Y_LABEL','ID','SAMPLE_TRANSFER_DAY']+drop_features]
    X_train = train_df[feats]
    y_train = train_df['Y_LABEL']    
    y_valid_pred_total = np.zeros(X_train.shape[0])
    
    # train 
    models0 = []
    valid_score = 0
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['Y_LABEL'])):
        
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['Y_LABEL'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['Y_LABEL'].iloc[valid_idx]
        
        ################################################## 
        model, y_pred_valid, log = fit_lgbm_with_pruning(
            trial, 
            train_x, train_y, valid_x, valid_y,
            drop_features,
            categorical_feature,
            seed_num
        )
        ################################################## 
        
        y_valid_pred_total[valid_idx] = y_pred_valid
        models0.append(model)
        gc.collect()
        
        valid_score += log["valid/f1"]
        
    valid_score /= len(models0)
    
    return valid_score
