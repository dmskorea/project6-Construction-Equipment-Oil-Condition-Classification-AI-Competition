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

def bayes_parameter_opt_lgb(
    train, 
    params,
    opt_params, 
    init_round=15, 
    opt_round=25, 
    n_folds=3, 
    seed_num=1, 
    output_process=False, 
    drop_features=[]
    ):   
    
    seed_everything(seed_num)
    
    train_df = train.copy()

    # label encoding 
    encoder = LabelEncoder()
    categorical_features = [i for i in train_df.select_dtypes(include=['object','category']).columns.tolist() if i not in ['ID']]
    categorical_features = [i for i in categorical_features if i in train_df.columns.tolist()]
    for each in categorical_features:
        train_df[each] = encoder.fit_transform(train_df[each])    
   
    # parameters
    def lgb_eval(**params):     
        
        stratified = True
        
        # Cross validation model
        if stratified:
            folds = StratifiedKFold(n_splits= n_folds, shuffle=True, random_state=seed_num)
        else:
            folds = KFold(n_splits= n_folds, shuffle=True, random_state=seed_num)

        # Create arrays and dataframes to store results
        oof_preds_lgb = np.zeros(train_df.shape[0])
        feats = [f for f in train_df.columns if f not in ['Y_LABEL','ID','SAMPLE_TRANSFER_DAY']+drop_features]
        
        params['learning_rate'] = max(min(params['learning_rate'], 1), 0)
        params["num_leaves"] = int(round(params['num_leaves']))
        params['colsample_bytree'] = max(min(params['colsample_bytree'], 1), 0)
        params['subsample'] = max(min(params['subsample'], 1), 0)
        params['max_depth'] = int(round(params['max_depth']))        
        params['reg_alpha'] = params['reg_alpha']
        params['reg_lambda'] = params['reg_lambda']        
        params['min_split_gain'] = params['min_split_gain']
        params['min_child_weight'] = params['min_child_weight']
        params['min_child_samples'] = int(round(params['min_child_samples']))  
        params['seed'] = seed_num
        params['verbose'] = -1

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['Y_LABEL'])):
            train_x, train_y = train_df[feats].iloc[train_idx], train_df['Y_LABEL'].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['Y_LABEL'].iloc[valid_idx]

            categorical_feature = []        
            lgb_train = lgb.Dataset(data=train_x, label=train_y, categorical_feature=categorical_feature)
            lgb_valid = lgb.Dataset(data=valid_x, label=valid_y, reference=lgb_train, categorical_feature=categorical_feature)

            with warnings.catch_warnings():

                warnings.filterwarnings('ignore')  

                clf = lgb.train(

                    params,                
                    train_set = lgb_train,
                    valid_sets = [lgb_train,lgb_valid],
                    verbose_eval = False,
                    num_boost_round = 10000,
                    categorical_feature = categorical_feature
                )

            oof_preds_lgb[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
            
        # find the best thred for f1-score
        f1_score_df = pd.DataFrame()
        for thred in [i/10000 for i in range(0,10000,1) if (i/10000>0.1) & (i/10000<0.3)]:

            a1 = pd.DataFrame()
            f1 = f1_score(train_df['Y_LABEL'], np.where(oof_preds_lgb>=thred,1,0), average='macro')
            a1['f1'] = [f1]
            a1['thred'] = [thred]
            f1_score_df = pd.concat([f1_score_df, a1], axis=0)

        thred = f1_score_df.loc[f1_score_df['f1']==f1_score_df['f1'].max(),'thred'].tolist()[0]
    
        # err
        oof_f1 = f1_score(train_df['Y_LABEL'], np.where(oof_preds_lgb>thred,1,0), average='macro')
       
        # ----

        return oof_f1

    lgbBO = BayesianOptimization(lgb_eval, opt_params, random_state=seed_num)
    
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
