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

def train_model_w_fs(train,test,params,stratified,num_folds,drop_features,seed_num,score_diff_thred=0.001,drop_size=1):
    
    # start log 
    print('-'*50)
    print('>> seed_num:',seed_num)   
    print('>> drop_features:',len(drop_features))    
    print('>> score_diff_thred',score_diff_thred)
    print('>> drop_size',drop_size)
    
    seed_everything(1)
    
    # Divide in training/validation and test data
    train_df = train.copy()
    test_df = test.copy()

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
    origin_feats = [f for f in train_df.columns if f not in ['Y_LABEL','ID','SAMPLE_TRANSFER_DAY']+drop_features] 
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[origin_feats], train_df['Y_LABEL'])):
        
        feats = [f for f in train_df.columns if f not in ['Y_LABEL','ID','SAMPLE_TRANSFER_DAY']+drop_features]
        
        def make_lgb(train_df,feats,train_idx, valid_idx,n_fold):        
        
            train_x, train_y = train_df[feats].iloc[train_idx], train_df['Y_LABEL'].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['Y_LABEL'].iloc[valid_idx]

            # LightGBM parameters found by Bayesian optimization
            clf = LGBMClassifier(

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
                    , eval_metric= 'auc'
                    , verbose= -1
                    , early_stopping_rounds= 500
                )
                
                fold_importance_df = pd.DataFrame()
                fold_importance_df["feature"] = feats
                fold_importance_df["importance"] = clf.feature_importances_
                fold_importance_df["fold"] = n_fold + 1  
             
            return clf, fold_importance_df, valid_x,valid_y
        
        clf, fold_importance_df, valid_x,valid_y = make_lgb(train_df,feats,train_idx, valid_idx,n_fold)          
        
        oof_preds_lgb[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        score = roc_auc_score(valid_y, oof_preds_lgb[valid_idx])
        print('Fold %2d AUC : %.6f' % (n_fold + 1, score))      
        
        new_score = 0
        while np.round(score,5) < 0.75:
            
            # 변수 중요도 낮은거 1개씩 제거 
            feats = fold_importance_df.sort_values(['importance'])['feature'].tolist()[:-drop_size]  
            clf, fold_importance_df, valid_x,valid_y = make_lgb(train_df,feats,train_idx, valid_idx,n_fold)            
            oof_preds_lgb[valid_idx] = clf.predict_proba(valid_x[feats], num_iteration=clf.best_iteration_)[:, 1]
            new_score = roc_auc_score(valid_y, oof_preds_lgb[valid_idx])     
            
            if (new_score < score):                
                
                # 점수가 낮으면 한번더 배기 
                feats = fold_importance_df.sort_values(['importance'])['feature'].tolist()[:-drop_size]  
                clf, fold_importance_df, valid_x,valid_y = make_lgb(train_df,feats,train_idx, valid_idx,n_fold)            
                oof_preds_lgb[valid_idx] = clf.predict_proba(valid_x[feats], num_iteration=clf.best_iteration_)[:, 1]
                new_score = roc_auc_score(valid_y, oof_preds_lgb[valid_idx])
                
            if (score-new_score)>score_diff_thred:
                
                # 다시 원상복귀 
                feats = origin_feats
                clf, fold_importance_df, valid_x,valid_y = make_lgb(train_df,feats,train_idx, valid_idx,n_fold)
                oof_preds_lgb[valid_idx] = clf.predict_proba(valid_x[feats], num_iteration=clf.best_iteration_)[:, 1]                
                sub_preds_lgb += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
                old_score = roc_auc_score(valid_y, oof_preds_lgb[valid_idx])
                print('---------------> Fold %2d AUC : %.6f' % (n_fold + 1, old_score)) 
                break       
            
            if (new_score > score):   
                
                # 점수 높으면 while 중지 
                sub_preds_lgb += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
                print('---------------> Fold %2d AUC : %.6f' % (n_fold + 1, new_score)) 
                break 

    print('Full AUC score %.6f' % roc_auc_score(train_df['Y_LABEL'], oof_preds_lgb))

    # Write submission file and plot feature importance
    test_df['Y_LABEL_lgb'] = sub_preds_lgb

    # vi
    # print('-'*50)
    # display(feature_importance_df.groupby(['feature'])['importance'].sum().sort_values(ascending=False).head(30))
    # print('-'*50)
    # display_importances(feature_importance_df)
    
    # train auc
    oof_auc = roc_auc_score(train_df['Y_LABEL'], oof_preds_lgb)

    # find the best thred for f1-score
    f1_score_df = pd.DataFrame()
    for thred in [i/10000 for i in range(0,10000,1) if (i/10000>0.1) & (i/10000<0.35)]:

        a1 = pd.DataFrame()
        f1 = f1_score(train_df['Y_LABEL'], np.where(oof_preds_lgb>thred,1,0), average='macro')
        a1['f1'] = [f1]
        a1['thred'] = [thred]
        f1_score_df = pd.concat([f1_score_df, a1], axis=0)

    thred = f1_score_df.loc[f1_score_df['f1']==f1_score_df['f1'].max(),'thred'].tolist()[0]
    print('thred:',thred)
    print('ncol',len(feats))

    # train f1
    print('auc:',oof_auc)
    oof_f1 = f1_score(train_df['Y_LABEL'], np.where(oof_preds_lgb>thred,1,0), average='macro')
    print('f1:',oof_f1)
    a1 = train_df['Y_LABEL'].value_counts()/len(train_df)
    print('Target ratio(real):',(a1[1]))

    # test err
    test_df['TARGET'] = np.where(test_df['Y_LABEL_lgb']>thred,1,0)
    a1 = test_df['TARGET'].value_counts()/len(test_df)
    print('Target ratio(pred):',(a1[1]))
    target_sum = test_df['TARGET'].sum()
    print('Target sum:',target_sum)        

    # save 
    train_df['Y_LABEL_lgb'] = oof_preds_lgb
    a1 = train_df[['ID','YEAR','COMPONENT_ARBITRARY','Y_LABEL_lgb','Y_LABEL']].copy()
    a1.to_csv('train_pred_'+str(seed_num)+'_'+str(np.round(oof_f1,10))+'.csv', index= False)    
    a1 = test_df[['ID','YEAR','COMPONENT_ARBITRARY','Y_LABEL_lgb']].copy()
    a1.to_csv('test_pred_'+str(seed_num)+'_'+str(np.round(oof_f1,10))+'.csv', index= False)

    # submit
    a1 = test_df[['ID', 'TARGET']].copy()
    a1 = a1.rename(columns={'TARGET':'Y_LABEL'})
    submission_file_name = 'sample_submission_lgb_'+str(np.round(oof_f1,4))+'.csv'
    a1.to_csv(submission_file_name, index= False)
