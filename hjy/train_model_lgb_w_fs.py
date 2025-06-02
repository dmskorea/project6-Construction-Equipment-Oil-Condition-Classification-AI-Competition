def train_model_lgb_w_fs(train,test,params,stratified,num_folds,drop_features,seed_num):
    
    
    def sub_model(train,test,params,stratified,num_folds,drop_features,seed_num):

        # start log 
        # print('-'*50)
        # print('>> seed_num:',seed_num)   
        # print('>> drop_features:',len(drop_features))

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
        new_feature = train_df.columns[train_df.columns.str.contains('new')].tolist()
        drop_features = list(set(drop_features))
        feats = [f for f in train_df.columns if f not in ['Y_LABEL','ID','SAMPLE_TRANSFER_DAY']+drop_features]
        feats = feats+new_feature
        
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['Y_LABEL'])):
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

            oof_preds_lgb[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
            sub_preds_lgb += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = feats
            fold_importance_df["importance"] = clf.feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            # print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds_lgb[valid_idx])))
        
        err = roc_auc_score(train_df['Y_LABEL'], oof_preds_lgb)
        # print('Full AUC score %.6f' % err)

        return err, feature_importance_df
    
    # ----------

    # start log 
    print('-'*50)
    print('>> seed_num:',seed_num)   
    print('>> drop_features:',len(drop_features))
    
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
    new_feature = train_df.columns[train_df.columns.str.contains('new')].tolist()
    drop_features = list(set(drop_features+new_feature))
    feats = [f for f in train_df.columns if f not in ['Y_LABEL','ID','SAMPLE_TRANSFER_DAY']+drop_features]
    # feats = list(set(feats))

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['Y_LABEL'])):
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

        oof_preds_lgb[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds_lgb += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        # print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds_lgb[valid_idx])))

    print('Full AUC score %.6f' % roc_auc_score(train_df['Y_LABEL'], oof_preds_lgb))

    # Write submission file and plot feature importance
    # train_df['Y_LABEL_lgb'] = oof_preds_lgb
    # test_df['Y_LABEL_lgb'] = sub_preds_lgb   

    rst_list = []
    vi_list = []
    for i in [176,1,2,3,4,5]:
        err, vi = sub_model(train,test,params,True,5,drop_features,seed_num=i)
        rst_list.append({'seed':i,'err':err})    
        vi_list.append(vi)
        
    print('>> new try')
    print('# of new',len(new_feature))
    print('AUC(mean):',pd.DataFrame(rst_list)['err'].mean()) 
    print('AUC(best):',pd.DataFrame(rst_list)['err'].max())
    print('AUC(stdv):',pd.DataFrame(rst_list)['err'].std())   
    display(pd.concat(vi_list).groupby(['feature'])['importance'].sum().sort_values(ascending=False).head(10))
    return pd.DataFrame(rst_list),pd.concat(vi_list)
