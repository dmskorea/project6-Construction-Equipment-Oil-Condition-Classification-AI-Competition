num_features = ['V40']
group_features = ['psudo_id2_YEAR']

for group in group_features:
    
    for feature in num_features:
        
        a1 = train.sort_values([group]).groupby([group])[feature].agg(['last']).to_dict()['last']
        train[group+'_'+feature+'_last'] = train[group].map(a1)
        a1 = train.sort_values([group]).groupby([group])[feature].agg(['last']).to_dict()['firt']
        train[group+'_'+feature+'_firt'] = train[group].map(a1)
        train[group+'_'+feature+'_LastFirtDiff'] = train[group+'_'+feature+'_last'] - train[group+'_'+feature+'_firt']

        a1 = test.sort_values([group]).groupby([group])[feature].agg(['last']).to_dict()['last']
        test[group+'_'+feature+'_last'] = test[group].map(a1)
        a1 = test.sort_values([group]).groupby([group])[feature].agg(['last']).to_dict()['firt']
        test[group+'_'+feature+'_firt'] = test[group].map(a1)
        test[group+'_'+feature+'_LastFirtDiff'] = test[group+'_'+feature+'_last'] - test[group+'_'+feature+'_firt']
