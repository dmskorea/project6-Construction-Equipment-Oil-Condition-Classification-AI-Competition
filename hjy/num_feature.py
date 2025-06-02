from DMS_202211 import common
from DMS_202211.seed_everything import seed_everything

def num_feature(df,num_features):
    if num_features[0][:5] == 'rank_':
        num_agg_df = df.groupby("YEAR",sort=False)[num_features].agg(['last'])
    else:
        num_agg_df = df.groupby("YEAR",sort=False)[num_features].agg(['mean', 'std', 'min', 'max', 'sum'])
    num_agg_df.columns = ['_'.join(x) for x in num_agg_df.columns]
    if num_features[0][:5] != 'rank_':
        for col in num_agg_df.columns:
            num_agg_df[col] = num_agg_df[col] // 0.01
    df = num_agg_df.reset_index()
    print('num feature shape after engineering', df.shape )

    return df
