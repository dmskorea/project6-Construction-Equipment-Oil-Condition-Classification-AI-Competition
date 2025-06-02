from DMS_202211 import common
from DMS_202211.seed_everything import seed_everything

def diff_feature(df,num_features):
    diff_num_features = [f'diff_{col}' for col in num_features]
    cids = df['YEAR'].values
    df = df.groupby('YEAR')[num_features].diff().add_prefix('diff_')
    df.insert(0,'YEAR',cids)
    num_agg_df = df.groupby("YEAR",sort=False)[diff_num_features].agg(['mean', 'std', 'min', 'max', 'sum'])
    num_agg_df.columns = ['_'.join(x) for x in num_agg_df.columns]
    for col in num_agg_df.columns:
        num_agg_df[col] = num_agg_df[col] // 0.01

    df = num_agg_df.reset_index()
    print('diff feature shape after engineering', df.shape )

    return df
