from DMS_202211 import common
from DMS_202211.seed_everything import seed_everything

def kde_target(var_name, df):
    
    # Calculate the correlation coefficient between the new variable and the target
    corr = df['Y_LABEL'].corr(df[var_name])
    
    # Calculate medians for repaid vs not repaid
    avg_target_0 = df.loc[df['Y_LABEL'] == 0, var_name].median()
    avg_target_1 = df.loc[df['Y_LABEL'] == 1, var_name].median()
    
    plt.figure(figsize = (12, 6))
    
    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot(df.loc[df['Y_LABEL'] == 0, var_name], label = 'Y_LABEL == 0')
    sns.kdeplot(df.loc[df['Y_LABEL'] == 1, var_name], label = 'Y_LABEL == 1')
    
    # label the plot
    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)
    plt.legend();
    
    # print out the correlation
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # Print out average values
    print('Median value for avg_target_0 = %0.4f' % avg_target_0)
    print('Median value for avg_target_1 = %0.4f' % avg_target_1)
