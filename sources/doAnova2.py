import pandas as pd
# load data file
df = pd.read_csv("foranova.csv", sep=",")
# reshape the d dataframe suitable for statsmodels package 
#df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=['A', 'B', 'C', 'D'])
# replace column names
#df_melt.columns = ['index', 'treatments', 'value']

plots_path = '../plots/'

print(df)

import matplotlib.pyplot as plt


fig = plt.figure(figsize=[10,10])
ax = fig.add_axes([0.15,0.1,0.8,0.8])
df['accuracy'] = df['accuracy']*100 
plt.boxplot([df.accuracy[df.Model == 'Model 1 _baseline_'], \
             df.accuracy[df.Model == 'Model 1 5_shards'], \
             #df.accuracy[df.Model == 'Model 2 _baseline_'], \
             #df.accuracy[df.Model == 'Model 2 5_shards']
             ],\
             labels=['1 shard','5 shards'])
             #labels=['',''])
plt.grid()
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
plt.grid(visible=True)
plt.title('p-value=3.2e-9', fontsize = 30)
#plt.xlabel('Model', fontsize = 20)
plt.ylabel('Accuracy (%)', fontsize =30)
plt.savefig(plots_path+'acc_dis1')


#fig = plt.figure(figsize=[10,10])
#ax = fig.add_axes([0.15,0.1,0.8,0.8])

plt.boxplot([df.training_time[df.Model == 'Model 1 _baseline_'], 
             df.training_time[df.Model == 'Model 1 5_shards'], \
             #df.training_time[df.Model == 'Model 2 _baseline_'], 
             #df.training_time[df.Model == 'Model 2 5_shards']
             ],\
             labels=['1 shard','5 shards'])

plt.grid()
plt.ylim([6, 9])
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
plt.grid(visible=True)
plt.title('p-value=2.5e-8', fontsize = 30)
#plt.xlabel('Model', fontsize = 20)
plt.ylabel('Training time (s/CPU)', fontsize = 30)
plt.savefig(plots_path+'training_dis1')
#plt.show()


# generate a boxplot to see the data distribution by treatments. Using boxplot, we can 
# easily detect the differences between different treatments
'''
import matplotlib.pyplot as plt
import seaborn as sns
ax = sns.boxplot(x='treatments', y='value', data=df_melt, color='#99c2a2')
ax = sns.swarmplot(x="treatments", y="value", data=df_melt, color='#7d0013')
plt.show()


'''


import scipy.stats as stats
# stats f_oneway functions takes the groups as input and returns ANOVA F and p value
fvalue, pvalue = stats.f_oneway(df.training_time[df.Model == 'Mdel 1 _baseline_'], 
             df.training_time[df.Model == 'Mdel 1 _baseline_'], \
             df.training_time[df.Model == 'Model 2 _baseline_'], 
             df.training_time[df.Model == 'Model 2 5_shards'])
print(fvalue, pvalue)
# 110.67148914037737 5.478060300543516e-10


fvalue, pvalue = stats.f_oneway(df.accuracy[df.Model == 'Mdel 1 _baseline_'], \
             df.accuracy[df.Model == 'Mdel 1 5_shards'], \
             df.accuracy[df.Model == 'Model 2 _baseline_'], \
             df.accuracy[df.Model == 'Model 2 5_shards'])
print(fvalue, pvalue)


fvalue, pvalue = stats.f_oneway(df.training_time[df.Model == 'Mdel 1 _baseline_'], \
             df.training_time[df.Model == 'Model 2 _baseline_'])
print(fvalue, pvalue)


fvalue, pvalue = stats.f_oneway(df.training_time[df.Model == 'Mdel 1 5_shards'], \
             df.training_time[df.Model == 'Model 2 5_shards'])
print(fvalue, pvalue)




fvalue, pvalue = stats.f_oneway(df.accuracy[df.Model == 'Mdel 1 _baseline_'], \
             df.accuracy[df.Model == 'Model 2 _baseline_'])
print(fvalue, pvalue)


fvalue, pvalue = stats.f_oneway(df.accuracy[df.Model == 'Mdel 1 5_shards'], \
             df.accuracy[df.Model == 'Model 2 5_shards'])
print(fvalue, pvalue)




fvalue, pvalue = stats.f_oneway(df.accuracy[df.Model == 'Mdel 1 _baseline_'], \
             df.accuracy[df.Model == 'Mdel 1 5_shards'])
print(fvalue, pvalue)

fvalue, pvalue = stats.f_oneway(df.training_time[df.Model == 'Mdel 1 _baseline_'], \
             df.training_time[df.Model == 'Mdel 1 5_shards'])
print(fvalue, pvalue)


fvalue, pvalue = stats.f_oneway(df.accuracy[df.Model == 'Model 2 _baseline_'], \
             df.accuracy[df.Model == 'Model 2 5_shards'])
print(fvalue, pvalue)


fvalue, pvalue = stats.f_oneway(df.training_time[df.Model == 'Model 2 _baseline_'], \
             df.training_time[df.Model == 'Model 2 5_shards'])
print(fvalue, pvalue)