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


fig = plt.figure(figsize=[20,10])
#ax = fig.add_axes([0.15,0.1,0.8,0.8])
gs = fig.add_gridspec(1,2, hspace=0)
axs = gs.subplots(sharex=False, sharey=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

#axs[0].plot(x, y ** 2)
#axs[1].plot(x, 0.3 * y, 'o')
#axs[2].plot(x, y, '+')


# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()

axs[0].boxplot([df.accuracy[df.Model == 'Mdel 1 _baseline_'], \
             df.accuracy[df.Model == 'Mdel 1 5_shards'], \
             #df.accuracy[df.Model == 'Model 2 _baseline_'], \
             #df.accuracy[df.Model == 'Model 2 5_shards']
             ],\
             labels=['1 shard','5 shards'])
axs[0].set_axis_on()             
#axs[0].xticks(fontsize = 20)
#axs[0].yticks(fontsize = 20)
axs[0].grid(visible=True)
axs[0].set_ylim([0.75, 0.8])
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

#ax.set_xticklabels([1,4,5], fontsize=12) 
axs[0].tick_params(axis='y', labelsize=20)

#plt.title(, fontsize = 21)
#plt.xlabel('Model', fontsize = 20)
axs[0].set_ylabel('Accuracy (%)', fontsize = 20)
axs[0].set_title('p-value (%)', fontsize = 20)

#plt.savefig(plots_path+'acc_dis')


#fig = plt.figure(figsize=[10,10])
#ax = fig.add_axes([0.15,0.1,0.8,0.8])
axs[1].set_axis_on()   
axs[1].boxplot([df.training_time[df.Model == 'Mdel 1 _baseline_'], 
             df.training_time[df.Model == 'Mdel 1 5_shards'], \
             #df.training_time[df.Model == 'Model 2 _baseline_'], 
             #df.training_time[df.Model == 'Model 2 5_shards']
             ],\
             labels=['1 shard','5 shards'])

#axs[1].xticks(fontsize = 20)
#axs[1].yticks(fontsize = 20)

axs[1].grid(visible=True)
axs[1].set_ylim([7, 8.5])
#yticks(fontsize=20)
plt.yticks(fontsize=20)
axs[1].tick_params(axis='y', labelsize=20)

#ax[1].set_xticklabels('ff', fontsize=12) 

#axs[1].set_ylabel("ddd")
#plt.title(a_name, fontsize = 21)
#plt.xlabel('Model', fontsize = 20)
axs[1].set_ylabel('Training time (s/CPU)', fontsize = 20)
axs[1].set_title('p-value (%)', fontsize = 20)

plt.savefig(plots_path+'training_dis')
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

