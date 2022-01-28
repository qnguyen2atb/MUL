from tkinter import font
from matplotlib import markers
from pyparsing import line
from lib import *
#%% Data visualization

# dataset path
#path = 'F:\Machine_Unlearning\Datatest'

class metricsAnalysis():
    
    def __init__(self) -> None:        
        self.path = './'
        self.plots_path = '../plots/'
        # import dataset
        d_name = 'metrics_all.csv'
        d = os.path.join(self.path, d_name)

        # read data
        self.df = pd.read_csv(d)
    
    def analyze_baselinemodels(self, ind_l, a_name):
        
        metrics = pd.DataFrame()
        for ind in ind_l:
            metrics=metrics.append(self.df.iloc[ind])

        fig = plt.figure(figsize=[15,15])
        #plt.figure(figsize=(16,9))
        #ax = fig.add_axes([0.1,0.3,0.8,0.65])
        
        
        plt.scatter(metrics.training_time, metrics.accuracy*100, marker='+', s=20, c ='b', linewidths=20)
        #plt.hlines(metrics.accuracy[50]*100,xmin=metrics.training_time.min()-3,xmax=metrics.training_time.max()+6)
        #plt.xlim(metrics.training_time.min()-3,metrics.training_time.max()+6)
        plt.xlim(-2,18)
        plt.ylim(74.8, 80)
        print(metrics)
        for i in metrics.index:         
            label = metrics.Model[i]
            print(label)
            
            if label == 'aggregated_model_B4_20s': 
                plt.annotate(label, # this is the text
                            (metrics.training_time[i], metrics.accuracy[i]*100.), # these are the coordinates to position the label
                            textcoords="offset points", # how to position the text
                            xytext=(20,30), # distance from text to points (x,y)
                            size=20,
                            ha='center') # horizontal alignment can be left, right or center
            elif label == 'Shard model for removing total_score < 5': 
                plt.annotate(label, # this is the text
                            (metrics.training_time[i], metrics.accuracy[i]*100.), # these are the coordinates to position the label
                            textcoords="offset points", # how to position the text
                            xytext=(-100,10), # distance from text to points (x,y)
                            size=20,
                            ha='center') # horizontal alignment can be left, right or center
            elif label == 'Original model with 5 shards': 
                plt.annotate(label, # this is the text
                            (metrics.training_time[i], metrics.accuracy[i]*100.), # these are the coordinates to position the label
                            textcoords="offset points", # how to position the text
                            xytext=(50,10), # distance from text to points (x,y)
                            size=20,
                            ha='center') # horizontal alignment can be left, right or center
            else:
                plt.annotate(label, # this is the text
                            (metrics.training_time[i], metrics.accuracy[i]*100.), # these are the coordinates to position the label
                            textcoords="offset points", # how to position the text
                            xytext=(0,10), # distance from text to points (x,y)
                            size=20,
                            ha='center') # horizontal alignment can be left, right or center

        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.grid(visible=True)
        plt.title(a_name, fontsize = 21)
        plt.xlabel('Training time (s/CPU)', fontsize = 20)
        plt.ylabel('Accuracy (%)', fontsize = 20)
        plt.savefig(self.plots_path+str(a_name))
        #plt.show()
            

m = metricsAnalysis()
#m.analyze_baselinemodels([0, 12, 25],'baseline_models')
#m.analyze_baselinemodels([58, 71, 89, 112, 140, 173],'different_sharding_models')
m.analyze_baselinemodels([0, 50, 58, 71, 89, 112, 140],'Request for removing the total_score from 20 to 25')
m.analyze_baselinemodels([0, 5, 45],'Request for removing the total_score less than 5')

m.analyze_baselinemodels([0,  20, 33, 9],'Original, baseline models (1 shards) and shard on original model')


