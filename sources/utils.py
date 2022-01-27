from lib import *
#%% Data visualization

# dataset path
#path = 'F:\Machine_Unlearning\Datatest'
path = './'
plots_path = '../plots/'

# import dataset
d_name = 'metrics_all.csv'
d = os.path.join(path, d_name)

# read data
df = pd.read_csv(d)
print(df)
# read data
#feature_names = ['Age','Tenure','PSYTE_Segment','Total_score','Trnx_count','num_products','mean_trnx_amt']

