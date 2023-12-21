from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import time

# Here I change the type of some feature becuase since they come from a network package they are supposed to be a certain amount of bit maximum, I also checked before to do the change.

# Then I will eclude the ip of the hosts, the port and the Unnamed: 0. Because the ip and ports are categorical but they are to many to fit in the model, and also there is not a good reason for train the model over the ip since it change based on the network so the attacker will always have a different one. About the Unnamed: 0 you can use that number to split this csv in mani csvs which is not a thing that we need to do so I removed that feature as well.

dtype_dict = {
    'Unnamed: 0': 'uint32',
    'uid': 'str',
    'originh': 'category',
    'originp': 'uint16',
    'responh': 'category',
    'responp': 'uint16',
    'flow_duration': 'float64',
    'fwd_pkts_tot': 'uint64',
    'bwd_pkts_tot': 'uint64',
    'fwd_data_pkts_tot': 'uint64',
    'bwd_data_pkts_tot': 'uint64',
    'fwd_pkts_per_sec': 'float64',
    'bwd_pkts_per_sec': 'float64',
    'flow_pkts_per_sec': 'float64',
    'down_up_ratio': 'float32',
    'fwd_header_size_tot': 'uint64',
    'fwd_header_size_min': 'uint8',
    'fwd_header_size_max': 'uint8',
    'bwd_header_size_tot': 'uint64',
    'bwd_header_size_min': 'uint8',
    'bwd_header_size_max': 'uint8',
    'flow_FIN_flag_count': 'uint64',
    'flow_SYN_flag_count': 'uint64',
    'flow_RST_flag_count': 'uint64',
    'fwd_PSH_flag_count': 'uint64',
    'bwd_PSH_flag_count': 'uint64',
    'flow_ACK_flag_count': 'uint64',
    'fwd_URG_flag_count': 'uint64',
    'bwd_URG_flag_count': 'uint64',
    'flow_CWR_flag_count': 'uint64',
    'flow_ECE_flag_count': 'uint64',
    'fwd_pkts_payload.min': 'uint16',
    'fwd_pkts_payload.max': 'uint16',
    'fwd_pkts_payload.tot': 'float64',
    'fwd_pkts_payload.avg': 'float64',
    'fwd_pkts_payload.std': 'float64',
    'bwd_pkts_payload.min': 'uint16',
    'bwd_pkts_payload.max': 'uint16',
    'bwd_pkts_payload.tot': 'float64',
    'bwd_pkts_payload.avg': 'float64',
    'bwd_pkts_payload.std': 'float64',
    'flow_pkts_payload.min': 'uint16',
    'flow_pkts_payload.max': 'uint16',
    'flow_pkts_payload.tot': 'float64',
    'flow_pkts_payload.avg': 'float64',
    'flow_pkts_payload.std': 'float64',
    'fwd_iat.min': 'float64',
    'fwd_iat.max': 'float64',
    'fwd_iat.tot': 'float64',
    'fwd_iat.avg': 'float64',
    'fwd_iat.std': 'float64',
    'bwd_iat.min': 'float64',
    'bwd_iat.max': 'float64',
    'bwd_iat.tot': 'float64',
    'bwd_iat.avg': 'float64',
    'bwd_iat.std': 'float64',
    'flow_iat.min': 'float64',
    'flow_iat.max': 'float64',
    'flow_iat.tot': 'float64',
    'flow_iat.avg': 'float64',
    'flow_iat.std': 'float64',
    'payload_bytes_per_second': 'float64',
    'fwd_subflow_pkts': 'float64',
    'bwd_subflow_pkts': 'float64',
    'fwd_subflow_bytes': 'float64',
    'bwd_subflow_bytes': 'float64',
    'fwd_bulk_bytes': 'float64',
    'bwd_bulk_bytes': 'float64',
    'fwd_bulk_packets': 'float32',
    'bwd_bulk_packets': 'float32',
    'fwd_bulk_rate': 'float64',
    'bwd_bulk_rate': 'float64',
    'active.min': 'float64',
    'active.max': 'float64',
    'active.tot': 'float64',
    'active.avg': 'float64',
    'active.std': 'float64',
    'idle.min': 'float64',
    'idle.max': 'float64',
    'idle.tot': 'float64',
    'idle.avg': 'float64',
    'idle.std': 'float64',
    'fwd_init_window_size': 'uint16',
    'bwd_init_window_size': 'uint16',
    'fwd_last_window_size': 'uint16',
    'traffic_category': 'category',
    'Label': 'bool'
}

selected_features = [
    "flow_duration", "fwd_pkts_tot", "bwd_pkts_tot",
    "fwd_data_pkts_tot", "bwd_data_pkts_tot", "fwd_pkts_per_sec", "bwd_pkts_per_sec", "flow_pkts_per_sec",
    "down_up_ratio", "fwd_header_size_tot", "fwd_header_size_min", "fwd_header_size_max",
    "bwd_header_size_tot", "bwd_header_size_min", "bwd_header_size_max", "flow_FIN_flag_count",
    "flow_SYN_flag_count", "flow_RST_flag_count", "fwd_PSH_flag_count", "bwd_PSH_flag_count", "flow_ACK_flag_count",
    "fwd_URG_flag_count", "bwd_URG_flag_count", "flow_CWR_flag_count", "flow_ECE_flag_count",
    "fwd_pkts_payload.min", "fwd_pkts_payload.max", "fwd_pkts_payload.tot", "fwd_pkts_payload.avg",
    "fwd_pkts_payload.std", "bwd_pkts_payload.min", "bwd_pkts_payload.max", "bwd_pkts_payload.tot",
    "bwd_pkts_payload.avg", "bwd_pkts_payload.std", "flow_pkts_payload.min", "flow_pkts_payload.max",
    "flow_pkts_payload.tot", "flow_pkts_payload.avg", "flow_pkts_payload.std", "fwd_iat.min",
    "fwd_iat.max", "fwd_iat.tot", "fwd_iat.avg", "fwd_iat.std", "bwd_iat.min", "bwd_iat.max",
    "bwd_iat.tot", "bwd_iat.avg", "bwd_iat.std", "flow_iat.min", "flow_iat.max", "flow_iat.tot",
    "flow_iat.avg", "flow_iat.std", "payload_bytes_per_second", "fwd_subflow_pkts", "bwd_subflow_pkts",
    "fwd_subflow_bytes", "bwd_subflow_bytes", "fwd_bulk_bytes", "bwd_bulk_bytes", "fwd_bulk_packets",
    "bwd_bulk_packets", "fwd_bulk_rate", "bwd_bulk_rate", "active.min", "active.max", "active.tot",
    "active.avg", "active.std", "idle.min", "idle.max", "idle.tot", "idle.avg", "idle.std",
    "fwd_init_window_size", "bwd_init_window_size", "fwd_last_window_size", "traffic_category", "Label"
]

x_features = [    
    "flow_duration", "fwd_pkts_tot", "bwd_pkts_tot",
    "fwd_data_pkts_tot", "bwd_data_pkts_tot", "fwd_pkts_per_sec", "bwd_pkts_per_sec", "flow_pkts_per_sec",
    "down_up_ratio", "fwd_header_size_tot", "fwd_header_size_min", "fwd_header_size_max",
    "bwd_header_size_tot", "bwd_header_size_min", "bwd_header_size_max", "flow_FIN_flag_count",
    "flow_SYN_flag_count", "flow_RST_flag_count", "fwd_PSH_flag_count", "bwd_PSH_flag_count", "flow_ACK_flag_count",
    "fwd_URG_flag_count", "bwd_URG_flag_count", "flow_CWR_flag_count", "flow_ECE_flag_count",
    "fwd_pkts_payload.min", "fwd_pkts_payload.max", "fwd_pkts_payload.tot", "fwd_pkts_payload.avg",
    "fwd_pkts_payload.std", "bwd_pkts_payload.min", "bwd_pkts_payload.max", "bwd_pkts_payload.tot",
    "bwd_pkts_payload.avg", "bwd_pkts_payload.std", "flow_pkts_payload.min", "flow_pkts_payload.max",
    "flow_pkts_payload.tot", "flow_pkts_payload.avg", "flow_pkts_payload.std", "fwd_iat.min",
    "fwd_iat.max", "fwd_iat.tot", "fwd_iat.avg", "fwd_iat.std", "bwd_iat.min", "bwd_iat.max",
    "bwd_iat.tot", "bwd_iat.avg", "bwd_iat.std", "flow_iat.min", "flow_iat.max", "flow_iat.tot",
    "flow_iat.avg", "flow_iat.std", "payload_bytes_per_second", "fwd_subflow_pkts", "bwd_subflow_pkts",
    "fwd_subflow_bytes", "bwd_subflow_bytes", "fwd_bulk_bytes", "bwd_bulk_bytes", "fwd_bulk_packets",
    "bwd_bulk_packets", "fwd_bulk_rate", "bwd_bulk_rate", "active.min", "active.max", "active.tot",
    "active.avg", "active.std", "idle.min", "idle.max", "idle.tot", "idle.avg", "idle.std",
    "fwd_init_window_size", "bwd_init_window_size", "fwd_last_window_size"
]

# Function to create dataframe with metrics
def performanceMetricsDF(metricsObj, yTrain, yPredTrain, yTest, yPredTest, average='binary'):
  measures_list = ['ACCURACY','PRECISION', 'RECALL','F1 SCORE','AUC']
  train_results = [metricsObj.accuracy_score(yTrain, yPredTrain),
                metricsObj.precision_score(yTrain, yPredTrain, average = average),
                metricsObj.recall_score(yTrain, yPredTrain, average = average),
                metricsObj.f1_score(yTrain, yPredTrain, average = average),
                metricsObj.roc_auc_score(yTrain, yPredTrain, average = None if average == 'binary' else average)
                ]
  test_results = [metricsObj.accuracy_score(yTest, yPredTest),
               metricsObj.precision_score(yTest, yPredTest, average = average),
               metricsObj.recall_score(yTest, yPredTest, average = average),
               metricsObj.f1_score(yTest, yPredTest, average = average),
               metricsObj.roc_auc_score(yTest, yPredTest, average = None if average == 'binary' else average)
               ]
  resultsDF = pd.DataFrame({'Measure': measures_list, 'Train': train_results, 'Test':test_results})
  return(resultsDF)

# Function to plot confusion matrix - Adapted from https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    ax = sns.heatmap(cf,annot=box_labels, fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)

def compute_ratio(data):
    # Get ratio instead of raw numbers using normalize=True
    ratio = data['traffic_category'].value_counts(normalize=True)

    # Round and then convert to percentage
    ratio = ratio.round(4)*100

    # convert to a DataFrame and store in variable 'traffic_category_ratios'
    # We'll use this variable to compare ratios for samples 
    # selected using SRS and Stratified Sampling 
    traffic_category_ratios = pd.DataFrame({'Ratio':ratio})
    print(traffic_category_ratios)

def show_corr_matrix(ds):
    sns.set_theme(style="white")

    # Compute the correlation matrix
    corr = ds.loc[:, ds.columns != 'traffic_category'].corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

class DataPreprocessingAndValidation:

    def __init__(self, ds, category_size, cv, init, params, scaler = None):
        self.ds = ds
        self.category_size = category_size
        self.feature_above_zero = None
        self.feature_importance = None
        # add also training time
        self.attack_f1 = []
        self.attack_recall = []
        self.attack_precision = []
        self.cv_score_avg = []
        self.cv_score_std = []
        self.n_features = []
        self.attacks = []
        self.fit_time = []
        self.pred_time = []
        self.X_res = None
        self.y_res = None
        self.warmup = True # warmup boolean, this variable will be used to load in memory the function in order to have reliable time measures
        self.model = init(**params)
        self.cv = cv
        self.scaler = scaler
        self.params = params
        self.init = init
        self.test_kfold = [] 
        self.train_kfold = []

    def get_undersampled_ds(self):
        '''
        This function is tailormade for the dataset used in this project. It will undersample the dataset to the specified category size.
        It will return two dataset the first one with the X features and the second one with the y feature.
        '''
        sampling_weights = {'Background': self.category_size * 2, 'Benign': self.category_size * 2, 'XMRIGCC CryptoMiner': self.category_size, 'Probing': self.category_size, 'Bruteforce': self.category_size, 'Bruteforce-XML': self.category_size}



        rus = RandomUnderSampler(random_state=42, sampling_strategy=sampling_weights)
        self.X_res, self.y_res = rus.fit_resample(self.ds[x_features], self.ds.traffic_category)
        return self.X_res, self.y_res

    def stratified_under_sample(self, group: pd.DataFrame, k: int, random_state: int):
        print(f"Running the stratified {self.cv}-fold")
        # shuffle data
        group = group.sample(frac=1, random_state=random_state)
        
        # making a dictionary for checking if all the groups are equally insert into the array
        unique_categories = set(group)
        
        # getting the size of each category per fold
        folded_category = self.category_size // k
        
        for i in range(k):
            test_indexes = []
            train_indexes = []
            
            # iterating over the unique categories        
            for category in unique_categories:
                # making a window of data to retreive
                if (category == 'Background') | (category == 'Benign'):
                    start = (folded_category * 2) * i
                    stop = (folded_category * 2) * (i + 1)
                else:
                    start = folded_category * i
                    stop = folded_category * (i + 1)
                test_indexes.extend(group[group == category].iloc[start:stop].index)
            
            for x in group.index:
                if x not in test_indexes:
                    train_indexes.append(x)
                    
            # shuffling the data with the same seed in order to have the same result in both the dataset
            np.random.shuffle(test_indexes)
            self.test_kfold.append(test_indexes)
            np.random.shuffle(train_indexes)
            self.train_kfold.append(train_indexes)
        print("Test and Train k-fold created")
            
    def cross_validation(self, X, y, group):
        if self.test_kfold == [] or self.train_kfold == []:
            self.stratified_under_sample(group, self.cv, 42)

        # the neural network need to have the number of features in the input layer
        if "n_features" in self.params:
            self.params["n_features"] = X.shape[1]
            print(f"setting the first layer to {X.shape[1]} neurons")

        # the random forest need to have the number of features in the input layer
        if "max_features" in self.params:
            self.params["max_features"] = X.shape[1]

        cvscores = []
        print("Running the cross validation")
        start_cv = time.time()
        for test, train in zip(self.test_kfold, self.train_kfold):
            # the model need to be reinitialized every time otherwise it will use the same model for every fold
            self.model = self.init(**self.params)

            if self.scaler is not None:
                X_train = self.scaler.transform(X.loc[train])
                X_test = self.scaler.transform(X.loc[test])
            else:
                X_train = X.loc[train]
                X_test = X.loc[test]

            self.model.fit(X_train, y.loc[train])
            
            y_predicted = self.model.predict(X_test)

            cvscores.append(metrics.f1_score(y.loc[test], y_predicted))
        end_cv = time.time()    
        
        return np.mean(cvscores), np.std(cvscores), end_cv - start_cv

    def get_score(self, features):
        X = self.X_res[features]
        y = self.ds.loc[self.y_res.index].Label
        cv_mean, cv_std, cv_time = self.cross_validation(X, y, self.y_res)
        return cv_mean, cv_std, len(features), cv_time

    def recursive_reduction(self):
        scores = []
        score_std = []
        n_features = []
        cv_time = []

        if self.scaler is not None:
            self.scaler.fit(self.ds[['fwd_iat.tot']])
        # making a warm up run otherwise the first one will be always slower than the others
        # only one features so that it can be as fast as possibile
        self.get_score(['fwd_iat.tot'])

        if self.scaler is not None:
            self.scaler.fit(self.ds[self.feature_above_zero])
        print(f"testing with {len(self.feature_above_zero)} features")
        result = self.get_score(self.feature_above_zero)
        scores.append(result[0])
        score_std.append(result[1])
        n_features.append(result[2])    
        cv_time.append(result[3])
        
        for i in range(1,len(self.feature_above_zero)):
            print(f"testing with {len(self.feature_above_zero[:-i])} features")
            if self.scaler is not None:
                self.scaler.fit(self.ds[self.feature_above_zero[:-i]])
            result = self.get_score(self.feature_above_zero[:-i])
            scores.append(result[0])
            score_std.append(result[1])
            n_features.append(result[2])
            cv_time.append(result[3])
            
        return scores, score_std, n_features, cv_time

    # sistemare warmup

    def test_zero_day(self, attack, features, rus, rus_attack):
        print(f"training with {len(features)} features")

        X_res, y_res = rus.fit_resample(self.ds[features], self.ds.traffic_category)
        y_res = self.ds.loc[y_res.index].Label

        X_attack, y_attack = rus_attack.fit_resample(self.ds[features], self.ds.traffic_category)
        y_attack = self.ds.loc[y_attack.index].Label
            
        cv_mean, cv_std, cv_time = self.cross_validation(X_res, self.ds.loc[y_res.index].Label, y_res)
        
        if self.scaler is not None:
            X_res = self.scaler.transform(X_res)
            X_attack = self.scaler.transform(X_attack)

        if self.warmup:
            self.model.fit(X_res, y_res)
            self.model.predict(X_attack)
        
        start_fit = time.time()
        self.model.fit(X_res, y_res)
        end_fit = time.time()
        
        start_pred = time.time()
        y_predicted = self.model.predict(X_attack)
        end_pred = time.time()
        
        self.attack_f1.append(metrics.f1_score(y_attack, y_predicted))
        self.attack_recall.append(metrics.recall_score(y_attack, y_predicted))
        self.attack_precision.append(metrics.precision_score(y_attack, y_predicted))
        self.cv_score_avg.append(cv_mean)
        self.cv_score_std.append(cv_std)
        self.n_features.append(len(features))
        self.attacks.append(attack)
        self.fit_time.append(end_fit - start_fit)
        self.pred_time.append(end_pred - start_pred)

        # at the very first iteration we change it to false
        self.warmup = False

    # eseguire la cross validation una volta dopo aver fatto il fit di ciascun attacco

    def recursive_reduction_over_attack(self, attack): # aggiungere scaling
        # making a sample for having a 1:1 ration for positive and negative class
        # keep in mind that in the training I will have only three attacks, while for the test only one attack
        sampling_weights = {'Background': int(self.category_size * 1.5), 'Benign': int(self.category_size * 1.5), 'XMRIGCC CryptoMiner': self.category_size, 'Probing': self.category_size, 'Bruteforce': self.category_size, 'Bruteforce-XML': self.category_size}
        sampling_attack = {'Background': int(self.category_size * 0.5), 'Benign': int(self.category_size * 0.5), 'XMRIGCC CryptoMiner': 0, 'Probing': 0, 'Bruteforce': 0, 'Bruteforce-XML': 0}

        # removing all the attack observations
        sampling_weights[attack] = 0
        
        # doing the undersampling
        rus = RandomUnderSampler(random_state=42, sampling_strategy=sampling_weights)
        
        # adding the attack to the test dataset with the non attack traffic
        sampling_attack[attack] = self.category_size
        
        # making the dataset with only one attack
        rus_attack = RandomUnderSampler(random_state=42, sampling_strategy=sampling_attack)
        
        # running the warmup
        self.warmup = True

        # setting this two array to empty, in order to make another pair of test and train kfold with the new dataset
        self.test_kfold = []
        self.train_kfold = []

        if self.scaler is not None:
            self.scaler.fit(self.ds[self.feature_above_zero])
        self.test_zero_day(attack, self.feature_above_zero, rus, rus_attack)

        for i in range(1,len(self.feature_above_zero)):
            if self.scaler is not None:
                self.scaler.fit(self.ds[self.feature_above_zero[:-i]])
            self.test_zero_day(attack, self.feature_above_zero[:-i], rus, rus_attack)

    def run_zero_day_test(self):
        for attack in ['XMRIGCC CryptoMiner','Probing','Bruteforce','Bruteforce-XML']: # aggiungere un tracking del tempo speso
            self.warmup = True
            print('traing for ', attack)
            start_time = time.time()
            self.recursive_reduction_over_attack(attack)
            end_time = time.time()
            print('has taken ', end_time - start_time, ' seconds')

        return pd.DataFrame({'attack_f1': self.attack_f1,  'attack_recall': self.attack_recall, 'attack_precision': self.attack_precision, 'cv_score_avg': self.cv_score_avg, 'cv_score_std': self.cv_score_std, 'n_features': self.n_features, 'attack_name': self.attacks, 'fit_time': self.fit_time, 'pred_time': self.pred_time})