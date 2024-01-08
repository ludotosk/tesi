from sklearnex import patch_sklearn
patch_sklearn(global_patch=True)
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from utils import *

def main():
    '''
    Whit this function I will run once everything untill the zero test, then I will save the class in a pickle file.
    '''
    ds = pd.concat([
        pd.read_csv('../datasets/HIKARI-2021/ALLFLOWMETER_HIKARI2021.csv', dtype=dtype_hikari, usecols=selected_features_hikari),
        pd.read_csv('../datasets/HIKARI-2021/ALLFLOWMETER_HIKARI2022.csv', dtype=dtype_hikari, usecols=selected_features_hikari)], 
        ignore_index=True)

    params = {
        "n_jobs": -1,
        "n_neighbors": 5
    }

    scaler = StandardScaler()
    scaler.fit(ds[x_features])

    pv = DataPreprocessingAndValidation(ds, 7988, 5, KNeighborsClassifier, params, scaler=scaler)
    pv.filename = 'knn_zero_day.csv'

    # since after a while the knn make a segmentation fault, we need to run once the undersampling in order to save it and then use the same data again
    pv.stratified_under_sample(ds.traffic_category, 5, 12)

    print("loading exp")
    with open('knn_data/exp.pickle', 'rb') as handle:
        exp = pickle.load(handle)

    pv.feature_importance = pd.DataFrame(data={'importance' : exp.abs.mean(0).values, 'feature': x_features})
    pv.feature_importance.sort_values('importance',ascending=False).head(20)

    pv.feature_above_zero = pv.feature_importance.query('importance > 0').sort_values('importance',ascending=False)['feature'].to_list()

    with open('knn_utils.pickle', 'wb') as handle:
        pickle.dump(pv, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run_zero_day():
    '''
    Loading the pv class in order to have the indexes of the cross validation already there.
    '''
    with open('knn_utils.pickle', 'rb') as handle:
        pv = pickle.load(handle)
        
    zero_day_feature_reduction_scores = pv.run_zero_day_test()
    with open('knn_data/zero_day_feature_reduction_scores.pickle', 'wb') as handle:
            pickle.dump(zero_day_feature_reduction_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)