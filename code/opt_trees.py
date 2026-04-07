import pandas as pd
import os
import numpy as np
import sys
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from math import sqrt
import time
import argparse



def prepare_data_df(data, charge, feature_code):
    if charge == 'anion':
        data_df = data.drop(columns = [
            'Cation Dihedral Angle', 'Mean Dihedral Angle_c', 'Delta Dihedral Angle_c', 
            'Cation Bond Length', 'Mean Bond Length_c', 'Delta Bond Length_c', 'DFT Cation λ'
        ])
    
    if charge == 'cation':
        data_df = data.drop(columns = [
            'Anion Dihedral Angle', 'Mean Dihedral Angle_a', 'Delta Dihedral Angle_a', 
            'Anion Bond Length', 'Mean Bond Length_a', 'Delta Bond Length_a', 'DFT Anion λ'
        ])
    
    feature_abbrev_dict = {
        'psa': 'PiSystemArea',
        'psl': 'PiSystemLength',
        'pss': 'PiSystemSize',
        'means': 'Mean',
        'deltas': 'Delta',
        'angles': 'Angle',
        'lengths': 'Bond Length',
        'ecfp': 'ECFP',
        'rdkfp': 'RDKFP'
    }
    
    charge_lambda = f"DFT {charge.capitalize()} λ"
    if feature_code != 'all-features':
        if feature_code == 'only-ecfp':
            data_df = data_df[['Monomer 1', 'Monomer 2', 'Oligomer Length'] + [f"ECFP_{i}" for i in range(2048)] + [charge_lambda]]
        elif 'only' in feature_code:
            columns = ['Monomer 1', 'Monomer 2', 'Oligomer Length']
            selected_fts = feature_code.split('-')[1:]
            for fs in selected_fts:
                selected_column = feature_abbrev_dict.get(fs)
                selected_column_names = [c for c in data.columns if selected_column in c]
                columns = columns + selected_column_names
            data_df = data_df[columns + [charge_lambda]]
        else:
            feature_selection = feature_code.split('-')[1:]
            for fs in feature_selection:
                column_name = feature_abbrev_dict.get(fs)
                data_df.drop(list(data_df.filter(regex = column_name)), axis = 1, inplace = True)

    data_df = data_df[data_df[f"{charge_lambda}"] > 0].reset_index(drop = False)
    data_df = data_df[data_df[f"{charge_lambda}"] < 2].reset_index(drop = False)

    print(list(dict.fromkeys(x for x in data_df.columns)))
    return data_df, charge_lambda



def optimize_trees(data_df, charge_lambda, rand_state):
    X = data_df.drop(columns = [charge_lambda])
    y = data_df[charge_lambda]
 
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.15, random_state = rand_state)
    n_cols = len(X.columns)

    results_lst = []

    for trees in [10,25,50,75,100,125,150,175,200,500,750,1000,1250,1500]:
        result_dict = {}
        regr = RandomForestRegressor(n_estimators = trees, max_features = None, min_samples_split=2, random_state=0, verbose = 0)
        start_time = time.time() 
        regr.fit(train_X, train_y)
        end_time = time.time()
        runtime = end_time - start_time
        score = regr.score(test_X, test_y)

        test_y_predictions = np.asarray(regr.predict(test_X))
        pdf = test_X[['Monomer 1','Monomer 2','Oligomer Length']].copy(deep=True)
        pdf[f"DFT {charge_lambda}"] = np.asarray(test_y)
        pdf[f"Trained {charge_lambda}"] =  [x for x in list(test_y_predictions)]

        slope, intercept, r_value, p_value, std_err = linregress(pdf[f"Trained {charge_lambda}"].astype(float),pdf[f"DFT {charge_lambda}"])
        rmse = sqrt(mean_squared_error(pdf[f"Trained {charge_lambda}"],pdf[f"DFT {charge_lambda}"]))

        result_dict['RandomState'] = rand_state
        result_dict['Trees'] = trees
        result_dict['Score'] = score
        result_dict['TotalRMSE'] = rmse
        result_dict['TotalR2'] = r_value ** 2
        result_dict['Runtime'] = runtime

        results_lst.append(result_dict)
        
    return results_lst



def main(charge, feature_code, rand_state, fingerprint):
    print(f"{charge=}, {rand_state=}, {fingerprint=}")
    rand_state = int(rand_state)
    data = pd.read_csv(f'/ix/ghutchison/kwl20/reorganization/csvs/final/gfn_geo_dft_energy_{fingerprint}.csv')
    data_df, charge_lambda = prepare_data_df(data, charge, feature_code)
    results_lst = optimize_trees(data_df, charge_lambda, rand_state)

    results_df = pd.DataFrame(results_lst)
    results_df.to_csv(f"tts_15/RF_optimization_{charge}_{feature_code}_{rand_state}_{fingerprint}.csv", index = False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("charge")
    parser.add_argument("feature_code")
    parser.add_argument("rand_state")
    parser.add_argument("fingerprint")
    args = parser.parse_args()
    main(args.charge, args.feature_code, args.rand_state, args.fingerprint)