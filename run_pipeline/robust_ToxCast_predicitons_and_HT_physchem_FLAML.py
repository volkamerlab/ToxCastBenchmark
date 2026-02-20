import os
import subprocess
import glob
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from flaml import AutoML
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold



def custom_mcc(
    X_val,
    y_val,
    estimator,
    labels,
    X_train,
    y_train,
    weight_val=None,
    weight_train=None,
    *args,
):
    from sklearn.metrics import matthews_corrcoef
    import time

    start = time.time()
    y_pred = estimator.predict(X_val)
    pred_time = (time.time() - start) / len(X_val)
    val_loss = 1-matthews_corrcoef(y_val, y_pred)
    y_pred = estimator.predict(X_train)
    train_loss = 1-matthews_corrcoef(y_train, y_pred )
    return val_loss, {
        "val_loss": val_loss,
        "train_loss": train_loss,
        "pred_time": pred_time,
    }



def main():
    files = {'physchem': '//home/lisa-marie-rolli/gestagen_local_lisa/lisa-marie.rolli/comptox_benchmark//automatic_ToxCast_Query/physchem_properties.csv',
             'maccs': '//home/lisa-marie-rolli/gestagen_local_lisa/lisa-marie.rolli/comptox_benchmark//automatic_ToxCast_Query/maccs.csv', 'morgan': '//home/lisa-marie-rolli/gestagen_local_lisa/lisa-marie.rolli/comptox_benchmark//automatic_ToxCast_Query/morgan.csv'}
    feature_types = ['physchem']

    

    task = 'classification'
    num_features = 100
    for feature_type in feature_types:
        for fs_name in ['pca', 'mrmr', 'mi', 'variance']:

            subfolders = ['androgens', 'estrogens', 'glucocorticoids', 'progestagens', 'steroidal']
            for subfolder in subfolders:

                directory = f'//home/lisa-marie-rolli/gestagen_local_lisa/lisa-marie.rolli/comptox_benchmark//ToxCast_Assays_Endpoint_Results/{subfolder}/'

                for content in os.listdir(directory):
                    if '.csv' in content:
                        continue
                    print(content)
                    path_to_CV_folds = f'{directory}/{content}/'
                    pattern = f'{content}-*_binary_response.csv'

                    matching_files = glob.glob(f'{directory}/{pattern}')

                    if matching_files:
                        response = matching_files[0]
                    else:
                        break
                    final_models = [False for _ in range(5)]
                    binary_response_df = pd.read_csv(response, sep = '\t')
                    binary_response_df.set_index('compound', drop  =True, inplace = True)
                    
                    for fold in range(5):
                        if not fs_name == 'pca':
                            feature_matrix_path = files[feature_type]
                        else:
                            feature_matrix_path = f'{path_to_CV_folds}/fold{fold}/{feature_type}_transformed_matrix_pca.csv'

                        feature_matrix = pd.read_csv(feature_matrix_path, sep ='\t')
                        feature_matrix.set_index('compound', drop = True, inplace = True)
                        training_samples_path = f'{path_to_CV_folds}/fold{fold}/train.txt'
                        with open(training_samples_path) as training_samples_file:
                            training_samples = training_samples_file.read().splitlines()
                        y_train = binary_response_df.loc[training_samples, :].values
                        
                        features_path = f'{path_to_CV_folds}/fold{fold}/{feature_type}_feature_names_{fs_name}.txt'
                        with open(features_path) as features_file:
                            features = features_file.read().splitlines()
                        
                        
                        X_train = feature_matrix.loc[training_samples, features]
                        standardizer = StandardScaler()
                        automl = AutoML()

                        automl_pipeline = Pipeline(
                            [("standardizer", standardizer), ("automl", automl)]
                        )
                        automl.fit(X_train=X_train, y_train=y_train, eval_method = "cv", split_type = StratifiedKFold(n_splits=5,shuffle=True, random_state=42), metric = custom_mcc, time_budget = 15*60, task = 'classification', estimator_list = ["rf","catboost", "svc"], n_jobs = -1)
                        test_samples_path = f'{path_to_CV_folds}/fold{fold}/test.txt'
                        with open(test_samples_path) as test_samples_file:
                            test_samples = test_samples_file.read().splitlines()
                        X_test = feature_matrix.loc[test_samples, features]
                        print(automl.best_config_per_estimator)
                        pred = automl.predict(X_test)
                        print(pred)
                        return
if __name__ == '__main__':
    main()
    
