This repository contains code and data for our comprehensive benchmarking study of machine learning (ML) pipelines (i.e., combinations of compound representation, dimension reduction, and ML model) on 52 hormone receptor activity assays from the ToxCast dataset.

Our repo contains our scripts for the following parts:

## 1. Data processing
* To reproduce our splits, you have to run the ```run_datasail.py``` script in ```data_processing``` folder. You can set one value for ```epsilon``` and ```delta``` via the command line. We first ran datasail with ```epsilon = delta = 0.05```, which is also the default of datasail. Then, we increased them to ```0.1```, ```0.2```, and ```0.3```. Note that we only ran datasail with relaxed constraints for assays that were not feasible with lower ```epsilon``` and ```delta``` values.
* To generate a dummy dimension reduction (DR) file for the "no DR" experiments, you can run ```generate_feature_name_list_no_DR.py``` in ```data_processing```.
* ```data_processing``` also contains a script to generate the embeddings


## 2. Dimension Reduction
* The dimension reduction (DR) methds are implemented in the ```dr_methods``` folder. It contains a ```main_dimension_reduction.py```, which can be called to run any of the implemented DR methods given a config file. To generate config files, we provide a bash script called ```json_config_generator.sh```.
* In ```run_pipeline```, we provide two python scripts ```execute_DR_on_ToxCast_Downloads.py``` and ```execute_DR_on_embeddings.py```, that run the dimension reduction for all train test splits and all feature types considered in our study.

## 3. Models
* In ```run_pipeline```, we provide two python scripts ```robust_ToxCast_predicitons_and_HT.py``` and ```execute_DR_on_embeddings.py```, that run all models with all DR methods for a compound representation given via the command line. Notably, for ```morgan``` and ```embeddings```, we could not run TabPFN without DR, as we use the version that only supports 500 features.