import sys
import json
from numpy.random import seed
from PCA import Principal_Component_Analysis
from Correlation import Correlation
from MRMR import MRMR
from Variance import Variance_FS
from SelectKBestMI import Select_MI


def decide_model(data_json_dict):
    '''
        @param data_json_dict: the json dict given in the config file
        @return the initialized NN given this json dict

        '''
    model_name = data_json_dict["fs_name"]
    if model_name == "pca":

        return Principal_Component_Analysis(data_json_dict)

    elif model_name == "corr":

        return Correlation(data_json_dict)

    elif model_name == "mrmr":
        return MRMR(data_json_dict)

    elif model_name == 'variance':
        return Variance_FS(data_json_dict)
    elif model_name == 'mi':
        return Select_MI(data_json_dict)
    else:
        print("The model " + model_name + " is not supported")
        raise NotImplementedError


def check_parameters(data_json_dict):
    '''
        @param data_json_dict: the json dict given in the config file
        @return list consisting of two elements, first element: information on whether the checks were successful, second element: error messages

        '''
    if not "fs_name" in data_json_dict:

        return [False, "Parameter fs_name was not given"]

    if not "output_dir" in data_json_dict:

        return [False, "Parameter output_dir was not given"]

    if not "features" in data_json_dict:

        return [False, "Parameter features was not given"]

    if not "num_features" in data_json_dict:

        return [False, "Parameter num_features was not given"]
    if not "samples" in data_json_dict:
        return [False, "Parameter samples was not given"]

    assert data_json_dict["fs_name"] != "", "Empty string as model_name"
    assert data_json_dict["num_features"] != "", "Empty string as training_samples"
    assert data_json_dict["output_dir"] != "", "Empty string as output_dir"
    assert data_json_dict["features"] != "", "Empty string as features"
    assert data_json_dict["samples"] != "", "Empty string as samples"
    return [True, ""]


def start_any_method(data_json_dict):

    checking_result = check_parameters(data_json_dict)
    if not checking_result[0]:

        sys.exit("Your parameters were invalid " + checking_result[1])

    my_seed = 42
    seed(my_seed)
    fs_method = decide_model(data_json_dict)
    fs_method.select_features()

    return


################ Main function ##################################

def main(config_filename):

    # print(config_filename)
    json_file = open(config_filename)

    data = json.load(json_file)

    start_any_method(data)

    json_file.close()


# Start of program
if __name__ == "__main__":
    # print(sys.argv)
    if len(sys.argv) != 2:
        sys.exit("This program needs the following arguments:\
                    \n- Config json file with information about \
                    \n a) the neural network model to be used (parameter: model_name)\
                    \n b) path to a file with training samples (parameter: training_samples)\
                    \n c) path to a file with test samples (parameter: test_samples)\
                    \n d) path to a file with drug list(parameter: drug_list)\
                    \n e) output directory for results (parameter: output_dir)\
                    \n f) desired level of randomness (parameter: rand)\
                    \n g) various model specific parameters (parameter: model_specific, is a model-specific dictionary)")

    main(sys.argv[1])
