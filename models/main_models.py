
import sys
import json
from numpy.random import seed
from Random_Forest import Random_Forest
from Support_Vector_Machine import Support_Vector_Machine
from Multi_Layer_Perceptron import MLP
from CatBoost import Cat_Boost
#from TabPFN import TabPFN
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def decide_model(data_json_dict):
    '''
            @param data_json_dict: the json dict given in the config file
            @return the initialized NN given this json dict

            '''
    model_name = data_json_dict["model_name"]
    if model_name == "rf":

        return Random_Forest(data_json_dict)
    elif model_name == 'svm':
        return Support_Vector_Machine(data_json_dict)
    elif model_name == 'mlp':
        return MLP(data_json_dict)
    elif model_name == 'cat_boost':
        return Cat_Boost(data_json_dict)
    #elif model_name == 'tabpfn':
    #    return TabPFN(data_json_dict)
    else:
        print("The model " + model_name + " is not supported")
        raise NotImplementedError


def check_parameters(data_json_dict):
    '''
            @param data_json_dict: the json dict given in the config file
            @return list consisting of two elements, first element: information on whether the checks were successful, second element: error messages

            '''
    if not "model_name" in data_json_dict:

        return [False, "Parameter model_name was not given"]

    if not "training_samples" in data_json_dict:

        return [False, "Parameter training_samples was not given"]

    if not "test_samples" in data_json_dict:

        return [False, "Parameter test_samples was not given"]

    if not "output_dir" in data_json_dict:

        return [False, "Parameter output_dir was not given"]

    if not "rand" in data_json_dict:

        return [False, "Parameter rand was not given"]

    if not "task" in data_json_dict:

        return [False, "Parameter task was not given"]

    if not "model_specific" in data_json_dict:

        return [False, "Parameter model_specific was not given"]
    if not "features" in data_json_dict:

        return [False, "Parameter features was not given"]

    if not "analysis_name" in data_json_dict:

        return [False, "Parameter analysis_name was not given"]

    assert data_json_dict["model_name"] != "", "Empty string as model_name"
    assert data_json_dict["training_samples"] != "", "Empty string as training_samples"
    # assert data_json_dict["test_samples"]!="", "Empty string as test_samples" if test_samples is empty string, training_samples will be used
    assert data_json_dict["output_dir"] != "", "Empty string as output_dir"
    assert data_json_dict["rand"] != "", "Empty string as rand"
    assert data_json_dict["features"] != "", "Empty string as features"
    assert data_json_dict["task"] in [
        'classification', 'regression'], "invalid input for task"

    return [True, ""]


def start_any_method(data_json_dict):

    checking_result = check_parameters(data_json_dict)
    if not checking_result[0]:

        sys.exit("Your parameters were invalid " + checking_result[1])

    my_seed = 42
    seed(my_seed)
    nn_model = decide_model(data_json_dict)
    nn_model.fit()
    nn_model.predict()

    return


################ Main function ##################################

def main(config_filename):

    # print(config_filename)
    json_file = open(config_filename)
    try:
        data = json.load(json_file)
    except:
        print(json_file)
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
