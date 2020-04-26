import sys
sys.path.append('../..')
import argparse
import os
import pickle
import time
import numpy as np
import copy
import pandas as pd
from scipy import linalg
import constants

from utils.error_utils import get_rmse, get_nrmse
from utils.validation_utils import get_n_fold_splits, get_n_widening_fold_splits, get_out_of_sample_fold_split, \
    get_n_walk_step_walking_splits

parser = argparse.ArgumentParser(description="Run experiments for generative mode.")

parser.add_argument('--dataset', type=str, default="electr",
                    help="Dataset to use. `unempl`, `gasoline`, `electr`, `sunspots` options" \
                         "are available")
parser.add_argument('--valid_method', default="accum", type=str)
parser.add_argument('--valid_implement', default="small_k", type=str, help="`naive` and `small_k` are supported")
parser.add_argument('--seed', default=0, type=int, help="Seed to use when generating random matrices")
parser.add_argument('--final_model', type=str, default="retrain", help="Final Wout weights computation method; "
                                                                       "`retrain`, `average` and `best` options are supported")

args = parser.parse_args()
valid_method = args.valid_method

regularize_bias = True
dataset_name = args.dataset

name_mapper = {
    "unempl":
        {  # total samples 360
            "full_name": "Unemployment Rate In America",
            "skip_begin": 0,
            "folds_count": 34,
            "train_len": 349,
            "test_len": 10,  # 54,
            "res_size": 50,
            "init_len": 9,  # 18, # transient
            "in_size": 1,
            "out_size": 1,
            "init_split_forward": 170,
            "validation_leng_forward": 10,
            "validation_leng_standard_cv": 10,  # 26,
            "validation_leng_oos": 10,
        },
    "gasoline":
        {  # total samples 1355, weekly
            "full_name": "US finished motor gasoline product supplied",
            "skip_begin": 0,
            "folds_count": 18,
            "train_len": 1287,
            "test_len": 67,
            "res_size": 50,
            "init_len": 81,  # transient
            "in_size": 1,
            "out_size": 1,
            "init_split_forward": 575,
            "validation_leng_forward": 67,
            "validation_leng_standard_cv": 67,
            "validation_leng_oos": 67
        },
    "electr":
        {  # total samples 4033
            "full_name": "Electricity Demand 30min sampling",
            "skip_begin": 0,
            "folds_count": 18,
            "train_len": 3830,
            "test_len": 201,
            "res_size": 50,
            "init_len": 213,  # transient
            "in_size": 1,
            "out_size": 1,
            "init_split_forward": 1915,
            "validation_leng_forward": 201,
            "validation_leng_standard_cv": 201,
            "validation_leng_oos": 201,
        },
    "sunspots":
        {  # total samples 1848
            "full_name": "Monthly sunspots data",
            "skip_begin": 0,
            "folds_count": 10,
            "train_len": 2000,
            "test_len": 200,  # 54,
            "res_size": 50,
            "init_len": 200,  # 18, # transient
            "in_size": 1,
            "out_size": 1,
            "init_split_forward": 1000,
            "validation_leng_forward": 200,
            "validation_leng_standard_cv": 200,  # 26,
            "validation_leng_oos": 200
        }
}
if dataset_name == "unempl":
    data = pd.read_csv("{}/data/unemployment.txt".format(constants.GLOBAL_PATH), sep=" ", header=None).values[0]
elif dataset_name == "gasoline":
    data = pd.read_csv("{}/data/gasoline.csv".format(constants.GLOBAL_PATH)).values[:, 1]
elif dataset_name == "electr":
    data = pd.read_csv("{}/data/electricity.csv".format(constants.GLOBAL_PATH)).values[:, 1].astype(np.float)
elif dataset_name == "sunspots":
    data = pd.read_csv("{}/data/sunspot.month.csv".format(constants.GLOBAL_PATH)).values[:, 2]

max_val, min_val = np.max(data), np.min(data)
data = (data - min_val) / (max_val - min_val)

conc_desc = name_mapper[dataset_name]

res_size, init_len = conc_desc["res_size"], conc_desc["init_len"]
in_size, out_size = conc_desc["in_size"], conc_desc["out_size"]

# subtract init len from trainlen
train_len, test_len = conc_desc["train_len"] - conc_desc["init_len"], conc_desc["test_len"]

################################################## Grid search ranges ##################################################
leaking_ranges = np.arange(0.1, 1, 0.1)
eigen_ranges = np.arange(0, 1.5, 0.1)
reg_ranges = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 0, 1, 10, 100, 1000]
########################################################################################################################

# valid, test pairs
every_measur = []

if valid_method == "cross":
    splits = get_n_fold_splits(train_len, folds_count=conc_desc["folds_count"],
                               validation_leng=conc_desc["validation_leng_standard_cv"])
elif valid_method == "accum":
    splits = get_n_widening_fold_splits(train_len, initial_length=conc_desc["init_split_forward"],
                                        folds_count=conc_desc["folds_count"],
                                        validation_leng=conc_desc["validation_leng_forward"])
elif valid_method == "standard":
    splits = get_out_of_sample_fold_split(train_len, validation_leng=conc_desc["validation_leng_oos"])

elif valid_method == "walk_forward":
    splits = get_n_walk_step_walking_splits(train_len, train_slider_leng=conc_desc["init_split_forward"],
                                            folds_count=conc_desc["folds_count"],
                                            validation_leng=conc_desc["validation_leng_forward"])

eye = np.eye(1 + in_size + res_size)
if not regularize_bias:
    eye[:, 0] = 0

dumpable_file_name = "{}/precision_experiments/generative_mode/results/FastCV_{}_{}_{}_{}_{}_{}_{}.pkl".format(
    constants.GLOBAL_PATH, dataset_name, conc_desc["folds_count"], conc_desc["skip_begin"], valid_method,
    args.final_model, args.valid_implement, args.seed)

if os.path.isfile(dumpable_file_name):
    print("File already exists. Exiting.")
    exit(0)

test_preds = np.zeros((out_size, test_len))

best_err = 1e9
for eigen in eigen_ranges:
    print("eigen - {}".format(eigen))

    np.random.seed(args.seed)
    Win = (np.random.rand(res_size, 1 + in_size) - 0.5) * 1
    W = np.random.rand(res_size, res_size) - 0.5

    rhoW = max(abs(linalg.eig(W)[0]))
    W *= eigen / rhoW

    # allocated memory for the design (collected states) matrix
    X = np.zeros((1 + in_size + res_size, train_len))
    # set the corresponding target matrix directly
    Yt = data[None, init_len + 1:train_len + init_len + 1]

    for a in leaking_ranges:
        # run the reservoir with the data and collect X
        x = np.zeros((res_size, 1))
        init_x = np.zeros((res_size, 1))

        ############################################## Collect states ################################################3#
        for t in range(init_len + train_len):
            u = data[t]
            x = (1 - a) * x + a * np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
            if t >= init_len:
                X[:, t - init_len] = np.vstack((1, u, x))[:, 0]
            if t == init_len - 1:
                init_x = copy.deepcopy(x)
        ################################################################################################################
        X_T = X.T

        if args.valid_implement == "small_k":
            XXT = np.dot(X, X_T)
            Ytarg_XT = np.dot(Yt, X_T)

            XiXti_mats = []  # list holding X_i . X_i^T matrices.
            Ytargi_XiT_mats = []  # list holiding Y_targ_i . X_i^T matrices

            for split in splits:
                idxes = np.r_[split["exclude_train_range_start"][0]: split["exclude_train_range_start"][1],
                        split["exclude_train_range_end"][0]: split["exclude_train_range_end"][1]]

                X_i = X[:, idxes]
                X_iT = X_i.T

                XiXti_mats.append(np.dot(X_i, X_iT))

                Y_targ_i = Yt[:, idxes]
                Ytargi_XiT_mats.append(np.dot(Y_targ_i, X_iT))

        for reg in reg_ranges:
            average_valid_mse = 0
            average_valid_nrmse = 0
            best_valid_mse = 1e9

            average_Wout = np.zeros((out_size, 1 + res_size + in_size))
            best_Wout = np.zeros((out_size, 1 + res_size + in_size))

            for split_id, split in enumerate(splits):  # doing the N-fold CV

                if args.valid_implement == "small_k":
                    leftover_Yt_XT = Ytarg_XT - Ytargi_XiT_mats[split_id]

                    leftover_XXt = XXT - XiXti_mats[split_id]

                    inversed = linalg.inv(leftover_XXt + reg * eye)

                    Wout = np.dot(leftover_Yt_XT, inversed)
                elif args.valid_implement == "naive":
                    take_range = np.r_[split["train_range_start"][0]:split["train_range_start"][1],
                                 split["train_range_end"][0]:split["train_range_end"][1]]

                    X_this = X[:, take_range]
                    X_this_T = X_this.T
                    Yt_this = Yt[:, take_range]

                    Wout = np.dot(np.dot(Yt_this, X_this_T), linalg.inv(np.dot(X_this, X_this_T) + \
                                                                        reg * eye))

                average_Wout += Wout

                if split["exclude_range"][0] - 1 <= 0:
                    x = copy.deepcopy(init_x)
                else:
                    x = np.expand_dims(X[1 + in_size:, split["exclude_range"][0] - 1], 1)

                ########################### Validate on excluded fold => generative mode #######################
                valid_preds = np.zeros(
                    (out_size, split["exclude_range"][1] - split["exclude_range"][0]))  # non inclusive
                u = data[init_len + split["exclude_range"][0]]

                for t in range(valid_preds.shape[1]):
                    x = (1 - a) * x + a * np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
                    y = np.dot(Wout, np.vstack((1, u, x)))
                    valid_preds[:, t] = y
                    u = y

                valid_rmse = get_rmse(
                    data[init_len + split["exclude_range"][0] + 1:init_len + split["exclude_range"][1] + 1] * (
                            max_val - min_val) + min_val,
                    valid_preds * (max_val - min_val) + min_val)

                valid_nrmse = get_nrmse(data[init_len + split["exclude_range"][0] + 1:init_len + split["exclude_range"][
                    1] + 1].flatten() * (max_val - min_val) + min_val,
                                        valid_preds.flatten() * (max_val - min_val) + min_val, use_axis=0)

                average_valid_mse += valid_rmse
                average_valid_nrmse += valid_nrmse

                if valid_rmse < best_valid_mse:
                    best_valid_mse = valid_rmse
                    best_Wout = copy.deepcopy(Wout)

                ########################################################################################################

            average_valid_mse /= len(splits)
            average_valid_nrmse /= len(splits)

            average_Wout /= len(splits)  # average Wout among folds

            if args.final_model == "retrain":
                Wout_retrained = np.dot(np.dot(Yt, X_T), linalg.inv(np.dot(X, X_T) + \
                                                                    reg * eye))
                average_Wout = copy.deepcopy(Wout_retrained)
            elif args.final_model == "best":
                average_Wout = copy.deepcopy(best_Wout)

            ############################################# Testing ######################################################
            u = data[init_len + train_len]
            x = np.expand_dims(X[1 + in_size:, train_len - 1], 1)

            for t in range(test_len):
                x = (1 - a) * x + a * np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
                y = np.dot(average_Wout, np.vstack((1, u, x)))
                test_preds[:, t] = y
                u = y
            test_mse = get_rmse(
                data[init_len + train_len + 1:init_len + train_len + test_len + 1] * (max_val - min_val) + min_val,
                test_preds * (max_val - min_val) + min_val)
            test_nrmse = get_nrmse(
                data[init_len + train_len + 1:init_len + train_len + test_len + 1] * (max_val - min_val) + min_val,
                test_preds * (max_val - min_val) + min_val, use_axis=0)
            ############################################################################################################

            every_measur.append([average_valid_mse, test_mse, average_valid_nrmse, test_nrmse, a, eigen, reg])

            if average_valid_mse.flatten() < best_err:
                # best_valid_seq = valid_preds
                best_test_seq = test_preds

                best_err = average_valid_mse
                oa, oeigen, oreg = a, eigen, reg
                print("Better params found! a - {}, eigen - {}, reg - {}".format(a, eigen, reg))
                print("valid_MSE - {}, test_MSE - {}".format(average_valid_mse, test_mse))
                print("valid_NRMSE - {}, test_NRMSE - {}".format(average_valid_nrmse, test_nrmse))

every_measur = np.array(every_measur)

# save it
pickle.dump({
    "dataset_name": dataset_name,
    "validation_type": valid_method,
    "validation_method": valid_method,
    "folds_mapper": conc_desc,
    "results_valid": every_measur[:, 0],
    "results_test": every_measur[:, 1],
    "results_valid_nrmse": every_measur[:, 2],
    "results_test_nrmse": every_measur[:, 3],
    "leakings": every_measur[:, 4],
    "eigens": every_measur[:, 5],
    "valid_implement": args.valid_implement,
    "regs": every_measur[:, 6],
    "final_model": args.final_model,
    "datetime": time.ctime(),
    "seed": args.seed,
    "regularize_bias": regularize_bias
}, open(dumpable_file_name, "wb"))
