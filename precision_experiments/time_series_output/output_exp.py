import sys
sys.path.append('../..')

import constants
import argparse
import copy
import os
import pickle
import time
import numpy as np
from scipy import linalg

from utils.data_reader import get_mitdb
from utils.error_utils import get_nrmse
from utils.validation_utils import get_n_fold_splits, get_n_widening_fold_splits, get_out_of_sample_fold_split, \
    get_n_walk_step_walking_splits

regularize_bias = False

###############
parser = argparse.ArgumentParser(description="Run experiments for series output mode")
parser.add_argument('--valid_method', default="cross", type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--final_model', type=str, default="retrain", help="Final Wout weights computation method; "
                                                                       "`retrain`, `average` and `best` options are supported")
args = parser.parse_args()
################

print(args)

valid_method = args.valid_method

conc_desc = \
    {
        "full_name": "MITDB",
        "folds_count": 10,
        "train_len": 455000,
        "test_len": 190000,
        "res_size": 50,

        "init_len": 5000,
        "in_size": 1,
        "out_size": 1,
        "init_split": 225000,

        "validation_leng": 45000,
        "validation_leng_standard_cv": 45000,
        "validation_leng_forward": 45000,
        "validation_leng_oos": 45000
    }

data = get_mitdb(read_names=["100"])


res_size, init_len = conc_desc["res_size"], conc_desc["init_len"]
in_size, out_size = conc_desc["in_size"], conc_desc["out_size"]

# subtract init len from trainlen
train_len, test_len = conc_desc["train_len"] - conc_desc["init_len"], conc_desc["test_len"]

best_err = 1e9
################################################## Grid search ranges ##################################################
leaking_ranges = np.linspace(0.1, 1, 4, endpoint=True)
eigen_ranges = np.arange(0.1, 1.5, 0.15)
reg_ranges = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 0, 1]
################################z########################################################################################

# valid, test pairs
every_measur = []

if valid_method == "cross":
    splits = get_n_fold_splits(train_len, folds_count=conc_desc["folds_count"],
                               validation_leng=conc_desc["validation_leng_standard_cv"])
elif valid_method == "accum":
    splits = get_n_widening_fold_splits(train_len, folds_count=conc_desc["folds_count"],
                                        initial_length=conc_desc["init_split"],
                                        validation_leng=conc_desc["validation_leng_forward"])
elif valid_method == "standard":
    splits = get_out_of_sample_fold_split(train_len, validation_leng=conc_desc["validation_leng_oos"])

elif valid_method == "walk_forward":
    splits = get_n_walk_step_walking_splits(train_len, folds_count=conc_desc["folds_count"],
                                            train_slider_leng=conc_desc["init_split"],
                                            validation_leng=conc_desc["validation_leng_forward"])

########################################################################################################################
Yt, Yt_test = [], []
for i in range(init_len, train_len + init_len):
    Yt.append(data[0][i, 3])
for i in range(train_len + init_len, train_len + init_len + test_len):
    Yt_test.append(data[0][i, 3])

Yt, Yt_test = np.expand_dims(np.array(Yt), 0), np.expand_dims(np.array(Yt_test), 0)
###############################################################

dumpable_file_name = "{}/precision_experiments/time_series_output/results/mitdb_{}_{}_{}_{}_{}_{}_{}.pkl".format(
    constants.GLOBAL_PATH, "MITDB",
    conc_desc["folds_count"],
    args.final_model, valid_method,
    conc_desc["folds_count"],
    conc_desc["res_size"],
    args.seed)

if os.path.isfile(dumpable_file_name):
    print("File {} exists, exiting.".format(dumpable_file_name))
    exit(0)

eye = np.eye(1 + in_size + res_size)
if not regularize_bias:
    eye[:, 0] = 0

X = np.zeros((1 + in_size + res_size, train_len))
X_test = np.zeros((1 + in_size + res_size, test_len))

for eigen in eigen_ranges:
    print("eigen - {}".format(eigen))
    np.random.seed(args.seed)
    Win = (np.random.rand(res_size, 1 + in_size) - 0.5) * 1
    W = np.random.rand(res_size, res_size) - 0.5

    rhoW = max(abs(linalg.eig(W)[0]))
    W *= eigen / rhoW

    for a in leaking_ranges:
        # run the reservoir with the data and collect X
        curr = 0

        # save collected X, X.X^T, YX matrices for each block, also, save "final" XXT and YX
        Xs, XXTs, YXTs, Ys = {}, {}, {}, {}

        x = np.zeros((res_size, 1))
        for t in range(0, init_len + train_len):
            u = np.expand_dims(data[0][t, 1], -1)
            x = (1 - a) * x + a * np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))

            if t >= init_len:
                X[:, curr] = np.vstack((1, u, x))[:, 0]
                curr += 1
        X_T = X.T

        curr = 0
        for t in range(init_len + train_len, init_len + train_len + test_len):
            u = np.expand_dims(data[0][t, 1], -1)
            x = (1 - a) * x + a * np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
            X_test[:, curr] = np.vstack((1, u, x))[:, 0]
            curr += 1

        X_T_test = X_test.T

        # calculate XXT and XY main matrices
        XXT_main = np.dot(X, X_T)
        YXT_main = np.dot(Yt, X_T)

        for split in splits:
            block_id = split["exclude_block"]
            excluded_range = np.r_[split["exclude_range"][0]:split["exclude_range"][1],
                             split["exclude_range_end"][0]:split["exclude_range_end"][1]]

            mini_X = X[:, excluded_range]
            mini_Y = Yt[:, excluded_range]

            XXT = np.dot(mini_X, mini_X.T)
            YX = np.dot(mini_Y, mini_X.T)

            # validation_range == excluded_range only if standard or cross validation is used
            validation_range = np.r_[split["exclude_range"][0]:split["exclude_range"][1]]

            Xs[block_id] = X[:, validation_range]
            Ys[block_id] = Yt[:, validation_range]

            XXTs[block_id] = XXT
            YXTs[block_id] = YX

        for reg in reg_ranges:
            average_valid_mse, average_valid_nrmse  = 0, 0

            average_Wout = np.zeros((out_size, 1 + res_size + in_size))
            best_Wout = np.zeros((out_size, 1 + res_size + in_size))

            best_valid_rmse, best_valid_nrmse = 1e9, 1e9

            for split in splits:  # doing the N-fold CV
                block_id = split["exclude_block"]

                XXT_sub = XXT_main - XXTs[block_id]
                YXT_sub = YXT_main - YXTs[block_id]

                Wout = np.dot(YXT_sub, linalg.inv(XXT_sub + reg * eye))

                average_Wout += Wout

                X_this = Xs[block_id]
                Yt_this = Ys[block_id]

                preds = np.dot(Wout, X_this)

                preds[preds < 0], preds[preds > 1] = 0, 1

                rmse, nrmse = np.sqrt(np.mean(np.square(preds - Yt_this))), get_nrmse(Yt_this, preds)

                if rmse < best_valid_rmse:
                    best_valid_rmse = rmse
                    best_Wout = copy.deepcopy(Wout)

                average_valid_mse += rmse
                average_valid_nrmse += nrmse

            average_valid_mse /= len(splits)
            average_valid_nrmse /= len(splits)

            average_Wout /= len(splits)  # average Wout among folds

            if args.final_model == "retrain":
                Wout_retrained = np.dot(YXT_main, linalg.inv(XXT_main + reg * eye))
                average_Wout = copy.deepcopy(Wout_retrained)
            elif args.final_model == "best":
                average_Wout = copy.deepcopy(best_Wout)
            # otherwise, we use the average of all Wouts

            test_preds = np.dot(average_Wout, X_test)
            test_preds[test_preds < 0], test_preds[test_preds > 1] = 0, 1

            rmse_test = np.sqrt(np.mean(np.square(test_preds - Yt_test)))
            nrmse_test = get_nrmse(Yt_test, test_preds)

            every_measur.append({
                "valid_rmse": average_valid_mse,
                "valid_nrmse": average_valid_nrmse,
                "test_rmse": rmse_test,
                "test_nrmse": nrmse_test,
                "leaking_rate": a,
                "eigen_scaling": eigen,
                "reg_degree": reg,
            })

            if average_valid_mse.flatten() < best_err:
                best_err = average_valid_mse
                oa, oeigen, oreg = a, eigen, reg
                print("Better params found! a - {}, eigen - {}, reg - {}".format(a, eigen, reg))
                print("valid_MSE - {}, test_MSE - {})".format(average_valid_mse, rmse_test))
                print("valid_NRMSE - {}, test_NRMSE - {})".format(average_valid_nrmse, nrmse_test))

pickle.dump({
    "dataset_name": "MITDB",
    "validation_type": valid_method,
    "validation_method": valid_method,
    "folds_mapper": conc_desc,
    "results": every_measur,
    "datetime": time.ctime(),
    "final_model": args.final_model,
    "regularize_bias": regularize_bias,
    "seed":args.seed
}, open(dumpable_file_name, "wb"))
print("Done")
