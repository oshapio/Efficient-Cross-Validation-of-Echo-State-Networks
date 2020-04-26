import sys

sys.path.append('../..')

import argparse
import copy
import os
import pickle
import time

import numpy as np
from scipy import linalg
from sklearn.utils import shuffle

import constants
from utils.data_reader import get_japanese_wovels
from utils.error_utils import get_nrmse
from utils.validation_utils import get_out_of_sample_fold_split, get_n_fold_splits

########################################################################################################################

parser = argparse.ArgumentParser(description="Run experiments for generative mode.")

parser.add_argument('--valid_method', default="standard", type=str)
parser.add_argument('--valid_implement', default="small_k", type=str, help="`naive` and `small_k` are supported")
parser.add_argument('--seed', default=0, type=int, help="Seed to use when generating random matrices")
parser.add_argument('--final_model', type=str, default="retrain", help="Final Wout weights computation method; "
                                                                       "`retrain`, `average`  options are supported")
parser.add_argument('--reg', type=str, default="default",
                    help="If `individ` is provided, individual regulzarion will be "
                         "used on every fold separetely, otherwise use `standard`")
args = parser.parse_args()

print(args)
########################################################################################################################

valid_method = args.valid_method

regularize_bias = True
conc_desc = {
    "full_name": "Japense Wovels",
    "skip_begin": 0,
    "folds_count": 18,
    "train_len": 270,
    "test_len": 370,

    "res_size": 50,
    "init_len": 0,

    "in_size": 12,
    "out_size": 9,
    "init_split": 0,

    "validation_leng": 15
}
########################################################################################################################
train_x, train_y, test_x, test_y = get_japanese_wovels()

np.random.seed(args.seed)


def unison_shuffled_copies(a, b):
    a, b = shuffle(a, b)
    return a, b


train_x, train_y = unison_shuffled_copies(train_x, train_y)
train_y = train_y.T
test_y = test_y.T

y_truth = np.argmax(test_y, axis=0)
########################################################################################################################

res_size, init_len = conc_desc["res_size"], conc_desc["init_len"]
in_size, out_size = conc_desc["in_size"], conc_desc["out_size"]

# subtract init len from trainlen
train_len, test_len = conc_desc["train_len"] - conc_desc["init_len"], conc_desc["test_len"]

validation_leng = conc_desc["validation_leng"]

################################################## Grid search ranges ##################################################
leaking_range = np.arange(0.1, 1, 0.1)
eigen_range = np.arange(0.1, 1.5, 0.1)
reg_range = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 0, 1]
########################################################################################################################
best_err = 1e9
# valid, test pairs
every_measur = []

best_valid_seq, best_test_seq = None, None
if valid_method == "cross":
    splits = get_n_fold_splits(train_len, folds_count=conc_desc["folds_count"],
                               validation_leng=conc_desc["validation_leng"])
elif valid_method == "standard":
    splits = get_out_of_sample_fold_split(train_len, validation_leng=conc_desc["validation_leng"])

eye = np.eye(1 + in_size + res_size)
if not regularize_bias:
    eye[:, 0] = 0

########################################################################################################################
dumpable_file_name = "{}/precision_experiments/time_series_classification/results/japwov_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(
    constants.GLOBAL_PATH, "japwovels", conc_desc["folds_count"], conc_desc["skip_begin"], args.final_model,
    valid_method,
    conc_desc["folds_count"], conc_desc["res_size"], conc_desc["validation_leng"], args.valid_implement, args.reg,
    args.seed)

if os.path.isfile(dumpable_file_name):
    print("File already exists. Exiting")
    exit(0)

########################################################################################################################
def evaluate_split(reg, split, valid_implement):
    if valid_implement == "naive":
        take_range = np.r_[split["train_range_start"][0]:split["train_range_start"][1],
                     split["train_range_end"][0]:split["train_range_end"][1]]

        X_this = X[:, take_range]

        X_this_T = X_this.T
        Yt_this = Yt[:, take_range]

        Wout = np.dot(np.dot(Yt_this, X_this_T), linalg.inv(np.dot(X_this, X_this_T) + \
                                                            reg * eye))
    elif valid_implement == "small_k":
        block_id = split["exclude_block"]

        XXT_sub = XXT_main - XXTs[block_id]
        YXT_sub = YXT_main - YXTs[block_id]

        Wout = np.dot(YXT_sub, linalg.inv(XXT_sub + reg * eye))

    X_this = Xs[split["exclude_block"]]
    Yt_this = Ys[split["exclude_block"]]

    preds = np.dot(Wout, X_this)

    nrmse = get_nrmse(Yt_this, preds)

    return Wout, nrmse

def test_model(Wout, reg, valid_NRMSE):
    global best_err, every_measur

    if args.final_model == "retrain":
        Wout_retrained = np.dot(YXT_main, linalg.inv(XXT_main + reg * eye))
        average_Wout = copy.deepcopy(Wout_retrained)
    else:
        average_Wout = copy.deepcopy(Wout)

    test_preds = np.dot(average_Wout, X_test)

    preds_maxs = np.argmax(test_preds, axis=0)
    hits = np.count_nonzero(preds_maxs == y_truth)

    nrmse_test = get_nrmse(Yt_test, test_preds)

    every_measur.append([valid_NRMSE, nrmse_test, y_truth.shape[0] - hits])

    if valid_NRMSE < best_err:
        best_err = valid_NRMSE
        print("valid_NRMSE - {}, test_NRMSE - {} test accuracy - {} ( {} / {} )".format(valid_NRMSE,
                                                                                        nrmse_test,
                                                                                        hits / y_truth.shape[0],
                                                                                        hits, y_truth.shape[0]))

########################################################################################################################
for eigen in eigen_range:
    print("eigen - {}".format(eigen))
    np.random.seed(args.seed)
    Win = (np.random.rand(res_size, 1 + in_size) - 0.5) * 1
    W = np.random.rand(res_size, res_size) - 0.5

    rhoW = max(abs(linalg.eig(W)[0]))
    W *= eigen / rhoW

    X = np.zeros((1 + in_size + res_size, train_len))
    Yt = train_y

    X_test = np.zeros((1 + in_size + res_size, test_len))
    Yt_test = test_y

    for a in leaking_range:
        # run the reservoir with the data and collect X
        for t in range(train_len):
            x = np.zeros((res_size, 1))
            for in_t in range(train_x[t].shape[0]):
                u = np.expand_dims(train_x[t][in_t, :], -1)
                x = (1 - a) * x + a * np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
            X[:, t] = np.vstack((1, u, x))[:, 0] # add only the last state to train state matrix

        X_T = X.T

        for t in range(test_len):
            x = np.zeros((res_size, 1))
            for in_t in range(test_x[t].shape[0]):
                u = np.expand_dims(test_x[t][in_t, :], -1)
                x = (1 - a) * x + a * np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))

            X_test[:, t] = np.vstack((1, u, x))[:, 0] # add only the last state to test state matrix

        # calculate XXT and XY main matrices
        XXT_main = np.dot(X, X_T)
        YXT_main = np.dot(Yt, X_T)

        ######################## Save collected X, X.X^T, YX matrices for each validation block ########################
        Xs, XXTs, YXTs, Ys = {}, {}, {}, {}

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
        ################################################################################################################

        if args.reg == "default":
            for reg in reg_range:
                average_valid_nrmse = 0
                average_Wout = np.zeros((out_size, 1 + res_size + in_size))

                for split in splits:  # doing the N-fold CV
                    Wout, nrmse = evaluate_split(reg, split, args.valid_implement)

                    average_Wout += Wout
                    average_valid_nrmse += nrmse

                average_valid_nrmse /= len(splits)
                average_Wout /= len(splits)  # average Wout among folds

                test_model(average_Wout, reg, average_valid_nrmse)

        elif args.reg == "individ":
            average_valid_nrmse = 0, 0
            average_Wout = np.zeros((out_size, 1 + res_size + in_size))

            best_regs_avg = 0
            for split in splits:
                best_Wout = np.zeros((out_size, 1 + res_size + in_size))
                best_reg, best_nrmse = 0, 1e9

                for reg in reg_range:
                    Wout, nrmse = evaluate_split(reg, split, args.valid_implement)

                    if best_nrmse > nrmse:
                        best_reg = reg
                        best_nrmse = nrmse
                        best_Wout = copy.deepcopy(Wout)

                best_regs_avg += best_reg
                average_Wout += best_Wout
                average_valid_nrmse += best_nrmse

            best_regs_avg /= len(splits)
            average_valid_nrmse /= len(splits)
            average_Wout /= len(splits)  # average Wout among folds

            test_model(average_Wout, best_regs_avg, average_valid_nrmse)

every_measur = np.array(every_measur)

pickle.dump({
    "dataset_name": "japwovels",
    "validation_type": valid_method,
    "validation_method": valid_method,
    "folds_mapper": conc_desc,
    "valid_NRMSEs": every_measur[:, 0],
    "test_NRMSEs": every_measur[:, 1],
    "misclassifications": every_measur[:, 2],
    "datetime": time.ctime(),
    "valid_implement": args.valid_implement,
    "final_model": args.final_model,
    "reg_type": args.reg,
    "seed": args.seed,
    "regularize_bias": regularize_bias
}, open(dumpable_file_name, "wb"))
