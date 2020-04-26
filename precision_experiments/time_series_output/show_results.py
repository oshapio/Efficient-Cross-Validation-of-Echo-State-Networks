import os
import pickle

import numpy as np


def read_folder(folder_path):
    results = {}
    for entry in os.scandir(folder_path):
        if os.path.isdir(entry.path) or entry.path.endswith("ini"): continue
        res = pickle.load(open(entry.path, "rb"))

        seed = int(entry.path.split("_")[-1][:-4])

        key = (res["validation_type"], res["final_model"])
        if key not in results: results[key] = {}
        results[key][seed] = res

    return results


folder = r"results"
results = read_folder(folder)

keys = sorted(list(results.keys()))
for idx, key in enumerate(keys):
    vals = results[key]

    # aggregate results of many seeds
    valid_NRMSEs, test_NRMSEs = [], []
    best_alphas, best_eigens, best_regs = [], [], []
    for seed, val in vals.items():
        valids, tests = [y["valid_nrmse"] for y in val["results"]], [y["test_nrmse"] for y in val["results"]]
        a, eigen, reg = [y["leaking_rate"] for y in val["results"]], [y["eigen_scaling"] for y in val["results"]], [
            y["reg_degree"] for y in val["results"]]

        best_valid_id = np.argmin(valids)

        best_alphas.append(a[best_valid_id])
        best_eigens.append(eigen[best_valid_id])
        best_regs.append(reg[best_valid_id])

        valid_NRMSEs.append(valids[best_valid_id])
        test_NRMSEs.append(tests[best_valid_id])

    mean_valid, mean_test, median_test = np.mean(valid_NRMSEs), np.mean(test_NRMSEs), np.median(test_NRMSEs)
    std_valid, std_test = np.std(valid_NRMSEs), np.std(test_NRMSEs)
    # print("{0: <50} Valid - {1: <30}, Test - {2: <30} ({3:})".format(str(key), round(mean_valid, 3), round(mean_test, 3),
    #                                                                 np.median(test_NRMSEs)
    #                                                                 ))
    print(
        "{} Valid - {} $\pm$ {}, Test - {} $\pm$ {} ".format(str(key), round(mean_valid, 3), round(std_valid, 3),
                                                             round(mean_test, 3), round(std_test, 3)
                                                             ))
    print("a - {}, eigen - {}, reg - {}".format(best_alphas[-1], best_eigens[-1], best_regs[-1]))
    lst = np.array([str(round(x, 3)) for x in test_NRMSEs]).astype(np.float32)
    # print(lst)

    if idx < len(keys) - 1:
        if (keys[idx + 1][1] != key[1]):
            print(
                "------------------------------------------------------------------------------------------------------------------------------------------")
        elif (keys[idx + 1][2] != key[2]):
            print()
