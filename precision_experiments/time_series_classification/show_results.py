import os
import pickle

import numpy as np

def read_folder(folder_path):
    results = {}
    for entry in os.scandir(folder_path):
        if os.path.isdir(entry.path) or entry.path.endswith("ini"): continue
        res = pickle.load(open(entry.path, "rb"))

        key = (("small_k" if "valid_implement" not in res else res["valid_implement"]), res["validation_type"],
               res["final_model"], res["reg_type"])
        if key not in results: results[key] = {}
        results[key][res["seed"]] = res

    return results

folder = r"results"
results = read_folder(folder)

keys = sorted(list(results.keys()))
for idx, key in enumerate(keys):
    vals = results[key]

    # aggregate results of many seeds
    valid_NRMSEs, test_NRMSEs, misclasses = [], [], []
    for seed, val in vals.items():
        best_valid_id = np.argmin(val["valid_NRMSEs"])

        best_test = val["test_NRMSEs"][best_valid_id]
        best_valid = val["valid_NRMSEs"][best_valid_id]
        misses = val["misclassifications"][best_valid_id]

        misclasses.append(misses)
        valid_NRMSEs.append(best_valid)
        test_NRMSEs.append(best_test)

    mean_valid, mean_test, misses_test = np.mean(valid_NRMSEs), np.mean(test_NRMSEs), np.mean(misclasses)
    std_valid, std_test, std_misses = np.std(valid_NRMSEs), np.std(test_NRMSEs), np.std(misclasses)


    # print("{0: <50} Valid - {1: <30}, Test - {2: <30}, Misses - {3:} ".format(
    #     str(key), round(mean_valid, 3), round(mean_test, 3),  round(misses_test, 3) ))
    print(len(valid_NRMSEs))
    print("{} => valid - {} $\pm$ {}, test - {} $\pm$ {}, misclass - {} $\pm$ {}".format(str(key), round(mean_valid, 3), round(std_valid, 3),
          round(mean_test, 3), round(std_test, 3), round(misses_test, 3), round(std_misses, 3)) )
    if idx < len(keys) - 1:
        if (keys[idx + 1][1] != key[1]):
            print(
                "------------------------------------------------------------------------------------------------------------------------------------------")
