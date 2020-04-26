import glob
import os
import pickle
import numpy as np

def read_folder(folder_path):
    results = {}
    for entry in os.scandir(folder_path):
        if os.path.isdir(entry.path) or entry.path.endswith("ini"): continue
        res = pickle.load(open(entry.path, "rb"))

        key = (("small_k" if "valid_implement" not in res else res["valid_implement"]), res["dataset_name"], res["validation_type"], res["final_model"])
        if key not in results: results[key] = {}
        results[key][res["seed"]] = res

    return results
folder = r"results"
results = read_folder(folder)

keys = sorted(list(results.keys()))
for idx, key in enumerate(keys):
    vals = results[key]

    # aggregate results of many seeds
    valid_NRMSEs, test_NRMSEs = [], []
    for seed, val in vals.items():
        best_valid_id = np.argmin(val["results_valid_nrmse"])

        best_test = val["results_test_nrmse"][best_valid_id]
        best_valid = val["results_valid_nrmse"][best_valid_id]

        valid_NRMSEs.append(best_valid)
        test_NRMSEs.append(best_test)

    mean_valid, mean_test, median_test = np.mean(valid_NRMSEs), np.mean(test_NRMSEs), np.median(test_NRMSEs)
    print("{0: <50} Valid - {1: <30}, Test - {2: <30}".format(str(key), round(mean_valid, 3), round(mean_test, 3)))

    if idx<len(keys)-1:
        if(keys[idx+1][1] != key[1]):
           print("------------------------------------------------------------------------------------------------------------------------------------------")
        elif(keys[idx+1][2] != key[2]):
            print()
