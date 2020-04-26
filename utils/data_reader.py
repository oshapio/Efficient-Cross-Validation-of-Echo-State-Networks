import os
import pickle

import numpy as np
import pandas as pd

import constants


def get_mitdb(read_names=["100"]):
    data = []

    pickled_set_path = "{}/data/time_series_output/pickled.pkl".format(constants.GLOBAL_PATH)
    if os.path.isfile(pickled_set_path):
        print("File is cached.")
        return pickle.load(open(pickled_set_path, "rb"))

    for name in read_names:
        main_name = "{}/data/time_series_output/{}.csv".format(constants.GLOBAL_PATH, name)
        annot_name = "{}/data/time_series_output/{}annotations.txt".format(constants.GLOBAL_PATH, name)

        csved_main = pd.read_csv(main_name)

        diffs = csved_main.values[1:, 0] - csved_main.values[0:-1, 0]
        assert np.min(diffs) == 1 and np.max(diffs) == 1, "Error! Dataset is corrupted"

        csved_main.columns = ['sample_id', 'MLII', 'Vx']

        csved_main["pattern"] = 0

        values_array = csved_main.values.astype(np.float)

        # normalize what matters
        min_val, max_val = np.min(values_array[:, 1]), np.max(values_array[:, 1])
        values_array[:, 1] = (values_array[:, 1] - min_val) / (max_val - min_val)

        crs = open(annot_name, "r")

        neigh = 10
        for column in (raw.strip().split() for raw in crs):
            if column[2] == "N":
                for j in range(-neigh, neigh + 1):
                    if j == 0 or int(column[1]) + j >= values_array.shape[0] or int(column[1]) + j < 0:
                        continue
                    values_array[int(column[1]) + j, 3] = 1 - np.abs((j / neigh))
                values_array[int(column[1]), 3] = 1
        data.append(values_array)
    # save it
    print("Saving file..")
    pickle.dump(data, open(pickled_set_path, "wb"))
    print("File saved.")
    return data


def get_japanese_wovels():
    path_train = "{}/data/time_series_classification/ae.train.txt".format(constants.GLOBAL_PATH)
    path_test = "{}/data/time_series_classification/ae.test.txt".format(constants.GLOBAL_PATH)

    path_pickled = "{}/data/time_series_classification/ae.pkl".format(constants.GLOBAL_PATH)

    if os.path.isfile(path_pickled):
        return pickle.load(open(path_pickled, "rb"))

    def read_chunks(name, block_sizes):
        with open(name) as f:
            content = f.readlines()

        data = []
        labels = []

        curr_batch = []

        curr_block = 0
        curr_block_count = 0
        for i in range(len(content)):
            if content[i] == "\n":
                data.append(curr_batch)
                curr_batch = []

                if curr_block_count >= block_sizes[curr_block]:
                    curr_block_count = 0
                    curr_block += 1
                curr_block_count += 1
                labels.append(curr_block)
            else:
                curr_batch.append([float(word.replace("\n", "")) for word in content[i].split(' ')[:-1]])
        return data, labels

    test_blocks = [31, 35, 88, 44, 29, 24, 40, 50, 29]

    train_data, train_labels = read_chunks(path_train, [30 for i in range(9)])
    test_data, test_labels = read_chunks(path_test, test_blocks)

    def get_minmax(array):
        mins, maxs= 1e9, -1e9

        for i in range(len(array)):
            for j in range(len(array[i])):
                for k in range(len(array[i][j])):
                    mins = min(mins, array[i][j][k])
                    maxs = max(maxs, array[i][j][k])
        return mins, maxs

    # normalize train, test and encode labels
    min1, max1 = get_minmax(train_data)
    min2, max2 = get_minmax(test_data)

    set_min, set_max = min(min1, min2), max(max1, max2)

    def indices_to_one_hot(data, nb_classes):
        targets = np.array(data)
        return np.eye(nb_classes)[targets]

    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            for k in range(len(train_data[i][j])):
                train_data[i][j][k] = (train_data[i][j][k] - set_min) / (set_max - set_min)

    for i in range(len(test_data)):
        for j in range(len(test_data[i])):
            for k in range(len(test_data[i][j])):
                test_data[i][j][k] = (test_data[i][j][k] - set_min) / (set_max - set_min)

    train_data, test_data = [np.array(curr) for curr in train_data], [np.array(curr) for curr in test_data]

    train_labels, test_labels = indices_to_one_hot(train_labels, 9), indices_to_one_hot(test_labels, 9)

    pickle.dump([train_data, train_labels, test_data, test_labels], open(path_pickled, "wb"))

    return train_data, train_labels, test_data, test_labels
