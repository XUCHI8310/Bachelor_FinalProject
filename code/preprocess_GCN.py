import numpy as np
import pandas as pd
from tqdm import *


def load_dataset(path, filename):
    tempPath = path + filename  # Define path
    tempData = pd.read_csv(tempPath, header=None)  # pandas read csv
    label = tempData.iloc[:, 0]  # Read 0th column
    value = tempData.iloc[:, 1:]  # Read other data as value
    return label, value


def make_graph():
    # Return labels
    def labels():
        temp_train_label, temp_train_value = load_dataset('../data/', 'mnist_train.csv')
        train_label = np.array(temp_train_label)  # label in tensor type
        np.save("../data/labels.npy", train_label)
        return train_label

    # Return values
    def values():
        temp_train_label, temp_train_value = load_dataset('../data/', 'mnist_train.csv')
        train_value = np.array(temp_train_value).reshape([60000, 28, 28])  # value in tensor type
        return train_value

    labels = labels()
    values = values()
    # print(np.size(labels))
    # print(np.shape(values))
    # plt.imshow(train_value[0], cmap='gray')
    # plt.title('%i' % train_label[0])
    # plt.show()
    data = np.where(values < 102, -1, 1000)  # Binary. High->1000, low->-1
    dim1, dim2, dim3 = data.shape  # Take dim1=60000 as total num
    with tqdm(total=dim1) as pbar:  # Progress bar
        pbar.set_description('Processing:')

        for e, imgtemp in enumerate(data):
            img = np.pad(imgtemp, [(2, 2), (2, 2)], "constant", constant_values=(-1))  # Padding: to protect image edge info
            count = 0

            #  Numbering nodes in each graph
            for i in range(2, 30):
                for j in range(2, 30):
                    if img[i][j] == 1000:
                        img[i][j] = count  # Replace pixel by node number
                        count += 1

            edges = []
            # y & x coordinates
            coord = np.zeros((count, 2))

            for i in range(2, 30):  # 2 to 30 after padding
                for j in range(2, 30):
                    if img[i][j] == -1:
                        continue  # Skip

                    # get neighbor value
                    filter0 = img[i - 2:i + 3, j - 2:j + 3].flatten()   # 3*3 data
                    filter1 = filter0[[6, 7, 8, 11, 13, 16, 17, 18]]    # Choose neighbor values

                    coord[filter0[12]][0] = i - 2
                    coord[filter0[12]][1] = j - 2

                    for tmp in filter1:
                        if not tmp == -1:
                            edges.append([filter0[12], tmp])

            np.save("../data/graph/" + str(e), edges)
            np.save("../data/node_features/" + str(e), coord)
            pbar.update(1)


def make_graph_test():
    def labels():
        temp_test_label, temp_test_value = load_dataset('../data/', 'mnist_test.csv')
        test_label = np.array(temp_test_label)  # label in tensor type
        np.save("../data/test_labels.npy", test_label)
        return test_label

    # Return values
    def values():
        temp_test_label, temp_test_value = load_dataset('../data/', 'mnist_test.csv')
        test_value = np.array(temp_test_value).reshape([10000, 28, 28])  # value in tensor type
        return test_value

    labels = labels()
    values = values()
    # print(np.size(labels))
    # print(np.shape(values))
    # plt.imshow(train_value[0], cmap='gray')
    # plt.title('%i' % train_label[0])
    # plt.show()
    data = np.where(values < 102, -1, 1000)  # Binary
    dim1, dim2, dim3 = data.shape  # Take dim1=60000 as total num
    with tqdm(total=dim1) as pbar:  # Progress bar
        pbar.set_description('Processing')

        for e, imgtemp in enumerate(data):
            img = np.pad(imgtemp, [(2, 2), (2, 2)], "constant", constant_values=(-1))  # Padding: to protect image edge info
            count = 0

            #  Numbering nodes in each graph
            for i in range(2, 30):
                for j in range(2, 30):
                    if img[i][j] == 1000:
                        img[i][j] = count  # Replace pixel by node number
                        count += 1

            edges = []
            # y & x coordinates
            coord = np.zeros((count, 2))

            for i in range(2, 30):  # 2 to 30 after padding
                for j in range(2, 30):
                    if img[i][j] == -1:
                        continue  # Skip

                    # get neighbor value
                    filter0 = img[i - 2:i + 3, j - 2:j + 3].flatten()
                    filter1 = filter0[[6, 7, 8, 11, 13, 16, 17, 18]]

                    # Record original coordinates
                    coord[filter0[12]][0] = i - 2
                    coord[filter0[12]][1] = j - 2

                    for tmp in filter1:
                        if not tmp == -1:   # if center's neighbor is a node, add the edge
                            edges.append([filter0[12], tmp])

            np.save("../data/test_graph/" + str(e), edges)
            np.save("../data/test_node_features/" + str(e), coord)
            pbar.update(1)
