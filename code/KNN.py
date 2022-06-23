import numpy as np
# import matplotlib.pyplot as plt
# import torch
from torch.utils.data import DataLoader
from torchvision import datasets  # , transforms



def train_KNN():
    def get_mean(x_train):
        x_train = np.reshape(x_train, (x_train.shape[0], -1))  # 28*28 -> 1*764
        mean_image = np.mean(x_train, axis=0)  # average
        return mean_image

    def centralized(x, mean):
        x = np.reshape(x, (x.shape[0], -1))
        x = x.astype(np.float)
        x -= mean
        return x

    class Knn:
        def __init__(self):
            pass

        def param_pass(self, X_train, y_train):
            self.Xtr = X_train
            self.ytr = y_train

        def predict(self, k, X_test):
            num_test = X_test.shape[0]
            label_list = []
            for i in range(num_test):
                # Euclidean
                distances = np.sqrt(np.sum(((self.Xtr - np.tile(X_test[i],
                                                                (self.Xtr.shape[0], 1)))) ** 2, axis=1))
                sorted_dist = np.argsort(distances)  # Sort
                nearest_k = sorted_dist[:k]
                class_count = {}  # Init dictionary
                for j in nearest_k:
                    class_count[self.ytr[j]] = class_count.get(self.ytr[j], 0) + 1  # Count
                sorted_list = sorted(class_count.items(), key=lambda elem: elem[1], reverse=True)
                label_list.append(sorted_list[0][0])
            return np.array(label_list)

    batch_size = 100
    trainset = datasets.MNIST(root='../data',
                                    train=True,
                                    transform=None,
                                    download=False)
    testset = datasets.MNIST(root='../data',
                                   train=False,
                                   transform=None,
                                   download=False)

    # Load data
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    # Preprocess
    x_train = train_loader.dataset.data.numpy()
    # normalize
    mean_image = get_mean(x_train)
    x_train = centralized(x_train, mean_image)
    y_train = train_loader.dataset.targets.numpy()  # Label
    # Choose k test data, 10000 test data means waiting for a long time.
    num_test = 200
    x_test = test_loader.dataset.data[:num_test].numpy()
    mean_image = get_mean(x_test)
    x_test = centralized(x_test, mean_image)
    y_test = test_loader.dataset.targets[:num_test].numpy()

    print("train_data:", x_train.shape)
    print("train_label:", len(y_train))
    print("test_data:", x_test.shape)
    print("test_labels:", len(y_test))

    for k in range(3, 6):  # Choose different k values
        classifier = Knn()
        classifier.param_pass(x_train, y_train)
        y_pred = classifier.predict(k, x_test)
        num_correct = np.sum(y_pred == y_test)
        accuracy = float(num_correct) / num_test
        print('Got %d / %d correct when k= %d => accuracy: %f' % (num_correct, num_test, k, accuracy))


if __name__ == "__main__":
    train_KNN()
