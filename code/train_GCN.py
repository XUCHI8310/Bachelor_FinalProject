import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
# from torch_geometric.nn import GCNConv
# from torch_scatter import scatter_max
import matplotlib.pyplot as plt
import model_GCN
# import preprocess_GCN as pgcn


def load_mnist_graph(data_size=60000):
    data_list = []
    labels = np.load("../data/labels.npy")

    # Load by numpy
    for i in range(data_size):
        edge = torch.tensor(np.load('../data/graph/' + str(i) + '.npy').T, dtype=torch.long)    # Transposed matrix
        x = torch.tensor(np.load('../data/node_features/' + str(i) + '.npy') / 28, dtype=torch.float)

        d = Data(x=x, edge_index=edge.contiguous(), t=int(labels[i]))
        data_list.append(d)
        if i % 1000 == 999:
            print("\rData loaded " + str(i + 1), end="  ")

    print("Data read complete.")
    return data_list


def load_mnist_graph_test(test_size=10000):
    data_list = []
    labels = np.load("../data/test_labels.npy")

    # Load by numpy
    for i in range(test_size):
        edge = torch.tensor(np.load('../data/test_graph/' + str(i) + '.npy').T, dtype=torch.long)
        x = torch.tensor(np.load('../data/test_node_features/' + str(i) + '.npy') / 28, dtype=torch.float)

        d = Data(x=x, edge_index=edge.contiguous(), t=int(labels[i]))
        data_list.append(d)
        if i % 1000 == 999:
            print("\rData loaded " + str(i + 1), end="  ")

    print("Data read complete.")
    return data_list


def train_GCN():
    data_size = 60000
    train_size = 60000
    test_size = 10000
    batch_size = 100
    epoch_num = 200

    img_train_acc = []
    img_train_loss = []
    img_test_acc = []
    img_test_loss = []
    # Preparation
    device = torch.device('cuda')
    model = model_GCN.Net().to(device)
    optimizer = torch.optim.Adam(model.parameters())  # Adam optimizer
    trainset = load_mnist_graph(data_size=data_size)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = load_mnist_graph_test(test_size=test_size)
    testloader = DataLoader(testset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()  # Cross entropy

    print("Start Train")

    # Training
    model.train()
    for epoch in range(epoch_num):
        train_loss = 0.0
        train_acc = 0.0
        train_total = 0
        train_batch_num = 0
        train_correct = 0
        for i, batch in enumerate(trainloader):
            batch = batch.to("cuda")    # CUDA
            optimizer.zero_grad()   # Set grad to zero, avoid grads auto adding
            outputs = model(batch)  # Train
            loss = criterion(outputs, batch.t)  # batch.t : label
            loss.backward()  # Backward
            optimizer.step()    # Update params

            _, pred = torch.max(outputs, 1)  # Get prediction label
            train_total += batch.t.size(0)
            train_batch_num += 1
            train_correct += (pred == batch.t).sum().cpu().item()
            train_acc = float(train_correct / train_total)
            train_loss += loss.cpu().item()
            if i % 10 == 9:
                progress_bar = '[' + ('=' * ((i + 1) // 10)) + (' ' * ((train_size // 100 - (i + 1)) // 10)) + ']'
                print('\repoch: {:d} train loss: {:.3f} train acc: {:.3f} % {}'
                      .format(epoch + 1, train_loss / (train_size / batch_size), 100 * train_acc, progress_bar), end="  ")

        end = ' ' * max(1, (train_size // 1000 - 39)) + "\n"
        print('\rFinished {:d} epoch. Train loss: {:.3f} Train acc: {:.3f} %'
              .format(epoch + 1, loss.cpu().item(), 100 * train_acc), end=end)
        img_train_acc = np.append(img_train_acc, train_acc)
        img_train_loss = np.append(img_train_loss, train_loss / (train_size / batch_size))

        correct = 0
        total = 0
        batch_num = 0
        loss = 0
        with torch.no_grad():
            for data in testloader:
                data = data.to(device)
                outputs = model(data)
                loss += criterion(outputs, data.t)
                _, pred = torch.max(outputs, 1)
                total += data.t.size(0)
                batch_num += 1
                correct += (pred == data.t).sum().cpu().item()

        end = ' ' * max(1, (train_size // 1000 - 39)) + "\n"
        test_acc = float(correct / total)
        test_loss = loss.cpu().item() / batch_num
        print('Test Accuracy: {:.2f} %'.format(100 * test_acc), end='  ')
        print(f'Test Loss: {test_loss:.3f}', end=end)
        img_test_acc = np.append(img_test_acc, test_acc)
        img_test_loss = np.append(img_test_loss, test_loss)

    torch.save(model, 'gcn_2.pt')

    fig = plt.figure()
    plt.suptitle('Prediction result', fontsize=16)
    X = np.linspace(1, epoch_num, epoch_num)
    line1, = plt.plot(X, img_test_acc, color="green", linewidth="1.0", linestyle="-")
    line2, = plt.plot(X, img_test_loss, color="blue", linewidth="1.0", linestyle="-")
    # Comma: UserWarning: Legend does not support [<matplotlib.lines.Line2D object at 0x0000025D8AF17FA0>] instances.
    # A proxy artist may be used instead.
    plt.text(1, img_test_acc[1], "(1,"+str(img_test_acc[0])+")")
    plt.text(20, img_test_acc[19], "(20," + str(img_test_acc[19]) + ")")
    plt.text(50, img_test_acc[49], "(50," + str(img_test_acc[49]) + ")")
    plt.text(100, img_test_acc[99], "(100," + str(img_test_acc[99]) + ")")
    plt.text(200, img_test_acc[199], "(200," + str(img_test_acc[199]) + ")")
    plt.text(1, img_test_loss[1], "(1,"+str(img_test_loss[0])+")")
    plt.text(20, img_test_loss[19], "(20," + str(img_test_loss[19]) + ")")
    plt.text(50, img_test_loss[49], "(50," + str(img_test_loss[49]) + ")")
    plt.text(100, img_test_loss[99], "(100," + str(img_test_loss[99]) + ")")
    plt.text(200, img_test_loss[199], "(200," + str(img_test_loss[199]) + ")")
    plt.legend(handles=[line1, line2], labels=["accuracy", "loss"], loc="upper right", fontsize=6)
    plt.savefig('../result/GCN_test_2.png')
    # plt.show()

    fig_train = plt.figure()
    plt.suptitle('Train result', fontsize=16)
    X = np.linspace(1, epoch_num, epoch_num)
    line1, = plt.plot(X, img_train_acc, color="green", linewidth="1.0", linestyle="-")
    line2, = plt.plot(X, img_train_loss, color="blue", linewidth="1.0", linestyle="-")
    plt.text(1, img_train_acc[1], "(1," + str(img_train_acc[0]) + ")")
    plt.text(20, img_train_acc[19], "(20," + str(img_train_acc[19]) + ")")
    plt.text(50, img_train_acc[49], "(50," + str(img_train_acc[49]) + ")")
    plt.text(100, img_train_acc[99], "(100," + str(img_train_acc[99]) + ")")
    plt.text(200, img_train_acc[199], "(200," + str(img_train_acc[199]) + ")")
    plt.text(1, img_train_loss[1], "(1," + str(img_train_loss[0]) + ")")
    plt.text(20, img_train_loss[19], "(20," + str(img_train_loss[19]) + ")")
    plt.text(50, img_train_loss[49], "(50," + str(img_train_loss[49]) + ")")
    plt.text(100, img_train_loss[99], "(100," + str(img_train_loss[99]) + ")")
    plt.text(200, img_train_loss[199], "(200," + str(img_train_loss[199]) + ")")
    plt.legend(handles=[line1, line2], labels=["accuracy", "loss"], loc="upper right", fontsize=6)
    plt.savefig('../result/GCN_train.png')
    # plt.show()

    print('Finished Training.')

    # Result
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            data = data.to(device)
            outputs = model(data)
            _, pred = torch.max(outputs, 1)
            total += data.t.size(0)
            correct += (pred == data.t).sum().cpu().item()

    print('Final test accuracy: {:.2f} %'.format(100 * float(correct / total)))


# if __name__ == "__main__":
#     train()
