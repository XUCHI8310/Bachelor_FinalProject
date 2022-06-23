import torch
from torch import nn
from torch_geometric.data import Data, DataLoader
import numpy as np
import matplotlib.pyplot as plt


test_size = 10000
batch_size = 100
epoch_num = 200
train_size = 60000


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


def test():
    testset = load_mnist_graph_test(test_size=test_size)
    testloader = DataLoader(testset, batch_size=batch_size)

    device = torch.device('cuda')
    model = torch.load('gcn_2.pt')
    # optimizer = torch.optim.Adam(model.parameters())  # Adam optimizer
    criterion = nn.CrossEntropyLoss()  # Cross entropy
    model.eval()

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
    # img_test_acc = np.append(img_test_acc, test_acc)
    # img_test_loss = np.append(img_test_loss, test_loss)

    # fig = plt.figure()
    # plt.suptitle('Prediction result', fontsize=16)
    # X = np.linspace(1, epoch_num, epoch_num)
    # line1, = plt.plot(X, img_test_acc, color="green", linewidth="1.0", linestyle="-")
    # line2, = plt.plot(X, img_test_loss, color="blue", linewidth="1.0", linestyle="-")
    # # Comma: UserWarning: Legend does not support [<matplotlib.lines.Line2D object at 0x0000025D8AF17FA0>] instances.
    # # A proxy artist may be used instead.
    # plt.text(1, img_test_acc[1], "(1,"+str(img_test_acc[0])+")")
    # plt.text(20, img_test_acc[19], "(20," + str(img_test_acc[19]) + ")")
    # plt.text(50, img_test_acc[49], "(50," + str(img_test_acc[49]) + ")")
    # plt.text(100, img_test_acc[99], "(100," + str(img_test_acc[99]) + ")")
    # plt.text(200, img_test_acc[199], "(200," + str(img_test_acc[199]) + ")")
    # plt.text(1, img_test_loss[1], "(1,"+str(img_test_loss[0])+")")
    # plt.text(20, img_test_loss[19], "(20," + str(img_test_loss[19]) + ")")
    # plt.text(50, img_test_loss[49], "(50," + str(img_test_loss[49]) + ")")
    # plt.text(100, img_test_loss[99], "(100," + str(img_test_loss[99]) + ")")
    # plt.text(200, img_test_loss[199], "(200," + str(img_test_loss[199]) + ")")
    # plt.legend(handles=[line1, line2], labels=["accuracy", "loss"], loc="upper right", fontsize=6)
    # plt.savefig('../result/GCN_test.png')
    # plt.show()

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
#     test()
