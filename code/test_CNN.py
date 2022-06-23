import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


def test_CNN():
    # Define param
    batch_size = 100  # Batch size. epoch = 10000/batch_size = 100

    # Dataset
    test_dataset = datasets.MNIST(root='../data', train=False, transform=transforms.ToTensor(), download=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = torch.load('cnn_2.pt')
    criterion = nn.CrossEntropyLoss()
    model.eval()
    eval_acc = 0
    eval_loss = 0
    img_acc = []
    img_loss = []

    if torch.cuda.is_available():
        print("CUDA available.")
        model = model.cuda()

    # Test
    for data in test_loader:
        img, label = data
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)

        _, pred = torch.max(out, 1)  # Get predicted label
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
        print('Test Loss: {:.6f}   ,   Acc: {:.6f}'.format(eval_loss / (len(test_dataset)), eval_acc / (len(test_dataset))))
        img_acc = np.append(img_acc, eval_acc / (len(test_dataset)))
        img_loss = np.append(img_loss, eval_loss / (len(test_dataset)))


    plt.figure()
    plt.suptitle('Test result', fontsize=16)
    X = np.linspace(1, 100, 100)  # 100: 10000/batch
    line1, = plt.plot(X, img_acc, color="green", linewidth="1.0", linestyle="-")
    line2, = plt.plot(X, img_loss, color="blue", linewidth="1.0", linestyle="-")
    # Comma: UserWarning: Legend does not support [<matplotlib.lines.Line2D object at 0x0000025D8AF17FA0>] instances.
    # A proxy artist may be used instead.
    plt.text(1, img_acc[1], "(1," + str(img_acc[0]) + ")")
    plt.text(20, img_acc[19], "(20," + str(img_acc[19]) + ")")
    plt.text(50, img_acc[49], "(50," + str(img_acc[49]) + ")")
    plt.text(100, img_acc[99], "(100," + str(img_acc[99]) + ")")
    # plt.text(200, img_acc[199], "(200," + str(img_acc[199]) + ")")
    plt.text(1, img_loss[1], "(1," + str(img_loss[0]) + ")")
    plt.text(20, img_loss[19], "(20," + str(img_loss[19]) + ")")
    plt.text(50, img_loss[49], "(50," + str(img_loss[49]) + ")")
    plt.text(100, img_loss[99], "(100," + str(img_loss[99]) + ")")
    # plt.text(200, img_loss[199], "(200," + str(img_loss[199]) + ")")
    plt.legend(handles=[line1, line2], labels=["accuracy", "loss"], loc="upper right", fontsize=6)
    plt.savefig('../result/CNN_test.png')
    # plt.show()
