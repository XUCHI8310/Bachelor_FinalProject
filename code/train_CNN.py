import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import *  # TypeError: 'module' object is not callable
import numpy as np
import matplotlib.pyplot as plt
import model_CNN


def train_CNN():
    # Hyper param
    BATCH_SIZE = 100
    LR = 0.01
    epoch = 200

    # def train_cnn(epoch):
    # Read dataset
    train_dataset = torchvision.datasets.MNIST(root='../data', train=True,
                                               transform=torchvision.transforms.ToTensor(), download=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define model, loss func, optimizer
    model = model_CNN.CNNmodel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR)

    # if torch.cuda.is_available():
    #     print("CUDA is enable!")
    #     model = model.cuda()
    #     model.train()

    # Import progress bar
    with tqdm(total=epoch) as pbar:
        pbar.set_description('Processing')
        img_acc = []
        img_loss = []
        # Train
        for epoch_temp in range(epoch):
            # Init loss & accuracy param
            train_loss = 0.0
            train_acc = 0.0

            for i, data in enumerate(train_loader, 1):
                # Autograd
                img, label = data
                img = torch.autograd.Variable(img)
                label = torch.autograd.Variable(label)

                # Forward
                optimizer.zero_grad()
                out = model(img)
                loss = criterion(out, label)

                # Backward updating weights
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * label.size(0)
                _, pred = out.max(1)
                num_correct = pred.eq(label).sum()  # Sum of correct num
                accuracy = pred.eq(label).float().mean()
                train_acc += num_correct.item()

            print('\nFinished {} epoch. Loss: {:.6f}, Acc: {:.6f}'.format(epoch_temp+1, train_loss / len(train_dataset), train_acc / len(train_dataset)))
            pbar.update(1)

            img_acc = np.append(img_acc, train_acc / len(train_dataset))
            img_loss = np.append(img_loss, train_loss / len(train_dataset))

    plt.figure()
    plt.suptitle('Train result', fontsize=16)
    X = np.linspace(1, epoch, epoch)
    line1, = plt.plot(X, img_acc, color="green", linewidth="1.0", linestyle="-")
    line2, = plt.plot(X, img_loss, color="blue", linewidth="1.0", linestyle="-")
    # Comma: UserWarning: Legend does not support [<matplotlib.lines.Line2D object at 0x0000025D8AF17FA0>] instances.
    # A proxy artist may be used instead.
    plt.text(1, img_acc[1], "(1," + str(img_acc[0]) + ")")
    plt.text(20, img_acc[19], "(20," + str(img_acc[19]) + ")")
    plt.text(50, img_acc[49], "(50," + str(img_acc[49]) + ")")
    plt.text(100, img_acc[99], "(100," + str(img_acc[99]) + ")")
    plt.text(200, img_acc[199], "(200," + str(img_acc[199]) + ")")
    plt.text(1, img_loss[1], "(1," + str(img_loss[0]) + ")")
    plt.text(20, img_loss[19], "(20," + str(img_loss[19]) + ")")
    plt.text(50, img_loss[49], "(50," + str(img_loss[49]) + ")")
    plt.text(100, img_loss[99], "(100," + str(img_loss[99]) + ")")
    plt.text(200, img_loss[199], "(200," + str(img_loss[199]) + ")")
    plt.legend(handles=[line1, line2], labels=["accuracy", "loss"], loc="upper right", fontsize=6)
    plt.savefig('../result/CNN_train.png')
    # plt.show()

    torch.save(model, 'cnn.pt')
