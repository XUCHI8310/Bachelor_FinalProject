import tkinter
from tkinter import *
from tkinter import ttk
import MNISTtoCSV
import preprocess_GCN
import train_CNN
import train_GCN
import test_CNN
import test_GCN
import tkinter.messagebox
import threading
import KNN


# from PIL import Image, ImageTk


def main_window():
    window_main = tkinter.Tk()  # Call tk window
    window_main.geometry("500x300+200+200")  # Size
    window_main.title('Hand-written Digits Recognizer')  # Title
    # Frame call
    frame1 = Frame(window_main)
    frame2 = Frame(window_main)
    frame3 = Frame(window_main)
    frame4 = Frame(window_main)

    # Brief intro
    Label(frame1, text='\nBased on MNIST dataset').grid(row=0, column=0)

    # Space
    Label(frame2, text=' ').grid(row=0, column=0)

    # Preprocess button
    button_pre = tkinter.Button(frame3, text='Preprocess', height=2, width=10, command=pre_window)
    button_pre.grid(row=0, column=0)

    # Space between preprocess button and train button
    Label(frame3, text='   ').grid(row=0, column=1)

    # Train button
    button_train = tkinter.Button(frame3, text='Train', height=2, width=10, command=train_window)
    button_train.grid(row=0, column=2)

    # Test button, row1 column0
    button_test = tkinter.Button(frame4, text='Test', height=2, width=10, command=test_window)
    button_test.grid(row=1, column=0)

    # Space
    Label(frame4, text='   ').grid(row=0, column=1)

    # Quit button
    button_quit = tkinter.Button(frame4, text='Quit', height=2, width=10, command=window_main.quit)
    button_quit.grid(row=1, column=2)

    # Frames pack to window
    frame1.pack()
    frame2.pack()
    frame3.pack()
    frame4.pack()
    window_main.mainloop()


def pre_window():
    def _pre_start():  # For threading
        com = combobox.get()  # Get selection
        if com == 'CSV':
            MNISTtoCSV.process("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte",
                               "../data/mnist_train.csv", 60000)  # Make CSV train
            MNISTtoCSV.process("../data/t10k-images.idx3-ubyte", "../data/t10k-labels.idx1-ubyte",
                               "../data/mnist_test.csv", 10000)  # Make CSV test
            tkinter.messagebox.showinfo(title='Info', message='CSV format file has been processed.')  # Mind the user
            progressbar.stop()  # Stop progress bar

        if com == 'Graph':
            preprocess_GCN.make_graph()  # Make train graph
            preprocess_GCN.make_graph_test()  # Make test graph
            tkinter.messagebox.showinfo(title='Info', message='Graphs has been saved.')
            progressbar.stop()

    def progress_bar_update():
        if combobox.get() != '':  # Avoid user selected nothing but click the button
            progressbar.start()

    # Threading
    def pre_start():
        thread1 = threading.Thread(target=_pre_start)
        thread2 = threading.Thread(target=progress_bar_update)
        thread1.start()
        thread2.start()

    window_pre = tkinter.Tk()
    window_pre.geometry("500x300+200+200")
    window_pre.title('Hand-written Digits Recognizer')
    frame1 = Frame(window_pre)
    space = Frame(window_pre)  # Space
    frame2 = Frame(window_pre)

    var = tkinter.StringVar()
    value = ['CSV', 'Graph']
    combobox = tkinter.ttk.Combobox(frame1, textvariable=var, value=value, state='readonly')  # Avoid user editing
    combobox.grid(row=0, column=0)

    button_start = tkinter.Button(frame1, text='Start', height=2, width=10, command=pre_start)
    button_start.grid(row=0, column=1)

    Label(space, text=' ').pack()

    progressbar = ttk.Progressbar(frame2, length=300, mode='indeterminate')
    progressbar.pack()

    frame1.pack()
    space.pack()
    frame2.pack()
    window_pre.mainloop()


def train_window():
    window_train = tkinter.Tk()
    window_train.geometry("500x300+200+200")
    window_train.title('Hand-written Digits Recognizer')
    # frame1 = Frame(window_train)
    space1 = Frame(window_train)
    frame2 = Frame(window_train)
    space2 = Frame(window_train)
    frame3 = Frame(window_train)
    space3 = Frame(window_train)
    frame4 = Frame(window_train)

    def _train_start():
        mode = combobox.get()
        if mode == 'GNN':
            train_GCN.train_GCN()  # GCN train
            progressbar.stop()
            tkinter.messagebox.showinfo(title='Info', message='GNN trained. Model has been saved.')
            # print(mode)
        if mode == 'CNN':
            train_CNN.train_CNN()  # CNN train
            progressbar.stop()
            tkinter.messagebox.showinfo(title='Info', message='CNN trained. Model has been saved.')
        if mode == 'KNN':
            KNN.train_KNN()  # KNN train
            progressbar.stop()
            tkinter.messagebox.showinfo(title='Info', message='KNN trained.')

    def progress_bar_update():
        if combobox.get() != '':
            progressbar.start()

    def train_start():
        thread1 = threading.Thread(target=_train_start)
        thread2 = threading.Thread(target=progress_bar_update)
        thread1.start()
        thread2.start()

    # epoch_typing = Label(frame1, text='Epoch (bigger than 0) :      ') # todo realize epoch selecting
    # epoch_typing.grid(row=0, column=0)

    # text = Entry(frame1)
    # text.grid(row=0, column=1)

    space_label_1 = Label(space1, text=' ')
    space_label_1.pack()

    var = tkinter.StringVar()
    value = ['KNN', 'CNN', 'GNN']
    combobox = tkinter.ttk.Combobox(frame2, textvariable=var, value=value, state='readonly')
    combobox.pack()

    space_label_2 = Label(space2, text=' ')
    space_label_2.pack()

    button_start = tkinter.Button(frame3, text='Start', height=2, width=10, command=train_start)
    button_start.pack()

    space_label_3 = Label(space3, text=' ')
    space_label_3.pack()

    progressbar = ttk.Progressbar(frame4, length=300, mode='indeterminate')
    progressbar.pack()

    # frame1.pack()
    space1.pack()
    frame2.pack()
    space2.pack()
    frame3.pack()
    space3.pack()
    frame4.pack()
    window_train.mainloop()


def test_window():
    window_test = tkinter.Tk()
    window_test.geometry("500x300+200+200")
    window_test.title('Hand-written Digits Recognizer')
    space1 = Frame(window_test)
    frame1 = Frame(window_test)
    space2 = Frame(window_test)
    frame2 = Frame(window_test)
    space3 = Frame(window_test)
    frame3 = Frame(window_test)

    # space4 = Frame(window_test)
    # frame4 = Frame(window_test)

    def _test_start():
        mode = combobox.get()
        if mode == 'GNN':
            test_GCN.test()
            progressbar.stop()
            tkinter.messagebox.showinfo(title='Info', message='Done.')
            # print(mode)
        if mode == 'CNN':
            test_CNN.test_CNN()
            progressbar.stop()
            tkinter.messagebox.showinfo(title='Info', message='Done.')
            # image = Image.open("../result/GCN_test_2.png")  # todo insert image
            # pyt = ImageTk.PhotoImage(image)
            # label = Label(frame4, image=pyt)
            # label.pack()

    def progress_bar_update():
        if combobox.get() != '':
            progressbar.start()

    def test_start():
        thread1 = threading.Thread(target=_test_start)
        thread2 = threading.Thread(target=progress_bar_update)
        thread1.start()
        thread2.start()

    Label(space1, text=' ').pack()

    var = tkinter.StringVar()
    value = ['CNN', 'GNN']
    combobox = tkinter.ttk.Combobox(frame1, textvariable=var, value=value, state='readonly')
    combobox.pack()

    space_label = Label(space2, text=' ')
    space_label.pack()

    button_start = Button(frame2, text='Start', height=2, width=10, command=test_start)
    button_start.pack()

    Label(space3, text=' ').pack()

    progressbar = ttk.Progressbar(frame3, length=300, mode='indeterminate')
    progressbar.pack()

    # Label(space4, text=' ').pack()

    space1.pack()
    frame1.pack()
    space2.pack()
    frame2.pack()
    space3.pack()
    frame3.pack()
    # space4.pack()
    # frame4.pack()
    window_test.mainloop()


if __name__ == "__main__":
    main_window()
