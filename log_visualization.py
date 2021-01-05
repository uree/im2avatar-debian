import matplotlib.pyplot as plt
import numpy as np
import re


def show_loss(filepath="../train_shape/02958343/log_train.txt"):
    x = []
    y = []
    file = open(filepath)
    lines = file.readlines()

    valined = [l for l in lines if l.startswith("<VAL>")]

    for v in valined:
        val = v.split("loss: ")
        preint = val[-1].rstrip(".\n")
        epoch = re.findall(r"[0-9]+", val[0])
        x.append(int(epoch[0]))
        y.append(float(preint))


    plt.plot(x, y)

    ymin = np.argmin(y)
    plt.annotate("min ("+str(x[ymin])+","+str(y[ymin])+")", xy=(x[ymin], y[ymin]))

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    plt.plot(x,p(x),"r--")
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.show()
