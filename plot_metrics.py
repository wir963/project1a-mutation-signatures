import numpy as np

import matplotlib.pyplot as plt


def plot_silhouette_score():
    with open("output/10_5_17/metrics/silhouette_score.txt", 'r') as f:
        y = f.read()
    y = y.replace("[", "")
    y = y.replace("]", "")
    y = y.split(",")
    y = [float(string) for string in y]
    x = np.arange(10,1001,10)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel="run #", ylabel="silhouette score", title="Silhouette Score over runs")
    plt.show()

def plot_reconstruction_error():
    with open("output/10_5_17/metrics/reconstruction_error.txt", 'r') as f:
        y = f.read()
    y = y.replace("[", "")
    y = y.replace("]", "")
    y = y.split(",")
    y = [float(string) for string in y]
    x = np.arange(10,1001,10)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel="run #", ylabel="reconstruction error", title="Reconstruction Error over runs")
    plt.show()

if __name__ == '__main__':
    plot_reconstruction_error()
