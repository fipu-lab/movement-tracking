import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("csv_name", help="Enter the name of your csv file")
args = parser.parse_args()

def animate_graph(i, csv_file):
    data = pd.read_csv(csv_file)
    result_data = data["result"]
    axisX = np.arange(len(result_data))
    axisY = result_data

    plt.cla()
    plt.plot(axisX, axisY, label="Movement")
    plt.xlabel("Frames")
    plt.ylabel("Movement")
    plt.legend()

ani = FuncAnimation(plt.gcf(), animate_graph, interval=1000, fargs=(args.csv_name, ))
plt.show()