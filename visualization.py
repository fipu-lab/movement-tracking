import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_graph(i):
    data = pd.read_csv("movement_data.csv")
    result_data = data["result"]
    axisX = np.arange(len(result_data))
    axisY = result_data

    plt.cla()
    plt.plot(axisX, axisY, label="Movement")

ani = FuncAnimation(plt.gcf(), animate_graph, interval=1000)
plt.show()