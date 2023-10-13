# -- coding: utf-8 --

import threading
import time

import matplotlib.pyplot as plt

from base_model import BaseModel

class Vis(object):

    @staticmethod
    def visualize_loss(model:BaseModel, x_data, y_data, win_size:int, lock:threading.Lock):
        plt.ioff()
        fig, ax = plt.subplots()
        ax.set_xlabel('epoch')
        ax.set_ylabel('Loss')
        line = ax.plot(x_data, y_data, linewidth=0.5)[0]

        while True:
            if len(x_data) > 2:
                with lock:
                    x = x_data[-win_size:]
                    y = y_data[-win_size:]

                plt.xlim(x[0], x[-1])
                plt.ylim(min(y), max(y))
                line.set_xdata(x)
                line.set_ydata(y)
                line.set_visible(True)
                plt.pause(0.0001)

            time.sleep(0.01)
            if model.is_trained:
                break

        plt.ioff()
        plt.show()