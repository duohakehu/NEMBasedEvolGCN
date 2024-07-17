import threading
import time

import matplotlib.pyplot as plt


class PlotUtil:

    def __init__(self, x_data, y_data, x_label, y_label):
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(x_data, y_data, 'b-', label="compare")
        self.x_data = x_data
        self.y_data = y_data
        self.ax.set_xlim(0, 0.01)
        self.ax.set_ylim(0, 0.2)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.legend()
        self.lock = threading.Lock()

    def show_figure(self):
        while True:
            with self.lock:
                self.line.set_data(self.x_data, self.y_data)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            time.sleep(1)

    def start_thread(self, function, args):
        thread = threading.Thread(target=function, args=args)
        thread.daemon = True
        thread.start()

    def update_data(self, x, y):
        with self.lock:
            self.x_data.append(x)
            self.y_data.append(y)
