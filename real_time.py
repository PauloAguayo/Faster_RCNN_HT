import matplotlib.pyplot as plt
import time

class Real_time(object):

  def __init__(self):
    self.x_vals = []
    self.y_real_vals = []
    self.y_dif_vals = []
    self.y_limit = []

    self.fig = plt.figure()
#    self.ax = self.fig.add_subplot(211)
    self.ax = self.fig.add_subplot(111)
    self.ax.set_ylabel('Current Detections')
    self.ax.set_xlabel('Time (s)')
    plt.grid()
    #self.ab = self.fig.add_subplot(212)
    #self.ab.set_ylabel('New Detections')
#    self.ab.set_xlabel('Time (s)')
    self.fig.suptitle('Detections v/s Time')
    self.fig.show()
    #plt.grid()
    plt.savefig('chart.png')

  def Online_plot(self, clock, real_counting, dif_counting,limit, window):
    self.x_vals.append(clock)
    self.y_real_vals.append(real_counting)
#    self.y_dif_vals.append(dif_counting)
    self.y_limit.append(limit)

    self.ax.plot(self.x_vals,self.y_real_vals, color='b')
    self.ax.plot(self.x_vals,self.y_limit, 'k--')
#    self.ab.plot(self.x_vals,self.y_dif_vals, color='r')
    self.fig.canvas.draw()

    self.ax.set_xlim(left=max(0,clock-window),right=clock+0.1*window+1)
#    self.ab.set_xlim(left=max(0,clock-window),right=clock+0.1*window+1)

    if len(self.x_vals)>window+2:
        self.x_vals.pop(0)
        self.y_real_vals.pop(0)
        self.y_limit.pop(0)
#        self.y_dif_vals.pop(0)

    plt.savefig('chart.png')
