import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
from models.config_WOAttention import *

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    fig.savefig(config_WOAttention['training_loss'] + '.png')
