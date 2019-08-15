from scipy import stats
from matplotlib import pyplot as plt

class Stats:

    def __init__(self):
        pass

    def correlate(self, x, y):
        pearsonr =  stats.pearsonr(x, y)
        print("pearson correlation; {}".format(pearsonr))
        plt.xlabel("DEPTH")
        plt.ylabel("ACC")
        plt.title("r = {0:.3f}".format(pearsonr[0]))
        plt.scatter(x, y, s=100)
        plt.show()
