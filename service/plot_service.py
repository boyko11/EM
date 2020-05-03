from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np


class PlotService:

    def __init__(self):
        pass

    @staticmethod
    def plot_clusters_per_label_barchart(labels, k, label_counts_per_cluster):

        # create plot
        fig, ax = plt.subplots()
        index = np.arange(len(labels))
        bar_width = 0.3
        opacity = 0.95

        current_bar_index = index
        for k_index in range(k):
            plt.bar(current_bar_index, label_counts_per_cluster[k_index], width=bar_width, alpha=opacity,
                label='Cluster {}'.format(k_index))
            current_bar_index = current_bar_index + bar_width

        plt.title('Cluster Membership per Label')
        plt.xticks(index + bar_width, labels)
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    labels = ["A", "B", "C"]
    k = 3
    label_counts_per_cluster = {
        0: [90, 55, 40],
        1: [85, 62, 54],
        2: [88, 57, 48]
    }

    PlotService().plot_clusters_per_label_barchart(labels, k, label_counts_per_cluster)