import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib as mpl


def draw_maps(mistakes, num_of_labels, draw_path, file_name):
    sum_instances = sum([sum(i) for i in mistakes])

    mpl.use('TkAgg')
    fig, ax = plt.subplots()
    # Loop over data dimensions and create text annotations.
    for i in range(num_of_labels):
        for j in range(num_of_labels):
            if mistakes[i, j] != 0:
                if mistakes[i, j] > sum_instances / 4:
                    ax.text(j, i, int(mistakes[i, j]), horizontalalignment='center', verticalalignment='center',
                            fontsize=20, color='w')
                else:
                    ax.text(j, i, int(mistakes[i, j]), horizontalalignment='center', verticalalignment='center',
                            fontsize=20)

    ax.imshow(mistakes, cmap='Greys')
    ax.set_xticks(range(0, num_of_labels, 1))
    ax.set_yticks(range(0, num_of_labels, 1))
    ax.set_ylabel('Actual class', fontsize=20)
    ax.set_xlabel('Predicted class', fontsize=20)
    fig.savefig(os.path.join(draw_path, file_name + '.png'))

    plt.show()



if __name__ == '__main__':
    path = r'C:\Users\tal43\Documents\studies\pythonProject\external_matrix_draw'
    num_of_labels = 5

    mistakes = np.array([[291,7,27,3,0], [104, 18, 27,4,0], [61,16,114,21,0], [6,0,18,79,3], [0,0,0,9,18]])
    file_name = 'true_ml_val_CE'
    draw_maps(mistakes, num_of_labels, path, file_name)

    mistakes = np.array([[293,30,5,0,0], [104, 36, 11, 2, 0], [47, 56,95,14,0], [4,4,22,72,4], [0,0,0,4,23]])
    file_name = 'true_ml_val_weighted_loss'
    draw_maps(mistakes, num_of_labels, path, file_name)

    mistakes = np.array([[264,43,20,1,0], [79,53,18,3,0], [38,63,92,19,0], [2,4,18,80,2], [0,0,0,10,17]])
    file_name = 'true_or_val_CE'
    draw_maps(mistakes, num_of_labels, path, file_name)

    mistakes = np.array([[259,61,8,0,0], [79,58,15,1,0], [24,72,105,11,0], [4,1,27,72,2], [0,0,0,6,21]])
    file_name = 'true_or_val_weighted_loss'
    draw_maps(mistakes, num_of_labels, path, file_name)

    mistakes = np.array([[264,43,21,0,0], [79,53,20,1,0], [38,63,109,0,2], [2,4,58,21,21], [0,0,0,3,24]])
    file_name = 'true_or_val_CE_3_3'
    draw_maps(mistakes, num_of_labels, path, file_name)

    mistakes = np.array([[255,67,6,0,0], [76,61,16,0,0], [25,74,113,0,0], [4,1,68,24,9], [0,0,0,1,26]])
    file_name = 'true_or_val_weighted_loss_3_3'
    draw_maps(mistakes, num_of_labels, path, file_name)