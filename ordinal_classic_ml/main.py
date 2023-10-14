import numpy as np
import argparse
import random
import torch
import os

# My functions
from ordinal_classic_ml import test, train
import ordinal_classic_ml.utils.utils_functions as uf
import utils.plots as plots

# Ignore randomness
# def set_seed(seed):
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)


def set_args():
    parser = argparse.ArgumentParser(description='Pytorch Fine-Tune')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--algo_name', type=str, default="decision_tree",
                        help='Algorithms options: '
                             'decision_tree, random_forest, adaboost, catboost, xgboost, '
                             'decision_tree_ordinal, random_forest_ordinal, adaboost_ordinal')
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--crit', type=str, default='no_mode',
                        help='For regular algorithms write no_mode '
                             'For ordinal algorithm the criterion options: WIGR_min, WIGR_max, WIGR_EV, WIGR_mode, WIGR_EV_fix, entropy')
    parser.add_argument('--split_number', type=int, default=5)
    parser.add_argument('--path', type=str,
                        default=r'C:\Users\tal43\Documents\studies\pythonProject\ordinal-classic-ml-and-optimization')
    parser.add_argument('--split_data_phase', type=str, default='no', help='Choose yes for spliting the data')
    parser.add_argument('--test_data_phase', type=str, default='no',
                        help='Choose yes for feature engineering for test data')
    parser.add_argument('--number_of_labels', type=int, default=5)
    parser.add_argument('--const_number', type=int, default=0.5)
    parser.add_argument('--labels_for_const', type=list, default=[2, 4])

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_args()

    args.cost_matrix = np.array([[1, 3, 5, 7, 9],
                                 [3, 1, 3, 5, 7],
                                 [5, 3, 1, 3, 5],
                                 [7, 5, 3, 1, 3],
                                 [9, 7, 5, 3, 1]])

    args.fault_price = uf.fault_price_generator(5)


    print('-------------------------------train')

    # Classic algorithms
    # args.algo_name = 'decision_tree'
    # for args.depth in [2, 3, 4, 5, 6]:
    #     print(f'crit {args.crit}, alpha {args.alpha}, depth {args.depth}')
    #     train.train_model(args)


    # Ordinal algorithms
    # args.algo_name = 'decision_tree_ordinal'
    # for args.crit in ['WIGR_min', 'WIGR_max', 'WIGR_EV', 'WIGR_mode', 'WIGR_EV_fix']:
    #     for args.alpha in [0.5, 1, 3]:
    #         for args.depth in [2, 4, 5, 6]:
    #             print(f'crit {args.crit}, alpha {args.alpha}, depth {args.depth}')
    #             train.train_model(args)


    # print('-------------------------------optimization')
    # mean_train_test_df = uf.parameters_optimization(args)
    # print('-------------------------------summary')
    # uf.creating_summary_excel(args, mean_train_test_df)


    # Create more graphs
    print('-------------------------------graphs')
    plots.create_external_graphs(args)
