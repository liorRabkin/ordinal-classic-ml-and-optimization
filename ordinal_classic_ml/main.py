import numpy as np
import argparse
import random
import torch
import os

# My functions
from ordinal_classic_ml import test, train
import ordinal_classic_ml.utils.utils_functions as uf


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
    parser.add_argument('--algo_name', type=str, default="dt")
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--crit', type=str, default='no_mode')
    parser.add_argument('--split_number', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default='proj/data/ClsKLData/kneeKL224')
    parser.add_argument('--model_dir', type=str, default='saving_models')
    parser.add_argument('--path', type=str,
                        default=r'C:\Users\tal43\Documents\studies\pythonProject\ordinal-classic-ml-and-optimization')
    parser.add_argument('--split_data_phase', type=str, default='no')

    parser.add_argument('--number_of_labels', type=int, default=5)
    parser.add_argument('--const_number', type=int, default=3)
    parser.add_argument('--labels_for_const', type=list, default=[3])

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
    args.model_name = '{}-{}-{}-{}'.format(args.algo_name, args.depth, args.alpha, args.crit)

    # Train NN (create weights):
    train.train_model(args)

    # Test the model:

    # test = data_eng.feature_engineering(path, 'test_data.csv')


    # best_epoch = test.phases_build_all_criterions(args)
    # print('The best epoch is: ' + str(best_epoch))

    # args.phase = 'test'
    # args.batch_size = 16
    # test.test_models(args, best_epoch)

    # Create more graphs
    # plots.create_external_graphs(args)
