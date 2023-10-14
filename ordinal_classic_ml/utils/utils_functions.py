from typing import Dict, Any
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import pandas as pd
import shutil

import ordinal_classic_ml.test as test


def fault_price_generator(num_of_labels):
    fault_price = np.zeros(num_of_labels)
    for i in range(num_of_labels):
        fault_price[i] = 0
    return fault_price


def cost_matrix_generator(num_of_labels):
    cost_matrix = np.zeros((num_of_labels, num_of_labels))
    for i in range(num_of_labels):
        for j in range(num_of_labels):
            cost_matrix[i, j] = abs(j - i) - 1
    return cost_matrix


def cost_pred_vector(cost_matrix, real_label, pred_label):
    cost_vector = np.zeros(len(real_label))
    for i in range(len(real_label)):
        cost_vector[i] = cost_matrix[real_label[i], pred_label[i]]
    return cost_vector


def cost_pred_matrix(cost_matrix, one_predict_vect, two_predict_matrix):
    cost_vector = np.zeros(len(one_predict_vect))
    for i in range(len(one_predict_vect)):
        for j in range(two_predict_matrix.shape[1]):
            cost_vector[i] += two_predict_matrix[i, j] * cost_matrix[one_predict_vect[i], j]
    return cost_vector


def min_cost_label(cost_matrix, NUM_OF_LABELS, ml_predict_prob):
    cost_eachSample_eachLabel = np.zeros((ml_predict_prob.shape[0], NUM_OF_LABELS))
    for s in range(ml_predict_prob.shape[0]):
        for i in range(NUM_OF_LABELS):
            for j in range(NUM_OF_LABELS):  # ml_predict_prob.shape[1]
                cost_eachSample_eachLabel[s, i] += ml_predict_prob[s, j] * cost_matrix[j, i]
    predict_labels = np.argmin(cost_eachSample_eachLabel, axis=1)

    return predict_labels, cost_eachSample_eachLabel


def indices(number_of_labels: int, labels: np.ndarray, predict_hard: np.ndarray,
            cost_matrix: np.ndarray) -> Dict[str, Any]:
    # Cost
    cost_sum = cost_pred_vector(cost_matrix, labels, predict_hard)
    # AUC
    auc = np.zeros(number_of_labels)
    for i in range(number_of_labels):
        predict_labels_binary = predict_hard == i
        labels_binary = labels == i
        # auc[i] = roc_auc_score(labels_binary, predict_labels_binary)
    # Accuracy
    acc = np.equal(labels, predict_hard).astype(int)
    return {'cost': cost_sum, 'auc': auc, 'acc': acc}


def data_distribution(dset_loaders):
    for phase in ['train', 'val', 'test']:
        print(f'{phase} Dataset lengh is: {len(dset_loaders[phase].dataset)}')

        labels = [x[1] for x in dset_loaders[phase].dataset.imgs]
        unique, counts = np.unique(labels, return_counts=True)
        print(dict(zip(unique, counts)))


def parameters_optimization(args):
    print('--Phase 4: Optimization - choosing the best models--')

    algo_path = os.path.join(args.path, 'Runs')
    if not os.path.exists(algo_path):
        assert os.path.exists(algo_path), 'no exist Runs files'

    best_path = os.path.join(args.path, 'Runs', 'Best_models')
    if not os.path.exists(best_path):
        os.mkdir(best_path)

    # # Clean Summary Best Parameters All Algorithms.xlsx
    # if os.path.exists(os.path.join(best_path, 'Summary_Best_Parameters_All_Algorithms.xlsx')):
    #     os.remove(os.path.join(best_path, 'Summary_Best_Parameters_All_Algorithms.xlsx'))
    classes_for_const = ''
    for cons_class in args.labels_for_const:
        classes_for_const += '_' + str(cons_class)

    algorithms = os.listdir(algo_path)
    best_df = pd.DataFrame()
    row_names = []
    for algo in ['decision_tree', 'decision_tree_ordinal']:  # algorithms:
        if algo != 'Best_models':
            constraints = os.listdir(os.path.join(algo_path, algo))
            for constraint in ['const 0.5% on class_2_4']:  # constraints:
                print(algo_path)
                print(algo)
                print(constraint)
                path = os.path.join(algo_path, algo, constraint)
                files_names_list = os.listdir(path)
                cost = args.cost_matrix[0, args.number_of_labels - 1]
                min_cost_file_name = ' '

                for file_name in files_names_list:
                    if file_name.endswith('.xlsx'):
                        read = pd.read_excel(os.path.join(path, file_name), sheet_name='total')
                        val_or_real_cost = read['OR real cost'].iloc[args.split_number * 2 + 1]
                        print(f"val_or_real_cost: {val_or_real_cost}")
                        print(f"file_name: {file_name}")

                        if val_or_real_cost < cost:
                            cost = val_or_real_cost
                            print(f"cost: {cost}")
                            min_cost_file_name = file_name

                # print(min_cost_file_name)
                # dict[algo] = min_cost_file_name

                assert min_cost_file_name != ' ', 'no such optimal file'

                print(f'min_cost_file_name {min_cost_file_name}')

                depth = int(min_cost_file_name.split(' ')[3])

                if float(min_cost_file_name.split(' ')[5]) == int(float(min_cost_file_name.split(' ')[5])):
                    alpha = int(min_cost_file_name.split(' ')[5])
                else:
                    alpha = float(min_cost_file_name.split(' ')[5])

                crit = min_cost_file_name.split(' ')[7]

                if float(constraint.split('%')[0].split(' ')[1]) == int(float(constraint.split('%')[0].split(' ')[1])):
                    constra = int(constraint.split('%')[0].split(' ')[1])
                else:
                    constra = float(constraint.split('%')[0].split(' ')[1])

                classes = constraint.split('class_')[1]
                constraint_classes = [int(x) for x in classes.split('_')]

                mean_train_test_results = test.test_model(args, best_path, algo, depth, alpha, crit, constra,
                                                          constraint_classes)

                phases = ['train', 'test']
                for phase in phases:
                    # best_df = best_df.append(pd.read_excel(os.path.join(best_path, algo, constraint, phase, min_cost_file_name), sheet_name=phase).mean()[4:], ignore_index=True)
                    row_names.append(phase + ' ' + min_cost_file_name)
                best_df = best_df.append(mean_train_test_results)
                # row_names.append(['train ' +min_cost_file_name, 'test '+min_cost_file_name])

    best_df['Algo & Const'] = row_names
    best_df.set_index('Algo & Const', inplace=True)
    # excel_name = os.path.join(best_path, "Summary_Best_Parameters_All_Algorithms.xlsx")
    # best_df.to_excel(excel_name)
    return best_df


def creating_summary_excel(args, mean_train_test):
    print(mean_train_test)
    best_path = os.path.join(args.path, 'Runs', 'Best_models')
    if not os.path.exists(best_path):
        os.mkdir(best_path)

    excel_name = os.path.join(best_path, "Summary_Best_Parameters_All_Algorithms.xlsx")
    exist_excel = pd.read_excel(excel_name)
    exist_excel.set_index('Algo & Const', inplace=True)
    exist_excel = exist_excel.append(mean_train_test)
    # exist_excel.set_index('Algo & Const', inplace=True)
    exist_excel.to_excel(excel_name)
