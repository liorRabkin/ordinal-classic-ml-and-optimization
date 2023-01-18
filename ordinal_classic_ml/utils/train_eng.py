import numpy as np
import pandas as pd

import ordinal_classic_ml.utils.utils_functions as uf

def build_excel_fold(lab, ml_predict_prob, or_predict_hard, cost_matrix, number_of_labels):
    # ML vect and matrix
    ml_predict_hard = np.argmax(ml_predict_prob, axis=1)

    # if ml_predict_hard.shape[0] != ml_predict_hard.size:
    #     ml_predict_hard_vect = np.argmax(ml_predict_hard, axis=1)
    #     ml_predict_hard_matrix = ml_predict_hard
    # else:
    # ml_predict_hard = ml_predict_hard.reshape(-1)
    ml_predict_hard_vect = ml_predict_hard
    ml_predict_hard_matrix = np.eye(number_of_labels)[ml_predict_hard_vect]

    results = pd.DataFrame()

    # Different Labels
    results['true labels'] = lab

    results['ML decision labels'] = ml_predict_hard_vect
    results['OR decision labels'] = or_predict_hard

    predict_labels_min_cost, _ = uf.min_cost_label(cost_matrix, number_of_labels, ml_predict_prob)
    results['min cost ML labels'] = predict_labels_min_cost

    # -------------------------------------------------------

    # Indices
    indices_ml = uf.indices(number_of_labels, lab, ml_predict_hard_vect, cost_matrix)
    indices_or = uf.indices(number_of_labels, lab, or_predict_hard, cost_matrix)

    results['ML accuracy'] = indices_ml['acc']
    results['OR accuracy'] = indices_or['acc']

    # Cost of something vs Real
    results['ML (max_likelihood) real cost'] = indices_ml['cost']
    results['OR real cost'] = indices_or['cost']
    results['ML (min-cost) real cost'] = uf.cost_pred_vector(cost_matrix, predict_labels_min_cost, lab)

    # cost per class
    # for test_label in range(number_of_labels):
    #     test_label_vector = test_label * np.ones(lab.shape, dtype=int)
    #     results['OR real class ' + str(test_label) + 'cost'] = uf.cost_pred_matrix(cost_matrix, test_label_vector, ml_predict_prob)

    # ML vs OR
    results['ML (max_likelihood) ML (min-cost) cost'] = uf.cost_pred_matrix(cost_matrix, predict_labels_min_cost,
                                                                            ml_predict_hard_matrix)
    results['ML (min-cost) OR cost'] = uf.cost_pred_vector(cost_matrix, predict_labels_min_cost, or_predict_hard)
    results['ML (max_likelihood) OR cost'] = uf.cost_pred_matrix(cost_matrix, or_predict_hard, ml_predict_hard_matrix)

    # ML Probability vector
    for obj in range(number_of_labels):
        name = 'ML Probability vector ' + str(obj)
        results[name] = ml_predict_prob[:, obj].tolist()

    # variance
    x = np.arange(number_of_labels)
    xpower2 = np.power(x, 2)
    xp = np.zeros(ml_predict_prob.shape[0])
    x2p = np.zeros(ml_predict_prob.shape[0])

    for index_prob_vector in range(number_of_labels):
        xp += x[index_prob_vector] * ml_predict_prob[:, index_prob_vector]
        x2p += xpower2[index_prob_vector] * ml_predict_prob[:, index_prob_vector]
    var = x2p - np.power(xp, 2)
    results['variance'] = var

    # Mean Squared Error
    mse = np.zeros(ml_predict_prob.shape[0])

    x = np.arange(number_of_labels)
    mode = np.argmax(ml_predict_prob, axis=1)
    for index_prob_vector in range(number_of_labels):
        mse += np.power((x[index_prob_vector] - mode), 2) * ml_predict_prob[:, index_prob_vector]
    results['MSE'] = mse

    # AUC
    results['ML AUC'] = indices_ml['auc'].mean()
    results['OR AUC'] = indices_or['auc'].mean()

    # Moves
    results['equal - no moves'] = abs(lab - ml_predict_hard) == abs(lab - or_predict_hard)
    results['pos move'] = abs(lab - ml_predict_hard) > abs(lab - or_predict_hard)
    results['neg move'] = abs(lab - ml_predict_hard) < abs(lab - or_predict_hard)

    # Diff

    diff = abs(lab - ml_predict_hard) - abs(lab - or_predict_hard)
    results['pos move diff'] = np.where(diff > 0, diff, 0)
    results['neg move diff'] = -1 * np.where(diff < 0, diff, 0)

    results['total moves diff'] = np.where(diff > 0, diff, 0) - np.where(diff < 0, diff, 0)
    results['moves - abs result'] = diff

    # Cost diff
    lab_predict_hard_matrix = np.eye(number_of_labels)[lab]
    diff_cost_lab_ml = uf.cost_pred_matrix(cost_matrix, ml_predict_hard, lab_predict_hard_matrix)
    diff_cost_lab_or = uf.cost_pred_matrix(cost_matrix, or_predict_hard, lab_predict_hard_matrix)
    diff_cost_ml_or = uf.cost_pred_matrix(cost_matrix, or_predict_hard, ml_predict_hard_matrix)

    diff_cost = diff_cost_lab_ml - diff_cost_lab_or
    results['pos move cost'] = np.where(diff_cost > 0, diff_cost, 0)
    results['neg move cost'] = -1 * np.where(diff_cost < 0, diff_cost, 0)

    results['total moves cost'] = np.where(diff_cost > 0, diff, 0) - np.where(diff_cost < 0, diff_cost, 0)
    sum_cost = diff_cost_lab_ml + diff_cost_ml_or
    results['Sum cost'] = sum_cost

    # Steps
    # results['steps'] = abs(lab - ml_predict_hard) + abs(ml_predict_hard - or_predict_hard)
    results['steps real ML'] = abs(lab - ml_predict_hard)
    results['steps ML OR'] = abs(ml_predict_hard - or_predict_hard)

    results['steps ML OR for moving'] = np.where(abs(ml_predict_hard - or_predict_hard) > 0,
                                                 abs(ml_predict_hard - or_predict_hard), np.nan)

    results['steps real Or'] = abs(lab - or_predict_hard)
    results['total steps sum'] = abs(lab - ml_predict_hard) + abs(ml_predict_hard - or_predict_hard)

    # #Prior Probability
    # prob_vect = np.array([])
    # for i in range(number_of_labels):
    #     count = np.count_nonzero(lab == i)
    #     prob_vect = np.append(prob_vect, count/len(lab)).reshape((1, -1))
    # # print("The Prior Probability of " +str(result_type)+ " fold " + str(fold_ind) + " is:" + str(prob_vect))
    # label_min_cost, cost_min_cost = uf.min_cost_label(cost_matrix, number_of_labels, prob_vect)
    #
    # prior = {'fold name': str(result_type) + ' ' + str(fold_ind+1), 'probability vector': prob_vect[0], 'accracy of min cost label': prob_vect[0, label_min_cost][0], 'cost of min cost label': cost_min_cost[0, label_min_cost][0]}
    # return results, prior

    return results


def build_excel_mean_of_folds(all_results, sheets_names):
    all_results = all_results.append(all_results.iloc[::2].mean(), ignore_index=True)
    sheets_names.append('mean train')

    all_results = all_results.append(all_results.loc[1::2].mean(), ignore_index=True)
    sheets_names.append('mean val')

    all_results['type'] = sheets_names
    all_results.set_index('type', inplace=True)
    return all_results
