import os
import pandas as pd
import torch
import numpy as np

# My functions
import ordinal_classic_ml.utils.model as ml
import ordinal_classic_ml.utils.train_eng as train_eng
import ordinal_classic_ml.utils.data_engineering as data_eng
from ordinal_classic_ml.utils.operation_research import operation_research_func as operation_research_func
import ordinal_classic_ml.utils.plots as plots


def test_model(args, best_path, algo, depth, alpha, crit, constraint, constraint_classes_list):
    """Evaluate the chosen model on train and test data"""

    print('--Phase 2.0: Argument settings--')

    print('--Phase 2.1: Getting data--')

    data = {}
    data['train'] = pd.read_csv(os.path.join(args.path, 'eng_train_data.csv'))
    data['test'] = pd.read_csv(os.path.join(args.path, 'eng_test_data.csv'))

    print('--Starting ML predict--')

    sheets_names = []
    all_results = pd.DataFrame()

    classes_for_const = ''
    for cons_class in constraint_classes_list:
        classes_for_const += '_' + str(cons_class)

    phases = ['train', 'test']

    for phase in phases:

        excel_name = 'results ' + algo + ' depth ' + str(depth) + ' alpha ' + str(
            alpha) + ' criterion ' + crit + ' const ' + str(constraint) + '% on class' + classes_for_const + '.xlsx'

        # print(f'--Phase 2.2: {phase} - ML Model fit & predict--')

        algo_path = os.path.join(best_path, algo)
        if not os.path.exists(algo_path):
            os.mkdir(algo_path)

        algo_constraint_path = os.path.join(best_path, algo,
                                            'const ' + str(constraint) + '% on class' + classes_for_const)
        if not os.path.exists(algo_constraint_path):
            os.mkdir(algo_constraint_path)

        phase_path = os.path.join(algo_constraint_path, phase)

        if not os.path.exists(phase_path):
            os.mkdir(phase_path)

        # print(os.path.join(phase_path))
        # if os.path.exists(os.path.join(phase_path)):
        for file in os.listdir(phase_path):
            os.remove(os.path.join(phase_path, file))

        save_excel_path = os.path.join(phase_path, excel_name)

        with pd.ExcelWriter(save_excel_path) as writer:
            labels = data[phase]['labels'].values
            labels = labels.astype(int)
            features = data[phase].drop(['labels'], axis=1).values

            if phase == 'train':
                model = ml.ml_model_fit(features, labels, algo, depth, alpha, crit)

            ml_predict_prob, ml_predict_hard = ml.ml_model_predict(model, features)

            objective_function, or_predict_hard = operation_research_func(ml_predict_prob, args.number_of_labels,
                                                                          args.cost_matrix, args.fault_price,
                                                                          constraint, constraint_classes_list)

            results = train_eng.build_excel_fold(labels, ml_predict_prob, or_predict_hard, args.cost_matrix,
                                                 args.number_of_labels)

            # results mean
            results_mean = results.mean()[4:]
            all_results = pd.concat((all_results, pd.DataFrame(results_mean).T), axis=0)
            sheets_names.append(phase)

            results.to_excel(writer, sheet_name=phase)
            results_mean.to_excel(writer, sheet_name='total')

            # print(f'--Phase 2.3: Mistakes matrix for {phase}--')
            mistakes_real_ml = plots.mistakes_matrix(results['true labels'], results['ML decision labels'],
                                                     args.number_of_labels)
            mistakes_ml_or = plots.mistakes_matrix(results['ML decision labels'], results['OR decision labels'],
                                                   args.number_of_labels)
            mistakes_real_or = plots.mistakes_matrix(results['true labels'], results['OR decision labels'],
                                                     args.number_of_labels)
            mistakes = [mistakes_real_ml, mistakes_ml_or, mistakes_real_or]

            plots.draw_maps(mistakes, args.number_of_labels, phase, phase_path)
            print('end draw maps')

    return all_results

    # print(f'--Phase 2.3: Mistakes matrix for {phase}--')
    # with pd.ExcelWriter(os.path.join(phase_path, excel_name)) as writer:
    #     total_results = train_eng.build_excel_mean_of_folds(all_results, sheets_names)
    #     total_results.to_excel(writer, sheet_name='total')
