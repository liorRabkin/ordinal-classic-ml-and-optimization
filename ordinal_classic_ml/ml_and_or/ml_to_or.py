import numpy as np
import pandas as pd
import os

# My functions
from ordinal_classic_ml.utils.operation_research import operation_research_func


def ml_and_or(args, df_epochs, cost_matrix, fault_price, const_num,
              number_of_labels, excel_title, phase, labels_for_const):
    sheets_names = []
    all_results = pd.DataFrame()

    classes_for_const = ''
    for cons_class in labels_for_const:
        classes_for_const += '_' + str(cons_class)

    excel_path = os.path.join(args.model_dir, args.model_name,
                              'excels_const_' + str(const_num) + '%_on_class' + classes_for_const)
    os.makedirs(excel_path, exist_ok=True)
    excel_name = os.path.join(excel_path, phase + '_' + excel_title + '.xlsx')

    with pd.ExcelWriter(excel_name) as writer:
        print('ml_to_or starting')

        df_cost = pd.DataFrame()
        df_acc = pd.DataFrame()

        or_real_cost = []
        ml_real_cost = []
        or_acc = []
        ml_acc = []
        ml_lab = pd.DataFrame()
        or_lab = pd.DataFrame()

        columns_output = [x for x in df_epochs.columns if 'labels' not in x]
        columns_labels = [x for x in df_epochs.columns if 'labels' in x]

        for outputs, labels in zip(columns_output,
                                   columns_labels):  # range(int(df_epochs.shape[1]/2)): #best_epoch_for_cost): #

            ml_predict_prob = np.array(df_epochs[outputs].tolist())
            objective_function, or_predict_hard = operation_research_func(
                ml_predict_prob, number_of_labels, cost_matrix, fault_price, const_num, labels_for_const)
            results = build_excel_fold(np.array(df_epochs[labels]), ml_predict_prob,
                                       or_predict_hard, cost_matrix, number_of_labels)

            ml_lab[outputs] = results['ML decision labels']
            or_lab[outputs] = results['OR decision labels']

            # or_df['epoch'+str(epoch_column)] = or_predict_hard
            # results mean
            results_mean = results.mean()[4:]
            # print('results_mean')
            # print(results_mean)
            all_results = pd.concat((all_results, pd.DataFrame(results_mean).T), axis=0)
            # sheet_name = phase + '_epoch_' + str(epoch_column+1)
            # print(sheet_name)
            sheets_names.append(outputs)

            results.to_excel(writer, sheet_name=outputs)
            or_real_cost.append(results_mean['OR real cost'])
            ml_real_cost.append(results_mean['ML (max_likelihood) real cost'])
            or_acc.append(results_mean['OR accuracy'])
            ml_acc.append(results_mean['ML accuracy'])

            # print(':::::::::::::')
            # print('epoch {}'.format(str(epoch_column+1)))
            # print(f'phase {phase} epoch {epoch_column+1} in train_eng going to ml_or')
            # print(f'acc {results_mean["ML accuracy"]}')

        df_cost['or cost'] = or_real_cost
        df_cost['ml cost'] = ml_real_cost
        df_acc['or acc'] = or_acc
        df_acc['ml acc'] = ml_acc

        total_results = build_excel_mean_of_folds(all_results, sheets_names)
        total_results.to_excel(writer, sheet_name='total')
        df_cost['epoch'] = pd.Series([i + 1 for i in range(len(or_real_cost))])
        df_acc['epoch'] = pd.Series([i + 1 for i in range(len(or_real_cost))])

    print('end')
    return df_cost, df_acc, ml_lab, or_lab

