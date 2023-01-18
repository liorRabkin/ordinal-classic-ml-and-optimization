import pandas as pd
import os

# My functions
import ordinal_classic_ml.utils.model as ml
import ordinal_classic_ml.utils.train_eng as train_eng
import ordinal_classic_ml.utils.data_engineering as data_eng
from ordinal_classic_ml.utils.operation_research import operation_research_func as operation_research_func


def read_split_data(fold_ind, folder):
    path = os.path.join(folder, 'splited_data')

    files = os.listdir(path)
    for file in files:
        name = str(fold_ind + 1) + '.xlsx'
        if file.endswith(name):
            df = pd.read_excel(os.path.join(path, file))
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # delete unnamed column
            train = df.groupby('job').get_group('train')
            train_lab = train['labels'].values
            train_feat = train.drop(['labels', 'job'], axis=1).values
            val = df.groupby('job').get_group('val')
            val_lab = val['labels'].values
            val_feat = val.drop(['labels', 'job'], axis=1).values
            return train_lab, train_feat, val_lab, val_feat


def train_model(args):
    """Train the model (fit & predict) and validate the model (predict)"""

    print('--Phase 0: Argument settings--')

    print('--Phase 1: Split data (if needed)--')
    if args.split_data_phase == 'yes':
        print('--Phase 1: Split data--')

        train = data_eng.feature_engineering(args.path, 'train_data.csv')
        data_eng.split_data(train, args.path, args.split_number, args.seed)

    print('--Starting ML fit & predict over Stratified k-folds--')

    sheets_names = []
    all_results = pd.DataFrame()
    excel_name = 'results ' + args.algo_name + ' depth ' + str(args.depth) + ' alpha ' + str(
        args.alpha) + ' criterion ' + args.crit + ' const ' + str(args.const_number) + '%.xlsx'
    print(excel_name)
    if not os.path.exists(os.path.join(args.path, 'Runs')):
        os.mkdir(os.path.join(args.path, 'Runs'))
    save_excel_path = os.path.join(args.path, 'Runs', excel_name)

    with pd.ExcelWriter(save_excel_path) as writer:
        # running on SPLIT_NUMBER folds

        for fold_ind in range(args.split_number):
            print(f'--Phase 2: ML Model fit folder {fold_ind}--')

            train_lab, train_feat, val_lab, val_feat = read_split_data(fold_ind, args.path)
            model = ml.ml_model_fit(args, train_feat, train_lab)
            # running on train/val in each fold
            print(f'--Phase 3: ML Model predict on train & valfolder {fold_ind}--')

            for lab, feats, result_type in [(train_lab, train_feat, 'train'), (val_lab, val_feat, 'val')]:
                ml_predict_prob, ml_predict_hard = ml.ml_model_predict(model, feats)
                # print(ml_predict_prob)
                objective_function, or_predict_hard = operation_research_func(ml_predict_prob, args.number_of_labels,
                                                                              args.cost_matrix, args.fault_price,
                                                                              args.const_number, args.labels_for_const)
                results = train_eng.build_excel_fold(lab, ml_predict_prob, or_predict_hard, args.cost_matrix,
                                                     args.number_of_labels)

                # results mean
                results_mean = results.mean()[4:]
                all_results = pd.concat((all_results, pd.DataFrame(results_mean).T), axis=0)
                sheet_name = result_type + str(fold_ind + 1)
                sheets_names.append(sheet_name)

                results.to_excel(writer, sheet_name=sheet_name)

        total_results = train_eng.build_excel_mean_of_folds(all_results, sheets_names)
        total_results.to_excel(writer, sheet_name='total')

    print('end')
