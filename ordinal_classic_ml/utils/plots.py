import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib as mpl


def plot_acc_func_epoch(num_epochs, acc_train_epochs, acc_val_epochs):
    x = [i for i in range(num_epochs)]
    plt.plot(x, acc_train_epochs)
    plt.plot(x, acc_val_epochs)
    plt.legend(["Dataset 1", "Dataset 2"])
    plt.show()


def mistakes_matrix(vector1, vector2, num_of_labels):
    mistakes = np.zeros((num_of_labels, num_of_labels))
    for i in range(len(vector1)):
        mistakes[vector1[i], vector2[i]] += 1
    return mistakes


def draw_maps(mistakes, num_of_labels, phase, draw_path):
    sum_instances = sum([sum(i) for i in mistakes[0]])

    mpl.use('TkAgg')
    fig, ax = plt.subplots()
    # Loop over data dimensions and create text annotations.
    for i in range(num_of_labels):
        for j in range(num_of_labels):
            if mistakes[0][i, j] != 0:
                if mistakes[0][i, j] > sum_instances / 4:
                    ax.text(j, i, int(mistakes[0][i, j]), horizontalalignment='center', verticalalignment='center',
                            fontsize=10, color='w')
                else:
                    ax.text(j, i, int(mistakes[0][i, j]), horizontalalignment='center', verticalalignment='center',
                            fontsize=10)

    ax.imshow(mistakes[0], cmap='Greys')
    ax.set_xticks(range(0, num_of_labels, 1))
    ax.set_yticks(range(0, num_of_labels, 1))
    ax.set_ylabel('Actual class', fontsize=12)
    ax.set_xlabel('Predicted class', fontsize=12)
    # ax.set_title('True vs ML')
    fig.savefig(os.path.join(draw_path, 'true_ml_' + phase + '.png'))

    fig, ax = plt.subplots()
    # Loop over data dimensions and create text annotations.
    for i in range(num_of_labels):
        for j in range(num_of_labels):
            if mistakes[1][i, j] != 0:
                if mistakes[1][i, j] > sum_instances / 4:
                    ax.text(j, i, int(mistakes[1][i, j]), horizontalalignment='center', verticalalignment='center',
                            fontsize=10, color='w')
                else:
                    ax.text(j, i, int(mistakes[1][i, j]), horizontalalignment='center', verticalalignment='center',
                            fontsize=10)

    image = ax.imshow(mistakes[1], cmap='Greys')
    ax.set_xticks(range(0, num_of_labels, 1))
    ax.set_yticks(range(0, num_of_labels, 1))
    ax.set_ylabel('Predicted class by ML model', fontsize=12)
    ax.set_xlabel('Predicted class by OR model', fontsize=12)
    # ax.set_title('ML vs OR')
    fig.savefig(os.path.join(draw_path, 'ml_or_' + phase + '.png'))

    fig, ax = plt.subplots()
    # Loop over data dimensions and create text annotations.
    for i in range(num_of_labels):
        for j in range(num_of_labels):
            if mistakes[2][i, j] != 0:
                if mistakes[2][i, j] > sum_instances / 4:
                    ax.text(j, i, int(mistakes[2][i, j]), horizontalalignment='center', verticalalignment='center',
                            fontsize=10, color='w')
                else:
                    ax.text(j, i, int(mistakes[2][i, j]), horizontalalignment='center', verticalalignment='center',
                            fontsize=10)

    image = ax.imshow(mistakes[2], cmap='Greys')
    ax.set_xticks(range(0, num_of_labels, 1))
    ax.set_yticks(range(0, num_of_labels, 1))
    ax.set_ylabel('Actual class', fontsize=12)
    ax.set_xlabel('Predicted class', fontsize=12)
    # ax.set_title('True vs OR')
    fig.savefig(os.path.join(draw_path, 'true_or_' + phase + '.png'))

    # plt.show()


# def crit_as_epochs(dict_phases, crit, path, title, best_epoch):
#     # create figure and axis objects with subplots()
#     fig, ax = plt.subplots()
#
#     # make a plot
#     ax.plot(dict_phases['train']['epoch'], dict_phases['train']['ml ' + crit], color='#176ccd', label='Train CS_VGG-19')
#     ax.plot(dict_phases['train']['epoch'], dict_phases['train']['or ' + crit], color='#176ccd', linestyle='dotted',
#             label='Train Hyb_CS')
#
#     # set x-axis label
#     ax.set_xlabel("Epochs", fontsize=14)
#     ax.set_ylabel("Accuracy", fontsize=14)
#
#     ax.plot(dict_phases['val']['epoch'], dict_phases['val']['ml ' + crit], color='#cd7817', label='Val CS_VGG-19')
#     ax.plot(dict_phases['val']['epoch'], dict_phases['val']['or ' + crit], color='#cd7817', linestyle='dotted',
#             label='Val Hyb_CS')
#
#     ax.axvline(x=best_epoch, color='black', linestyle='dashed', label=f'Selected epoch')  # {best_epoch}')
#     ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
#     ax.legend()
#     ax.set_ylim([0.3, 0.85])
#
#     # save the plot as a file
#     fig.savefig(os.path.join(path, f'{crit}__{title}.png'))
#
#     fig2, ax2 = plt.subplots()
#
#     # make a plot
#     ax2.plot(dict_phases['test']['epoch'], dict_phases['test']['ml ' + crit], color='red', label='test ml')
#     ax2.plot(dict_phases['test']['epoch'], dict_phases['test']['or ' + crit], color='blue', label='test or')
#     # set x-axis label
#     ax2.set_xlabel("Epochs", fontsize=14)
#     ax2.set_ylabel("Accuracy", fontsize=14)
#     ax2.legend()
#
#     # save the plot as a file
#     fig2.savefig(os.path.join(path, f'{crit}__test__{title}.png'))


# def crit_as_epochs_all_constraints(paths, args, constraints, phase, crit, type):
#     dict = {}
#
#     for excel_path in paths:
#         excel_files = os.listdir(excel_path)
#         for file in excel_files:
#
#             matches = [phase, "20"]
#
#             if all(x in file for x in matches):
#                 print(file)
#                 constraint = file.split('const_')[2].split('%')[0]
#                 algo = file.split('_20')[0].split('val_')[1]
#
#                 path = os.path.join(excel_path, file)
#
#                 read_total_sheet = pd.read_excel(path, sheet_name='total')
#                 df = pd.DataFrame()
#                 df['or ' + crit] = read_total_sheet['OR real cost'][:20]
#                 df['ml ' + crit] = read_total_sheet['ML (max_likelihood) real cost'][:20]
#                 print(df)
#                 df['epoch'] = pd.Series([i + 1 for i in range(read_total_sheet.shape[1] - 2)])
#                 if constraint == '100':
#                     best_epoch = stopping_epoch(df, args.early_stopping)
#
#                 dict[constraint + '%'] = df
#
#     # create figure and axis objects with subplots()
#     fig, ax = plt.subplots()
#
#     if type == 'CE':
#         colors = ['#dcbd9d', '#d88c64', '#966735', '#3e2b16']  # orange
#     else:
#         colors = ['#9de3e6', '#2baba3', '#1b5d6c', '#2b33ab']  # , '#2c2182'] #blue
#
#     # make a plot
#     for ind, name in enumerate(constraints):
#         if name == '100%':
#             ax.plot(dict[name]['epoch'], dict[name]['or ' + crit],
#                     label='W/o constraints', color=colors[ind])
#         else:
#             ax.plot(dict[name]['epoch'], dict[name]['or ' + crit],
#                     label=' $n_4$=' + constraints[ind], color=colors[ind])
#
#     ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
#
#     # set x-axis label
#     ax.set_xlabel("Epochs", fontsize=12)
#     # set y-axis label
#     ax.set_ylabel("Cost", fontsize=12)
#     ax.set_ylim([1.8, 2.8])
#
#     ax.legend()
#     # save the plot as a file
#     fig.savefig(os.path.join('/home/dsi/liorrabkin/projects/thesis_server/saving_models',
#                              crit + ' as function of epochs ' + algo + ' algorithm.png'))


def moves_bars(args, paths, groups, class_num):
    equal = []
    pos = []
    neg = []

    mpl.use('TkAgg')

    for excel_path in paths:
        excel_files = os.listdir(excel_path)
        for file in excel_files:
            if file.endswith('xlsx'):
                path = os.path.join(excel_path, file)

                read_total_sheet = pd.read_excel(path, sheet_name='total', index_col=0)
                # df_cost = pd.DataFrame()
                # df_cost['or cost'] = read_total_sheet.loc['OR real cost']
                # df_cost['ml cost'] = read_total_sheet.loc['ML (max_likelihood) real cost']
                #
                # read_total_sheet = read_total_sheet.set_index('type')
                #
                # best_epoch_row = read_total_sheet.loc['mean val']

                equal.append((np.floor(read_total_sheet.loc['equal - no moves'].values * 1000) / 1000).tolist())
                pos.append((np.floor(read_total_sheet.loc['pos move'].values * 1000) / 1000).tolist())
                neg.append((np.floor(read_total_sheet.loc['neg move'].values * 1000) / 1000).tolist())

    flat_equal = [item for sublist in equal for item in sublist]
    flat_pos = [item for sublist in pos for item in sublist]
    flat_neg = [item for sublist in neg for item in sublist]

    flat_neg[0] = flat_neg[0] + 0.001
    flat_equal[3] = flat_equal[3] + 0.001
    flat_pos[3] = flat_pos[3] - 0.001

    N = len(groups)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.5  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()

    p1 = ax.bar(ind, flat_equal, width, label='Equal', color='#bdbdbd')
    p2 = ax.bar(ind, flat_pos, width, bottom=flat_equal, label='Pos', color='#ffffff')  # cd7817') #b5f38d')
    p3 = ax.bar(ind, flat_neg, width, bottom=[x + y for x, y in zip(flat_equal, flat_pos)], label='Neg',
                color='#292929')  # 176ccd') #ef796a')

    ax.set_ylabel('Percentage of samples', fontsize=12)
    # ax.set_title('Percentage equal/pos/neg moves')
    ax.set_xticks(ind, labels=groups, fontsize=10)
    ax.legend()

    # Label with label_type 'center' instead of the default 'edge'
    ax.bar_label(p1, label_type='center')
    ax.bar_label(p2, label_type='center',padding=-3)
    ax.bar_label(p3, color='gray', label_type='center', padding=5)

    plt.show()
    fig.savefig(os.path.join(args.path, 'Runs', 'Best_models', 'moves_bars_constraints_class_' + class_num + '.png'))


# def steps_ml_or_bars(paths, args, names, class_num):
#     steps_ce = []
#     steps_ord = []
#     for excel_path in paths:
#         excel_files = os.listdir(excel_path)
#         for file in excel_files:
#
#             matches = ["val", "20"]
#
#             if all(x in file for x in matches):
#                 path = os.path.join(excel_path, file)
#
#                 read_total_sheet = pd.read_excel(path, sheet_name='total')
#                 df_cost = pd.DataFrame()
#                 df_cost['or cost'] = read_total_sheet['OR real cost']
#                 df_cost['ml cost'] = read_total_sheet['ML (max_likelihood) real cost']
#                 df_cost['epoch'] = pd.Series([i + 1 for i in range(read_total_sheet.shape[1])])
#                 best_epoch = stopping_epoch(df_cost, args.early_stopping)
#                 best_epoch_row_name = 'val epoch' + str(best_epoch)
#
#                 read_total_sheet = read_total_sheet.set_index('type')
#
#                 best_epoch_row = read_total_sheet.loc[best_epoch_row_name]
#
#                 if 'vgg-19-SGD-0' in excel_path:
#                     steps_ce.append(np.round(best_epoch_row['steps ML OR'], 3))
#
#                 else:
#                     steps_ord.append(np.round(best_epoch_row['steps ML OR'], 3))
#
#     N = len(names)
#     print(F'N {N}')
#     ind = np.arange(N)  # the x locations for the groups
#     width = 0.4  # the width of the bars: can also be len(x) sequence
#
#     fig, ax = plt.subplots()
#
#     p1 = plt.bar(ind - 0.2, steps_ce, width, label='CE', color='#cd7817')  # e9d6c3') #orange
#     p2 = plt.bar(ind + 0.2, steps_ord, width, label='OL', color='#176ccd')  # c6eff0') #blue
#
#     plt.xticks(ind, names, fontsize=12)
#     ax.set_ylabel('Mean Steps Number', fontsize=12)
#     ax.legend(loc=4)
#
#     # Label with label_type 'center' instead of the default 'edge'
#     ax.bar_label(p1)
#     ax.bar_label(p2)
#
#     plt.show()
#     fig.savefig(os.path.join('/home/dsi/liorrabkin/projects/thesis_server/saving_models',
#                              'Steps_ML_OR_constraints_class_' + class_num + '.png'))


def standard_deviation(args, root_dir, const_list, algo_list):
    mpl.use('TkAgg')
    val_list_of_const_list = []
    for const in const_list:
        val_paths_specific_const = []
        for root_dir_specific in root_dir:
            val_paths_specific_const.append(os.path.join(root_dir_specific, const, 'val 5 folds'))
        cost = []
        for excel_path in val_paths_specific_const:
            excel_files = os.listdir(excel_path)
            for file in excel_files:
                if file.endswith('xlsx'):
                    path = os.path.join(excel_path, file)

                    read_total_sheet = pd.read_excel(path, sheet_name='total', index_col=0)

                    cost_val_list = []
                    for index in range(1, 6):
                        cost_val_list.append(read_total_sheet['OR real cost'].loc['val'+str(index)])
                    cost.append(cost_val_list)

        val_list_of_const_list.append(cost)

    print(val_list_of_const_list)
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    fig1.tight_layout(pad=3)

    barWidth = 0.1

    r1 = 0
    r2 = r1 + barWidth
    r3 = r2 + barWidth
    r4 = r3 + barWidth

    np_cost_1andhalf = np.array(val_list_of_const_list[1])
    np_cost_1 = np.array(val_list_of_const_list[2])
    np_cost_half = np.array(val_list_of_const_list[3])

    ax1.plot([r3-barWidth/2, r3-barWidth/2],  [0, 2], linestyle='dashed', color='black')

    ax1.bar(r1, np_cost_half[0].mean(), barWidth/2, color='gainsboro', yerr=np_cost_half[0].std(), capsize=2, edgecolor='grey', label='$n_4$ = 0.5%')
    ax1.bar(r2, np_cost_half[1].mean(), barWidth/2, color='gainsboro', yerr=np_cost_half[1].std(), capsize=2, edgecolor='grey')
    ax1.bar(r3, np_cost_half[2].mean(), barWidth/2, color='gainsboro', yerr=np_cost_half[2].std(), capsize=2, edgecolor='grey')
    ax1.bar(r4, np_cost_half[3].mean(), barWidth/2, color='gainsboro', yerr=np_cost_half[3].std(), capsize=2, edgecolor='grey')

    ax2.plot([r3-barWidth/2, r3-barWidth/2],  [0, 2], linestyle='dashed', color='black')

    ax2.bar(r1, np_cost_1[0].mean(), barWidth/2, color='darkgrey', yerr=np_cost_1[0].std(), capsize=2, edgecolor='grey', label='$n_4$ = 1%')
    ax2.bar(r2, np_cost_1[1].mean(), barWidth/2, color='darkgrey', yerr=np_cost_1[1].std(), capsize=2, edgecolor='grey')
    ax2.bar(r3, np_cost_1[2].mean(), barWidth/2, color='darkgrey', yerr=np_cost_1[2].std(), capsize=2, edgecolor='grey')
    ax2.bar(r4, np_cost_1[3].mean(), barWidth/2, color='darkgrey', yerr=np_cost_1[3].std(), capsize=2, edgecolor='grey')

    ax3.plot([r3-barWidth/2, r3-barWidth/2],  [0, 2], linestyle='dashed', color='black')

    ax3.bar(r1, np_cost_1andhalf[0].mean(), barWidth/2, color='dimgray', yerr=np_cost_1andhalf[0].std(), capsize=2, edgecolor='grey', label='$n_4$ = 1.5%')
    ax3.bar(r2, np_cost_1andhalf[1].mean(), barWidth/2, color='dimgray', yerr=np_cost_1andhalf[1].std(), capsize=2, edgecolor='grey')
    ax3.bar(r3, np_cost_1andhalf[2].mean(), barWidth/2, color='dimgray', yerr=np_cost_1andhalf[2].std(), capsize=2, edgecolor='grey')
    ax3.bar(r4, np_cost_1andhalf[3].mean(), barWidth/2, color='dimgray', yerr=np_cost_1andhalf[3].std(), capsize=2, edgecolor='grey')


    ax1.set_xlabel('Algorithms', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Cost', fontweight='bold', fontsize=10)
    ax1.set_xticks([r * barWidth for r in range(len(algo_list))], algo_list)
    # ax1.set_title('The cost result and STD for framework with constraint $n_4=0.5\%$')

    ax2.set_xlabel('Algorithms', fontweight='bold', fontsize=10)
    ax2.set_ylabel('Cost', fontweight='bold', fontsize=10)
    ax2.set_xticks([r * barWidth for r in range(len(algo_list))], algo_list)
    # ax2.set_title('The cost result and STD for framework with constraint $n_4=1\%$')

    ax3.set_xlabel('Algorithms', fontweight='bold', fontsize=10)
    ax3.set_ylabel('Cost', fontweight='bold', fontsize=10)
    ax3.set_xticks([r * barWidth for r in range(len(algo_list))], algo_list)
    # ax3.set_title('The cost result and STD for framework with constraint $n_4=1.5\%$')

    ax1.set_ylim(1.67, 1.82)
    ax2.set_ylim(1.67, 1.82)
    ax3.set_ylim(1.67, 1.82)

    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.show()
    # fig.savefig(os.path.join(args.path, 'Runs', 'Best_models', 'std_graphs.png'))


def test_cost_bars(args, root_dir, const_list, algo_list):
    mpl.use('TkAgg')
    test_list_of_const_list = []
    val_list_of_const_list = []
    for const in const_list:
        test_paths_specific_const = []
        for root_dir_specific in root_dir:
            test_paths_specific_const.append(os.path.join(root_dir_specific, const, 'test'))
        cost = []
        for excel_path in test_paths_specific_const:
            excel_files = os.listdir(excel_path)
            for file in excel_files:
                if file.endswith('xlsx'):
                    path = os.path.join(excel_path, file)
                    read_total_sheet = pd.read_excel(path, sheet_name='total', index_col=0)
                    cost.append(read_total_sheet.loc['OR real cost'].iloc[0])

        test_list_of_const_list.append(cost)

    algo = []
    ord_algo = []
    for c_list in range(len(test_list_of_const_list)):
        algo.append(test_list_of_const_list[c_list][0])
        ord_algo.append(test_list_of_const_list[c_list][1])

    print(algo)
    print(ord_algo)
    print(test_list_of_const_list)



    fig, ax1 = plt.subplots()
    fig.tight_layout(pad=3)

    # set width of bar
    barWidth = 0.2
    # Set position of bar on X axis
    br1 = range(len(const_list))
    br2 = [x + barWidth for x in br1]


    ax1.bar(br1, ord_algo, color='gainsboro', width=barWidth,
            edgecolor='grey', label=algo_list[1])
    ax1.bar(br2, algo, color='dimgray', width=barWidth,
            edgecolor='grey', label=algo_list[0])



    ax1.set_xlabel('Algorithms', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Cost', fontweight='bold', fontsize=10)

    xticks = [x + barWidth/4 for x in br1]

    ax1.set_xticks(xticks, ['W/o constraints', '$n_4=1.5\%$', '$n_4=1\%$', '$n_4=0.5\%$'])
    ax1.set_title('Comparison of cost results over different constraints on test dataset')

    ax1.set_ylim(1.4, 1.8)
    ax1.legend()


    plt.show()
    # fig.savefig(os.path.join(args.path, 'Runs', 'Best_models', 'std_graphs.png'))


def test_cost_bars_per_algo(args, root_dir, const_list, algo_list):
    mpl.use('TkAgg')
    test_list_of_const_list = []
    val_list_of_const_list = []
    for const in const_list:
        test_paths_specific_const = []
        for root_dir_specific in root_dir:
            test_paths_specific_const.append(os.path.join(root_dir_specific, const, 'test'))
        cost = []
        for excel_path in test_paths_specific_const:
            excel_files = os.listdir(excel_path)
            for file in excel_files:
                if file.endswith('xlsx'):
                    path = os.path.join(excel_path, file)
                    read_total_sheet = pd.read_excel(path, sheet_name='total', index_col=0)
                    cost.append(read_total_sheet.loc['OR real cost'].iloc[0])

        test_list_of_const_list.append(cost)


    fig, ax1 = plt.subplots()
    fig.tight_layout(pad=3)

    # set width of bar
    barWidth = 0.1
    # Set position of bar on X axis
    br1 = range(len(algo_list))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]


    ax1.bar(br1, test_list_of_const_list[0], color='gainsboro', width=barWidth,
            edgecolor='grey', label=const_list[0])
    ax1.bar(br2, test_list_of_const_list[1], color='dimgray', width=barWidth,
            edgecolor='grey', label=const_list[1])
    ax1.bar(br3, test_list_of_const_list[2], color='grey', width=barWidth,
            edgecolor='grey', label=const_list[2])
    ax1.bar(br4, test_list_of_const_list[3], color='lightgrey', width=barWidth,
            edgecolor='grey', label=const_list[3])


    ax1.set_xlabel('Algorithms', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Cost', fontweight='bold', fontsize=10)
    ax1.set_xticks(br1, algo_list)
    # ax1.set_title('Comparison of different constraints on cost test result')

    ax1.set_ylim(1.4, 1.8)
    ax1.legend()


    plt.show()
    fig.savefig(os.path.join(args.path, 'Runs', 'Best_models', 'std_graphs.png'))


def create_external_graphs(args):
    root_dir_dt_CE = os.path.join(args.path, r'Runs\Best_models\decision_tree')
    root_dir_dt_ord = os.path.join(args.path, r'Runs\Best_models\decision_tree_ordinal')
    root_dir_rf_CE = os.path.join(args.path, r'Runs\Best_models\random_forest')
    root_dir_rf_ord = os.path.join(args.path, r'Runs\Best_models\random_forest_ordinal')

    root_dir_dt = [root_dir_dt_CE, root_dir_dt_ord]
    root_dir_rf = [root_dir_rf_CE, root_dir_rf_ord]
    root_dir = [root_dir_dt_CE, root_dir_dt_ord, root_dir_rf_CE, root_dir_rf_ord]
    const_list = ['const 100% on class_0', 'const 1.5% on class_4', 'const 1% on class_4', 'const 0.5% on class_4']
    # const_list = ['const 100% on class_0']
    algo_list_dt = ['DT', 'DT_ORD']
    algo_list_rf = ['RF', 'RF_ORD']
    algo_list = ['DT', 'DT_ORD', 'RF', 'RF_ORD']
    # x_names = ['RF', 'ord_RF', r'RF $n_3=3\%$', r'ord_RF $n_3=3\%$']
    # paths_dt = []
    # paths_dt_ord = []
    # paths_rf = []
    # paths_rf_ord = []
    # for const in ['const 0.5% on class_4', 'const 1% on class_4', 'const 1.5% on class_4' 'const 0.5% on class_2_4', 'const 3% on class_3', 'const 100% on class_0']:
    #     paths.append(os.path.join(root_dir_dt_CE, const, 'test'))
    #     paths.append(os.path.join(root_dir_dt_ord, const, 'test'))


    # moves_bars(args, paths, x_names, '3')
    # steps_ml_or_bars(paths1, args, constraints, '4')
    # standard_deviation(args, root_dir, const_list, algo_list)
    test_cost_bars(args, root_dir_dt, const_list, algo_list_dt)
    test_cost_bars(args, root_dir_rf, const_list, algo_list_rf)
    # test_cost_bars_per_algo(args, root_dir_dt, const_list, algo_list_dt)
    # test_cost_bars_per_algo(args, root_dir_rf, const_list, algo_list_rf)
