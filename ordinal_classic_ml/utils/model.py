from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from ordinal_classifier_master.OrdinalClassifier import OrdinalClassifier
import pandas as pd

def decision_tree(features_train, labels_train, depth, alpha=None, criterion=None):
    dtree_classifier = tree.DecisionTreeClassifier(max_depth=depth,
                                                   random_state=42)  # , min_samples_split=100, class_weight=[]
    dtree_classifier = dtree_classifier.fit(features_train, labels_train)
    return dtree_classifier


def random_forest(features_train, labels_train, depth, alpha=None, criterion=None):
    rforest_classifier = RandomForestClassifier(max_depth=depth, random_state=42)
    rforest_classifier.fit(features_train, labels_train)
    return rforest_classifier


def adaboost(features_train, labels_train, depth, alpha=None, criterion=None):
    adaboost_classifier = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=depth), random_state=42)
    adaboost_classifier.fit(features_train, labels_train)
    return adaboost_classifier


def catboost(features_train, labels_train, depth, alpha, criterion=None):
    # catboost_classifier = CatBoostClassifier(iterations=2, depth=2, learning_rate=1, loss_function='Logloss', thread_count=-1, verbose=0)
    catboost_classifier = CatBoostClassifier(iterations=alpha, depth=depth, thread_count=-1, verbose=0)
    catboost_classifier.fit(features_train, labels_train)
    return catboost_classifier


def xgboost(features_train, labels_train, depth, alpha, criterion=None):
    xgboost_classifier = XGBClassifier(max_depth=depth, eta=alpha)
    xgboost_classifier.fit(features_train, labels_train)
    return xgboost_classifier


def decision_tree_ordinal_old(features_train, labels_train, depth, alpha=None, criterion=None):
    order = range(0, 21)
    dtree_classifier = OrdinalClassifier(order, tree.DecisionTreeClassifier, {})
    labels_train = pd.qcut(labels_train, len(order), labels=order)
    dtree_classifier.fit(features_train, labels_train)
    return dtree_classifier


def random_forest_ordinal_old(features_train, labels_train, depth, alpha=None, criterion=None):
    order = range(0, 21)
    rforest_classifier = OrdinalClassifier(range(0, 21), RandomForestClassifier, {'n_estimators': 10})
    labels_train = pd.qcut(labels_train, len(order), labels=order)
    rforest_classifier.fit(features_train, labels_train)
    return rforest_classifier


def adaboost_ordinal_old(features_train, labels_train, depth, alpha=None, criterion=None):
    order = range(0, 21)
    adaboost_classifier = OrdinalClassifier(range(0, 21), AdaBoostClassifier, {'n_estimators': 10})
    labels_train = pd.qcut(labels_train, len(order), labels=order)
    adaboost_classifier.fit(features_train, labels_train)
    return adaboost_classifier


def decision_tree_ordinal(features_train, labels_train, depth, alpha, criterion):
    dtree_classifier = tree.DecisionTreeClassifier(criterion=criterion, random_state=42, WIGR_power=alpha,
                                                   max_depth=depth)  # , min_samples_split=100, class_weight=[]
    dtree_classifier = dtree_classifier.fit(features_train, labels_train)
    return dtree_classifier


# criterion options: 'WIGR_min', 'WIGR_max', 'WIGR_EV', 'WIGR_mode', 'WIGR_EV_fix', 'entropy'
def random_forest_ordinal(features_train, labels_train, depth, alpha, criterion):
    rforest_classifier = RandomForestClassifier(criterion=criterion, random_state=42, WIGR_power=alpha,
                                                max_depth=depth)  # , n_estimators=10000) #, max_leaf_nodes=500)
    rforest_classifier.fit(features_train, labels_train)
    return rforest_classifier


def adaboost_ordinal(features_train, labels_train, depth, alpha, criterion):
    adaboost_classifier = AdaBoostClassifier(
        tree.DecisionTreeClassifier(criterion=criterion, random_state=42, WIGR_power=alpha, max_depth=depth))
    adaboost_classifier.fit(features_train, labels_train)
    return adaboost_classifier




def ml_model_fit(features_train, labels_train, algo, depth, alpha, crit):
    algorithm = decision_tree
    if algo == 'decision_tree':
        algorithm = decision_tree
    elif algo == 'random_forest':
        algorithm = random_forest
    elif algo == 'adaboost':
        algorithm = adaboost
    elif algo == 'catboost':
        algorithm = catboost
    elif algo == 'xgboost':
        algorithm = xgboost
    elif algo == 'decision_tree_ordinal':
        algorithm = decision_tree_ordinal
    elif algo == 'random_forest_ordinal':
        algorithm = random_forest_ordinal
    elif algo == 'adaboost_ordinal':
        algorithm = adaboost_ordinal
    # elif algo == 'decision_tree_ordinal_old':
    #     algorithm = decision_tree_ordinal_old
    # elif algo == 'random_forest_ordinal_old':
    #     algorithm = random_forest_ordinal_old
    # elif algo == 'adaboost_ordinal_old':
    #     algorithm = adaboost_ordinal_old

    model = algorithm(features_train, labels_train, depth, alpha, crit)
    return model


def ml_model_predict(model, test_features):
    predict_proba = model.predict_proba(test_features)
    predict_hard = model.predict(test_features)
    return predict_proba, predict_hard
