import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.model_selection import train_test_split


def feature_engineering(path, file_name):
    df = pd.read_csv(os.path.join(path, file_name))

    df = df.drop(['customer_id', 'Name', 'security_no', 'referral_id', 'last_visit_time'], axis=1)

    # Date to days
    df['joining_date'] = pd.to_datetime(df['joining_date'])
    df['days_since_joined'] = df['joining_date'].apply(lambda x: (pd.Timestamp('today') - x).days)
    df.drop(['joining_date'], inplace=True, axis=1)

    # Drop all rows with nan
    df['medium_of_operation'][df['medium_of_operation'] == '?'] = np.nan
    df['avg_time_spent'][df['avg_time_spent'] < 0] = np.nan
    df['days_since_last_login'][df['days_since_last_login'] == -999] = np.nan
    df['joined_through_referral'][df['joined_through_referral'] == '?'] = np.nan
    df['avg_frequency_login_days'][df['avg_frequency_login_days'] == 'Error'] = np.nan
    df['avg_frequency_login_days'] = df['avg_frequency_login_days'].astype('float64')
    if 'train' in file_name:
        df['churn_risk_score'][df['churn_risk_score'] == -1] = np.nan

    df = df.dropna()
    df = df.reset_index(drop=True)

    # Categories to numbers
    categorical_columns_subset = []
    numerical_columns_subset = []
    for i in df.columns:
        if df[i].dtype == 'object':
            categorical_columns_subset.append(i)
        else:
            numerical_columns_subset.append(i)

    for col in categorical_columns_subset:
        df[col] = df[col].astype('category').cat.codes

    # Create labels column

    df['labels'] = (df['churn_risk_score'] - 1) * 2
    df = df.drop(['churn_risk_score'], axis=1)
    df.to_csv(os.path.join(path, 'eng_all_data.csv'))

    # rain rest split
    X = df.drop(['labels'], axis=1)
    y = df['labels']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1666, random_state=42)
    df_train = pd.concat([X_train, y_train], axis=1)
    df_train.to_csv(os.path.join(path, 'eng_train_data.csv'))
    df_test = pd.concat([X_test, y_test], axis=1)
    df_test.to_csv(os.path.join(path, 'eng_test_data.csv'))

    return df


def split_data(train, path, split_number, seed):
    labels = train['labels']
    features = train.drop(['labels'], axis=1)
    # features = train.values[:, 0:-1]
    # labels = train.values[:, -1].astype(int)
    skf = StratifiedKFold(n_splits=split_number, shuffle=True, random_state=seed)
    # split to folders by cross-validation
    for fold_ind, (train_index, val_index) in enumerate(skf.split(features, labels)):
        train.loc[train_index, 'job'] = 'train'
        train.loc[val_index, 'job'] = 'val'

        if not os.path.exists(os.path.join(path, 'splited_data')):
            os.mkdir(os.path.join(path, 'splited_data'))
        excel_name = os.path.join(path, 'splited_data', 'training_data' + str(fold_ind + 1) + '.xlsx')

        train.to_excel(excel_name)


def main():
    # path should include data.csv and splited_data empty folder
    split_number = 5
    seed = 42

    path = r'C:\Users\tal43\Documents\studies\pythonProject\ordinal-classic-ml-and-optimization'

    _ = feature_engineering(path, 'train_data.csv')
    # split_data(df, path, split_number, seed)


if __name__ == '__main__':
    main()
