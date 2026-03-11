#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd
import numpy as np
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score
    )
from sklearn.model_selection import train_test_split


home = Path.home()
data_path = home / 'Programming/data/fraud-detection/data'
data_df = pd.read_csv(
    data_path / 'transactions_obf.csv', parse_dates=['transactionTime'],
    dtype={'availableCash': np.int64, 'transactionAmount': np.float64}
    )
data_df.sort_values(by='transactionTime', inplace=True)

labels_df = pd.read_csv(
    data_path / 'labels_obf.csv', parse_dates=['reportedTime']
    )
labels_df.sort_values(by='reportedTime', inplace=True)

# turned some category types represented by integrals such
# as 'mcc' 'merchantCountry' and 'posEntryMode' to strings
category_list = [
    'eventId', 'accountNumber', 'merchantId', 'mcc',
    'merchantCountry', 'merchantZip', 'posEntryMode'
    ]
data_df[category_list] = data_df[category_list].astype('string')
labels_df['eventId'] = labels_df['eventId'].astype('string')
# data_df['fraudCase'] = data_df.eventId.isin(labels_df.eventId)

data_df = data_df.merge(labels_df, on='eventId', how='left')
data_df['fraudCase'] = data_df['reportedTime'].apply(
    lambda x: 0 if pd.isnull(x) else 1
    )
data_df['fraudCase'] = data_df['fraudCase'].astype(bool)
data_df['merchantZip'] = data_df['merchantZip'].replace(
    {np.nan: 'Unknown', '0': 'Unknown'}
    )

categorical_list = list(data_df.select_dtypes('string'))
numerical_list = list(data_df.select_dtypes('number'))

data_df.drop(
    ['eventId', 'transactionTime', 'merchantId'],
    axis=1, inplace=True
    )
updated_categorical_list = list(data_df.select_dtypes('string'))

selected_list = updated_categorical_list + numerical_list
dicts = data_df[selected_list].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
dicts_arr = dv.fit_transform(dicts)
print(f'Shape of processed dataframe: {dicts_arr.shape}\n')

feature_names_list = dv.get_feature_names_out(selected_list)

processed_df = pd.DataFrame(dicts_arr, columns=feature_names_list)

y = data_df['fraudCase'].copy()
X = processed_df.copy()

# training set: first 10 months
len_monthly_data_set = int(X.shape[0]/12)
X_train_full = X[:len(X)-2*len_monthly_data_set]
y_train_full = y[:len(y)-2*len_monthly_data_set]

# test set: last two months
X_test = X[len(X)-2*len_monthly_data_set:]
y_test = y[len(y)-2*len_monthly_data_set:]

# training set: first 8 months
X_train = X_train_full[:len(X_train_full)-2*len_monthly_data_set]
y_train = y_train_full[:len(y_train_full)-2*len_monthly_data_set]

# first test month
X_test_1 = X_test[:len_monthly_data_set]
y_test_1 = y_test[:len_monthly_data_set]
# second test month
X_test_2 = X_test[len_monthly_data_set:]
y_test_2 = y_test[len_monthly_data_set:]

print('Model pipeline uses random under sampler and random '
      'forest classifier.\n')
print('Training set comes from the first 10 months of data, validation '
      'sets from the 9th\nand 10th months, and test sets come from '
      'the 11th and 12th month respectively.\n')

pipeline = make_pipeline(
    RandomUnderSampler(random_state=0),
    RandomForestClassifier(
        max_depth=8, n_estimators=200, n_jobs=-1,
        random_state=0, bootstrap=True,
        )
    )
pipeline.fit(X_train, y_train)


class ModelOverRandomDetection():
    '''
    Checks performance of fraud detection in model over random-transaction
    selection. The number of bootstrapped cases and the number of transaction
    checks can be changed for each instance of the class in the 'eval' method.
    '''

    def __init__(self, X_test, y_test, pipeline):
        self.X_test = X_test
        self.y_test = y_test
        self.pipeline = pipeline

    def eval(self, num_bootstrapped_cases=30, num_transaction_checks=400):
        self.num_transaction_checks = num_transaction_checks
        self.num_bootstrapped_cases = num_bootstrapped_cases
        self.percent_frauds_control = []

        # check about class index using 'classes_' attribute
        clf = self.pipeline.named_steps['randomforestclassifier']
        fraud_class_idx = list(clf.classes_).index(np.int64(1))
        print(f'fraud_class_idx: {fraud_class_idx}')
        y_score = self.pipeline.predict_proba(self.X_test)[:, fraud_class_idx]

        # test set
        sorted_indexes = np.argsort(y_score, stable=True)
        selected_indexes = sorted_indexes[-self.num_transaction_checks:]
        y_sel = self.y_test.iloc[selected_indexes]
        # print(f'len(y_sel): {len(y_sel)}')
        print(f'sum(y_sel): {sum(y_sel)}')
        # print(f'len(self.y_test): {len(self.y_test)}')
        print(f'sum(self.y_test): {sum(self.y_test)}')
        self.percent_frauds = 100*np.sum(y_sel)/np.sum(self.y_test)
        print(f'Model fraud detection rate on test set '
              f'using the 400 most-likely detections: '
              f'{np.round(self.percent_frauds, 2)}%')
        y_threshold = np.sort(y_score)[-self.num_transaction_checks]
        y_pred = (y_score >= y_threshold).astype(int)
        mdl_ba_score = balanced_accuracy_score(self.y_test, y_pred)
        print(f'Balanced accuracy score on test set: '
              f'{np.round(100*mdl_ba_score, 2)}%')
        recall = recall_score(self.y_test, y_pred, pos_label=1)
        print(f'Recall score on test set: '
              f'{np.round(100*recall, 2)}%')

        # control on random selection
        for n_tests in range(self.num_bootstrapped_cases):
            rng = np.random.RandomState(seed=n_tests)
            shuffled_index = rng.permutation(np.arange(0, len(self.y_test)))
            selected_indexes = shuffled_index[:num_transaction_checks]
            y_sel = self.y_test.iloc[selected_indexes]
            sens = 100*np.sum(y_sel)/np.sum(self.y_test)
            self.percent_frauds_control.append(sens)

        self.percent_frauds_control_mean = np.mean(self.percent_frauds_control)
        print(f'Average random fraud detection rate on '
              f'{self.num_bootstrapped_cases} bootstrapped test sets: '
              f'{np.round(self.percent_frauds_control_mean, 2)}%')

        imp_ratio = self.percent_frauds/self.percent_frauds_control_mean
        print(f'Improvement of model detection over average random detection '
              f'on test set: '
              f'{np.round(imp_ratio, 1)}x\n')


test_1_eval = ModelOverRandomDetection(X_test_1, y_test_1, pipeline)
test_1_eval.eval()

test_2_eval = ModelOverRandomDetection(X_test_2, y_test_2, pipeline)
test_2_eval.eval()

print(f'Model detection rate average: '
      f'{np.round(np.mean(
          [test_1_eval.percent_frauds,
           test_2_eval.percent_frauds]
           ), 2)}')

print(f'Random detection rate average: '
      f'{np.round(np.mean(
          [test_1_eval.percent_frauds_control_mean,
           test_2_eval.percent_frauds_control_mean]
           ), 2)}\n')


# training and testing model using complete, shuffled and stratified data set
X_train_tts, X_test_tts, y_train_tts, y_test_tts = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
    )
print('Training and testing data set from '
      'complete, shuffled and stratified data set.\n')


class CrossValidationCheck():
    def __init__(self, X_train, X_val, y_train, y_val):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

    def cv_eval(self, pipeline, num_transaction_checks=400):
        self.num_transaction_checks = num_transaction_checks
        self.pipeline = pipeline

        pipeline = clone(self.pipeline)
        pipeline.fit(self.X_train, self.y_train)

        # test set
        y_score = pipeline.predict_proba(self.X_val)[:, 1]
        sorted_indexes = np.argsort(y_score)
        selected_indexes = sorted_indexes[-self.num_transaction_checks:]
        y_sel = self.y_val.iloc[selected_indexes]
        self.percent_frauds = 100*np.sum(y_sel)/np.sum(self.y_val)

        print(f'Model fraud detection rate on test set '
              f'using the 400 most-likely detections: '
              f'{np.round(self.percent_frauds, 2)}%')

        y_threshold = np.sort(y_score)[-self.num_transaction_checks]
        y_pred = (y_score >= y_threshold).astype(np.int64)
        mdl_ba_score = balanced_accuracy_score(self.y_val, y_pred)
        print(f'Balanced accuracy score on test set: '
              f'{np.round(100*mdl_ba_score, 2)}%')
        recall = recall_score(self.y_val, y_pred, pos_label=1)
        print(f'Recall score on test set: '
              f'{np.round(100*recall, 2)}%')


cvc = CrossValidationCheck(X_train_tts, X_test_tts, y_train_tts, y_test_tts)
cvc.cv_eval(pipeline)
