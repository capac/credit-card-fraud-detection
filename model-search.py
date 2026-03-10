#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd
import numpy as np
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    )
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

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
# as 'mcc', 'merchantCountry' and 'posEntryMode' to strings
category_list = [
    'eventId', 'accountNumber', 'merchantId', 'mcc',
    'merchantCountry', 'merchantZip', 'posEntryMode'
    ]
data_df[category_list] = data_df[category_list].astype('string')
labels_df['eventId'] = labels_df['eventId'].astype('string')
data_df['fraudCase'] = data_df.eventId.isin(labels_df.eventId)
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

# validation set: 9th and 10th month
X_val = X_train_full[len(X_train_full)-2*len_monthly_data_set:]
y_val = y_train_full[len(y_train_full)-2*len_monthly_data_set:]

# first test month
X_test_1 = X_test[:len_monthly_data_set]
y_test_1 = y_test[:len_monthly_data_set]
# second test month
X_test_2 = X_test[len_monthly_data_set:]
y_test_2 = y_test[len_monthly_data_set:]

print(f'Length of full training set (first 10 months): {len(X_train_full)}')
print(f'Length of full testing set (last 2 months): {len(X_test)}')
print(f'Length of training set (first 8 months): {len(X_train)}')
print(f'Length of validation set (9th & 10th months): {len(X_val)}')
print(f'len(X_train) + len(X_val) + len(X_test) == len(X): '
      f'{len(X_train)+len(X_val)+len(X_test) == len(X)}\n')

model_dict = {
    'logistic regression': LogisticRegression(max_iter=4000, random_state=0,),
    'decision tree': DecisionTreeClassifier(max_depth=6, random_state=0),
    'random forest': RandomForestClassifier(
        max_depth=8, n_estimators=200, n_jobs=-1,
        random_state=0, bootstrap=True,
        ),
    'histogram gradient boosting': HistGradientBoostingClassifier(
        max_depth=8, max_iter=400, random_state=0, early_stopping=True,
        ),
    }

list_of_models = list(model_dict.keys())
mod_list = str(list_of_models)[1:-1].split(', ')
new_list = ', '.join(mod_list[:-1]) + ' and ' + ' '.join(mod_list[-1:])
print(f'Modeling data on {new_list}:\n')


class DataSetEvaluation():
    def __init__(self, X_train, X_val, y_train, y_val):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

    def cv_evaluate(self):
        for name, model in model_dict.items():
            pipeline = make_pipeline(
                RandomUnderSampler(random_state=0),
                model
                )
            cv_res = cross_validate(
                pipeline, self.X_train, self.y_train,
                scoring="balanced_accuracy",
                return_train_score=True, n_jobs=-1,
                return_estimator=True, error_score='raise'
                )
            print(f"Balanced accuracy mean ± std. dev. for {name}: "
                  f"{np.round(cv_res['test_score'].mean(), 3)} ± "
                  f"{np.round(cv_res['test_score'].std(), 3)}")
            scores = []
            for cv_model in cv_res["estimator"]:
                scores.append(
                    balanced_accuracy_score(
                        self.y_val, cv_model.predict(self.X_val)
                        )
                    )
            print(f"Balanced accuracy mean ± std. dev. "
                  f"for {name} on validation dataset: "
                  f"{np.round(np.mean(scores), 3)} ± "
                  f"{np.round(np.std(scores), 3)}"
                  f"\n")


validation_eval = DataSetEvaluation(X_train, X_val, y_train, y_val)
validation_eval.cv_evaluate()
