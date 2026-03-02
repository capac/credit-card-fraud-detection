#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    balanced_accuracy_score,
    mutual_info_score
    )
from sklearn.model_selection import (
    cross_validate,
    train_test_split
    )
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
    )
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler

# start of analysis
start = time.perf_counter()

home_dir = Path.home()
data_path = home_dir / 'Programming/data/fraud-detection/data'
data_df = pd.read_csv(
    data_path / 'transactions_obf.csv', parse_dates=['transactionTime'],
    dtype={'availableCash': np.float64, 'transactionAmount': np.float64}
    )
data_df.sort_values(by='transactionTime', inplace=True)

col_names = [
    'eventId', 'accountNumber', 'merchantId', 'mcc',
    'merchantCountry', 'merchantZip', 'posEntryMode'
    ]
data_df[col_names] = data_df[col_names].astype('string')

print('Number of unique entries: ')
print(f'{data_df.nunique().sort_values(ascending=False)}\n')
print(f'Number of missing values:\n{data_df.isnull().sum()}\n')
percent_missing = 100*data_df.isnull().sum()/data_df.shape[0]
print(f'Percentage of missing values:\n{round(percent_missing, 1)}\n')

merchant_df = data_df.merchantZip.value_counts(dropna=False, normalize=True)
print(f'Sorted list of normalized merchantZip codes:\n{merchant_df[:10]}\n')

print(f"Percentage of NAs and 0s in 'merchantZip': "
      f"{round(100*((data_df.merchantZip == '0').sum() +
                    data_df.merchantZip.isna().sum())/len(data_df), 2)}%.")
print("Made decision to drop 'merchantZip' column\n")
data_df.drop('merchantZip', axis=1, inplace=True)

labels_df = pd.read_csv(
    data_path / 'labels_obf.csv', parse_dates=['reportedTime']
    )
labels_df.sort_values(by='reportedTime', inplace=True)

print(f'Number of unique entries in labels dataset:\n{labels_df.nunique()}')

# adding fraud case column from labels_df to data_df
data_df['fraudCase'] = data_df.eventId.isin(labels_df.eventId).astype(bool)
neg, pos = np.bincount(data_df['fraudCase'])
total = neg + pos
print('Total: {}\nPositive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))


def mutual_info(x):
    return mutual_info_score(x, data_df.fraudCase)


categorical_columns = list(data_df.select_dtypes('string'))
numerical_columns = list(data_df.select_dtypes('number'))

print('Mutual information score')
mutual_df = data_df[categorical_columns].apply(mutual_info)
print(f'{mutual_df.sort_values(ascending=False)}\n')

corr_df = data_df[numerical_columns].corr()
print(f'Correlation matrix of numerical features:\n{corr_df}\n')
data_df.drop(['eventId', 'merchantId', 'transactionTime'],
             axis=1, inplace=True)
print("Dropped 'eventId', 'merchantId and 'transactionTime'\n")
updated_categorical_columns = list(data_df.select_dtypes('string'))

print("One-hot encoding with 'DictVectorizer' class")
selected_columns = updated_categorical_columns + numerical_columns
dicts = data_df[selected_columns].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
dicts_arr = dv.fit_transform(dicts)
print(f'Shape of processed dataframe: {dicts_arr.shape}\n')

feature_names_columns = dv.get_feature_names_out(selected_columns)
processed_df = pd.DataFrame(dicts_arr, columns=feature_names_columns)
print('Splitting data into training and testing sets\n')

X = processed_df
y = data_df['fraudCase']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
    )

model_dict = {
    'logistic regression': LogisticRegression(max_iter=3000, random_state=0),
    'decision tree': DecisionTreeClassifier(max_depth=6, random_state=0),
    'random forest': RandomForestClassifier(
        max_depth=8, n_estimators=200, n_jobs=-1, random_state=0
        ),
    'histogram gradient boosting': HistGradientBoostingClassifier(
        max_depth=8, max_iter=400, random_state=0
        ),
    }

list_of_models = list(model_dict.keys())
print(f'Modeling data on {str(list_of_models)[1:-1]}:\n')

for name, model in model_dict.items():
    pipeline = make_pipeline(
        RandomUnderSampler(random_state=0),
        model
        )
    cv_results = cross_validate(
        pipeline, X_train, y_train, scoring="balanced_accuracy",
        return_train_score=True, return_estimator=True,
        n_jobs=-1, error_score='raise'
        )
    print(f"Balanced accuracy mean ± std. dev. for {name}: "
          f"{np.round(cv_results['test_score'].mean(), 3)} ± "
          f"{np.round(cv_results['test_score'].std(), 3)}"
          )

    scores = []
    for cv_model in cv_results["estimator"]:
        scores.append(
            balanced_accuracy_score(y_test, cv_model.predict(X_test))
            )
    print(
        f"Balanced accuracy mean ± std. dev. for {name} on test dataset: "
        f"{np.mean(scores):.3f} ± {np.std(scores):.3f}\n"
        )

end = time.perf_counter()
print(f"Total time: {round((end - start), 2)} seconds")
