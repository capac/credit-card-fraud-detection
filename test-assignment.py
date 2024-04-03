import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import balanced_accuracy_score
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data_path = Path().cwd() / 'data'
data_df = pd.read_csv(data_path / 'transactions_obf.csv',
                      parse_dates=['transactionTime'],
                      dtype={'availableCash': np.float64,
                             'transactionAmount': np.float64})
data_df.sort_values(by='transactionTime', inplace=True)
data_df[['eventId', 'accountNumber',
         'merchantId', 'mcc',
         'merchantCountry', 'merchantZip',
         'posEntryMode']] = data_df[['eventId', 'accountNumber',
                                     'merchantId', 'mcc',
                                     'merchantCountry', 'merchantZip',
                                     'posEntryMode']].astype('string')

print('Information on data_df dataframe')
print(data_df.info())
print('\n')
print(data_df.describe())
print('\n')
print('Number of unique entries: ')
print(f'{data_df.nunique().sort_values(ascending=False)}')
print('\n')
print(f'Number of missing values:\n{data_df.isnull().sum()}')
print('\n')
percent_missing = 100*data_df.isnull().sum()/data_df.shape[0]
print(f'Percentage of missing values:\n{round(percent_missing, 1)}')
print('\n')
merchant_df = data_df.merchantZip.value_counts(dropna=False, normalize=True)
print(f'Sorted list of normalized merchantZip codes:\n{merchant_df[:10]}')
print('\n')
print('''Percentage of NAs and 0s in 'merchantZip':''')
print(f'''{round(100*((data_df.merchantZip == '0').sum() +
                      data_df.merchantZip.isna().sum())/len(data_df), 2)}%.''')

print('''Made decision to drop 'merchantZip' column.''')
data_df.drop('merchantZip', axis=1, inplace=True)

data_path = Path().cwd() / 'data'
labels_df = pd.read_csv(data_path / 'labels_obf.csv',
                        parse_dates=['reportedTime'])
labels_df.sort_values(by='reportedTime', inplace=True)
print('\n')
print('Information on labels_df dataframe')
print(labels_df.info())
print('\n')
print(f'Number of unique entries in labels dataset:\n{labels_df.nunique()}')

data_df['fraudCase'] = data_df.eventId.isin(labels_df.eventId).astype(bool)
print('\n')
neg, pos = np.bincount(data_df['fraudCase'])
total = neg + pos
print('Total: {}\nPositive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))


def mutual_info(x):
    return mutual_info_score(x, data_df.fraudCase)


categorical_list = list(data_df.select_dtypes('string'))
numerical_list = list(data_df.select_dtypes('number'))

print('Mutual information score:')
mutual_df = data_df[categorical_list].apply(mutual_info)
print(f'{mutual_df.sort_values(ascending=False)}')
print('\n')
corr_df = data_df[numerical_list].corr()
print(f'Correlation matrix of numerical features:\n{corr_df}')
print('\n')
print('''Dropped 'eventId', 'transactionTime' and 'merchantId'.''')
data_df.drop(['eventId', 'transactionTime', 'merchantId'],
             axis=1, inplace=True)
categorical_list = list(data_df.select_dtypes('string'))

print('''One-hot encoding with 'DictVectorizer' class''')
selected_list = categorical_list + numerical_list
dicts = data_df[selected_list].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
dicts_arr = dv.fit_transform(dicts)
print(f'Shape of processed dataframe: {dicts_arr.shape}')

feature_names_list = dv.get_feature_names_out(selected_list)
processed_df = pd.DataFrame(dicts_arr, columns=feature_names_list)
print('Data split into training and testing data.')
X = processed_df
y = data_df['fraudCase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y,
                                                    random_state=42)

print('\n')
print('Ratio of non-fraud (False )and fraud (True) cases:')
print({key: round(value / len(y), 4) for key, value in Counter(y).items()})
print('\n')

processed_categorical_list = \
    set(feature_names_list) - set(['transactionAmount', 'availableCash'])

print('Modelling data on logistic regression, decision tree and random forest')
model_dict = {'logistic regression': LogisticRegression(random_state=0,
                                                        max_iter=4000),
              'decision tree': DecisionTreeClassifier(max_depth=6,
                                                      random_state=0),
              'random forest': RandomForestClassifier(n_jobs=-1,
                                                      random_state=0,
                                                      max_depth=6,
                                                      n_estimators=150)}

for name, model in model_dict.items():
    pipeline_ = make_pipeline(RandomUnderSampler(random_state=0), model)
    cv_results_from_pipeline_ = cross_validate(pipeline_, X_train, y_train,
                                               scoring="balanced_accuracy",
                                               return_train_score=True,
                                               return_estimator=True,
                                               n_jobs=-1,
                                               error_score='raise')
    print(f"Balanced accuracy mean +/- std. dev. for {name}: "
          f"{cv_results_from_pipeline_['test_score'].mean():.3f} +/- "
          f"{cv_results_from_pipeline_['test_score'].std():.3f}")
    scores = []
    for fold_id, cv_model in enumerate(cv_results_from_pipeline_["estimator"]):
        scores.append(balanced_accuracy_score(y_test,
                                              cv_model.predict(X_test)))
    print(f"Balanced accuracy mean +/- std. dev. for {name} on test dataset: "
          f"{np.mean(scores):.3f} +/- {np.std(scores):.3f}"
          f"\n")
