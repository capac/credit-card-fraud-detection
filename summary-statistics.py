#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import mutual_info_score

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

# turned some category types represented by integrals such as 'mcc'
# 'merchantCountry' and 'posEntryMode' to strings
category_list = [
    'eventId', 'accountNumber', 'merchantId', 'mcc',
    'merchantCountry', 'merchantZip', 'posEntryMode'
    ]
data_df[category_list] = data_df[category_list].astype('string')

unique_strings_df = data_df.select_dtypes('string').nunique()
print(f'Number of unique categorical entries:\n'
      f'{unique_strings_df.sort_values(ascending=False)}\n')

print(f'Number of missing values:\n{data_df.isnull().sum()}\n')

print(f'Percentage of missing values:\n'
      f'{round(100*data_df.isnull().sum()/data_df.shape[0], 1)}\n')

print(f'Sorted list of normalized merchantZip codes:\n'
      f'{data_df.merchantZip.value_counts(dropna=False, normalize=True)[:10]}'
      f'\n')

merchant_codes_sel = (
    np.sum(data_df.merchantZip == '0') +
    np.sum(data_df.merchantZip.isna())
    )
print(f"Percentage of NAs and 0s in 'merchantZip': "
      f"{round(100*(merchant_codes_sel)/data_df.shape[0], 2)}%.\n")

data_df['merchantZip'] = data_df['merchantZip'].replace(
    {np.nan: 'Unknown', '0': 'Unknown'}
    )

print(f'Sorted list of normalized merchantZip codes:\n'
      f'{data_df.merchantZip.value_counts(dropna=False, normalize=True)[:10]}'
      f'\n')

print(f'Sorted list of normalized posEntryMode codes:\n'
      f'{data_df.posEntryMode.value_counts(
          dropna=False, normalize=True
          )[:10]}\n')

labels_df['eventId'] = labels_df['eventId'].astype('string')

print(f'Number of unique entries in labels dataset:\n{labels_df.nunique()}\n')

data_df['fraudCase'] = data_df.eventId.isin(labels_df.eventId).astype(bool)

neg, pos = np.bincount(data_df['fraudCase'])
total = neg + pos
print(
    f'Total number of transactions: {total}\n'
    f'Number of non-fraudulent transactions: '
    f'{neg} ({100 * neg / total:.2f}% of total)\n'
    f'Number of fraudulent transactions: '
    f'{pos} ({100 * pos / total:.2f}% of total)\n'
    )

# information on the data set
total_num_accounts = data_df['accountNumber'].nunique()
fraud_df = data_df.loc[data_df.fraudCase == 1]
grouped = fraud_df.groupby('accountNumber')['transactionAmount']
fraud_amount_per_account_df = grouped.sum()
num_accounts_with_fraud = len(fraud_amount_per_account_df)
num_accounts_frauds_less_than_1000 = np.sum(fraud_amount_per_account_df < 1000)

results_dict = {
    'total_num_accounts': total_num_accounts,
    'num_accounts_with_fraud': num_accounts_with_fraud,
    'percent_fraud_per_transaction': np.round(
        100*((data_df['fraudCase'] == 1).sum()/data_df.shape[0]), 2
        ),
    'percent_accounts_with_fraud': np.round(
        100*num_accounts_with_fraud/total_num_accounts, 2
        ),
    'percent_frauds_less_than_1000_GBP': np.round(
        100*(num_accounts_frauds_less_than_1000/num_accounts_with_fraud), 2),
}

print(f'''Total number of accounts: {
    results_dict['total_num_accounts']}'''
    )
print(f'''Number of accounts with fraud: {
    results_dict['num_accounts_with_fraud']}'''
    )
print(f'''Percentage of fraud per transaction: {
    results_dict['percent_fraud_per_transaction']}%'''
    )
print(f'''Percentage of accounts with fraud: {
    results_dict['percent_accounts_with_fraud']}%'''
    )
print(f'''Percentage of accounts with less than £1000 of fraud: {
    results_dict['percent_frauds_less_than_1000_GBP']}%\n'''
    )


def mutual_info(x):
    return mutual_info_score(x, data_df.fraudCase)


categorical_list = list(data_df.select_dtypes('string'))
numerical_list = list(data_df.select_dtypes('number'))

cat_mutual_df = data_df[categorical_list].apply(mutual_info)
print(f'Mutual information score:\n'
      f'{cat_mutual_df.sort_values(ascending=False)}\n')


corr_df = data_df[numerical_list].corr()
print(f'Correlation matrix of numerical features:\n{corr_df}\n')
