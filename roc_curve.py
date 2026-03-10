#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_curve, roc_auc_score


home = Path.home()
data_path = home / 'Programming/data/fraud-detection/data'
work_dir = home / (
    'Programming/Python/machine-learning-exercises/credit-card-fraud-detection'
    )

data_df = pd.read_csv(
    data_path / 'transactions_obf.csv', parse_dates=['transactionTime'],
    dtype={'availableCash': np.int64, 'transactionAmount': np.float64}
    )
data_df.sort_values(by='transactionTime', inplace=True)

labels_df = pd.read_csv(
    data_path / 'labels_obf.csv', parse_dates=['reportedTime']
    )
labels_df.sort_values(by='reportedTime', inplace=True)

# output directory for plots
plot_dir = work_dir / 'plots'
plot_dir.mkdir(exist_ok=True, parents=True)

# matplotlib style file
mplstyle_file = work_dir / 'barplot-style.mplstyle'
plt.style.use(mplstyle_file)

# colormap
cmap = plt.cm.Paired.colors

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


def roc_curve_plot_with_auc(fpr, tpr, auc_val, work_dir):
    fig, axes = plt.subplots(figsize=(6, 5))
    axes.plot(fpr, tpr, '.-', color=cmap[1], label=f'AUC: {auc_val:.4f}')
    axes.plot([0, 1], [0, 1], '--', color=cmap[0])
    axes.set_xlim([-0.02, 1.0])
    axes.set_ylim([0.0, 1.02])
    axes.set_xlabel('False positive rate', fontsize=10)
    axes.set_ylabel('True positive rate', fontsize=10)
    axes.set_title('ROC curve for random forest classifier', fontsize=10)
    axes.legend(loc='lower right', fontsize=12)
    plt.setp(axes.get_xticklabels(), fontsize=10)
    plt.setp(axes.get_yticklabels(), fontsize=10)
    axes.tick_params(axis='y', which='major', pad=0)
    fig.tight_layout()
    plt.savefig(work_dir / 'plots/auc_plot.png',
                bbox_inches='tight', dpi=288)


# model pipeline
pipeline = make_pipeline(
    RandomUnderSampler(random_state=0),
    RandomForestClassifier(
        max_depth=8, n_estimators=200, n_jobs=-1,
        random_state=0, bootstrap=True,
        )
    )
pipeline.fit(X_train, y_train)

# roc curve
print("Calculating ROC plot...")
pipeline.fit(X_train, y_train)
y_proba = pipeline.predict_proba(X_val)[:, 1]
auc_val = roc_auc_score(y_val, y_proba)
fpr, tpr, _ = roc_curve(y_val, y_proba)
roc_curve_plot_with_auc(fpr, tpr, auc_val, work_dir)
