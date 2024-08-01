import os
import pandas as pd
import numpy as np


def count_words(text, words):
    c = sum([text.count(word) for word in words])
    return c


# Load the data
filename = "results/doc2vec_10Years_RollingWindow.csv"
data = pd.read_csv(filename, parse_dates=['date'])
# Load fiscal policy terms
expansive = pd.read_csv('expansionary_terms_preprocessed.csv')['expansionary_terms'].values
restrictive = pd.read_csv('contractionary_terms_preprocessed.csv')['contractionary_terms'].values
benchmark1 = []
benchmark2 = []
for i, row in data.iterrows():
    n_exp = count_words(row.text_preprocessed_lemmatized, expansive)
    n_res = count_words(row.text_preprocessed_lemmatized, restrictive)
    benchmark1.append((n_exp - n_res) / len(row.text_preprocessed_lemmatized.split()))
    if (n_exp + n_res) == 0:
        benchmark2.append(0)
    else:
        benchmark2.append((n_exp - n_res) / (n_exp + n_res))
data['benchmark1'] = benchmark1
data['benchmark2'] = benchmark2

benchmark1_quarterly = data.groupby(['Year', 'Quarter']).mean()['benchmark1'].values
benchmark2_quarterly = data.groupby(['Year', 'Quarter']).mean()['benchmark2'].values
benchmarks_quarterly = pd.DataFrame({'Dictionary1': benchmark1_quarterly,
                                     'Dictionary2': benchmark2_quarterly},
                                    index=data.date.dt.to_period(freq='Q').unique())
# save the data
benchmarks_quarterly.to_csv("results/dictionary_benchmarks_quarterly.csv")
