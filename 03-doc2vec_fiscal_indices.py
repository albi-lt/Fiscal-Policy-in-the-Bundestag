import os
import pandas as pd
import numpy as np

# load the file containing fiscal sentiment values
filename = "results/doc2vec_10Years_RollingWindow.csv"
data = pd.read_csv(filename, parse_dates=['date'])
data['Year'] = data['date'].dt.year
data['Quarter'] = data['date'].dt.quarter

data['sentiment'] = data.similarity_expansive - data.similarity_restrictive
sentiment_full = data.groupby(['Year', 'Quarter']).mean()['sentiment'].values
sentiment_government = data[data.governing_Party == 1].groupby(['Year', 'Quarter']).mean()['sentiment'].values
sentiment_opposition = data[data.governing_Party == 0].groupby(['Year', 'Quarter']).mean()['sentiment'].values
data_quarterly = pd.DataFrame({'sentiment_gesamt': sentiment_full,
                               'sentiment_government': sentiment_government,
                               'sentiment_opposition': sentiment_opposition},
                              index=data.date.dt.to_period(freq='Q').unique())
# save the data
data_quarterly.to_csv("results/fiscal_sentiment_quarterly.csv")
