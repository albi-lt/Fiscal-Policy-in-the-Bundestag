import pandas as pd
import numpy as np
import os

os.getcwd()

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('german')

# import custom preprocessing functions
from functions_text_preprocessing import *

# load data
print("Load the data ...")
data_all_lp = pd.read_csv("data/all_bundestag_speeches_replication_data.csv")

# filter out certain speeches: keep only 'MdB','Bundeskanzler','Bundesminister',
# 'Staatssekretär', 'Staatsminister'
print("Filter out certain speeches")
data_all_lp = data_all_lp[
    data_all_lp.Role.isin(['MdB', 'Bundesminister', 'Bundeskanzler', 'Staatssekretär', 'Staatsminister'])]

# data_all_lp  # 523282 Speeches

# pd.DataFrame(data_all_lp.text_length).describe(percentiles=[0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99,
#                                                            0.995])  # mean:359, median:94, 99,5%-Perzentil:3573
print("Filter out too long and too short speeches.")
# remove too short and too long speeches: Swearing-in, Interpellations etc.
data_all_lp = data_all_lp[(data_all_lp.text_length >= 100) & (
            data_all_lp.text_length <= np.quantile(data_all_lp.text_length, 0.995))].reset_index(drop=True)
print(f"The dataset now contains {data_all_lp.shape[0]} speeches.")  # 253014 speeches
print("Start text preprocessing ...")
# use text column and add preprocessed text columns (with and without lemmatizing)

# lemmatizing
lemmatized = lemmatize_texts(data_all_lp.text.values, 'de_core_news_lg')

data_all_lp = data_all_lp.replace('-\n', '', regex=True)  # otherwise words get seperated

# convert umlauts
data_all_lp['text_preprocessed'] = convert_umlauts_strings(data_all_lp.text)
# remove special chars, exclude tokens with one char and those with more than 30 char
data_all_lp['text_preprocessed'] = list(sent_to_words(data_all_lp.text_preprocessed))

# convert umlauts lemmatized
data_all_lp['text_preprocessed_lemmatized'] = convert_umlauts_strings(lemmatized)
# for lemmatized texts: remove special chars, exclude tokens with one char and those with more than 30 char
data_all_lp['text_preprocessed_lemmatized'] = list(sent_to_words(data_all_lp.text_preprocessed_lemmatized))

# remove standard nltk-stopwords
data_all_lp['text_preprocessed'] = remove_words([' '.join(i) for i in data_all_lp.text_preprocessed],
                                                convert_umlauts_strings(stop_words))

data_all_lp['text_preprocessed_lemmatized'] = remove_words(
    [' '.join(i) for i in data_all_lp.text_preprocessed_lemmatized],
    convert_umlauts_strings(stop_words))  # removes unlemmatized nltk stopwords

# domain-specific stopwords
# df_stopwords = pd.read_pickle('bundestags_stoppworter_preprocessed.pkl') #adjust path!
# lemmatized = lemmatize_texts([df_stopwords.stoppworter[i] for i in range(df_stopwords.shape[0])], 'de_core_news_lg') #lemmatize stopwords

# df_stopwords['lemmatized'] = lemmatized
# df_stopwords.to_pickle('data\stopwords_german_bundestag.pkl') #This file is already provided in the repository.

# limit the dataset to the period from 1960 onwards (because of the economic variables)
data_all_lp['date'] = pd.to_datetime(data_all_lp['date'], format='%d.%m.%Y')
df_bundestag_speeches_1960 = data_all_lp.loc[data_all_lp['date'].dt.year >= 1960]

# remove domain-specific stopwords
df_stopwords = pd.read_csv("data/stopwords_german_bundestag.csv")
df_bundestag_speeches_1960['text_preprocessed'] = remove_stopwords(df_bundestag_speeches_1960.text_preprocessed,
                                                                        df_stopwords.stopwords.values)
df_bundestag_speeches_1960['text_preprocessed_lemmatized'] = remove_stopwords(
    df_bundestag_speeches_1960.text_preprocessed_lemmatized, df_stopwords.lemmatized.values)

text_length_preprocessed = [len(i.split()) for i in
                            df_bundestag_speeches_1960.text_preprocessed]  # counts document lengths after removing stopwords
text_length_lemmatized = [len(i.split()) for i in df_bundestag_speeches_1960.text_preprocessed_lemmatized]

df_bundestag_speeches_1960['text_length_preprocessed'] = text_length_preprocessed
df_bundestag_speeches_1960['text_length_lemmatized'] = text_length_lemmatized

df_bundestag_speeches_1960['date_year'] = df_bundestag_speeches_1960.date.dt.year
df_bundestag_speeches_1960['date_quarter'] = df_bundestag_speeches_1960.date.dt.quarter

df_bundestag_speeches_1960.to_pickle('data\df_bundestag_speeches_1960_preprocessed.pkl')
