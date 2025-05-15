import os
import pandas as pd
import numpy as np
import pickle
import nltk
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
np.random.seed(265) 

data = pd.read_pickle('data\df_bundestag_speeches_1960_preprocessed.pkl')

#### LDA ESTIMATION

list_data = []
for i in range(0,len(data.text_preprocessed)):
    list_data.append(data.text_preprocessed.iloc[i].split())
    
len(list_data) #235129

id2word = corpora.Dictionary(list_data)
# all words that appear less than 20 times in the corpus are removed from the vocabulary
id2word.filter_extremes(no_below=20, no_above=1.0, keep_n=None)
# Korpus
texts = list_data

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Save the Dict and Corpus
id2word.save('results/id2word-text_preprocessed.dict')
corpora.MmCorpus.serialize('results/corpus-text_preprocessed.mm', corpus)

# training lda model
lda_model_100 = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics= 100, 
                                           random_state=100,
                                           update_every=10,
                                           chunksize=10000,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

lda_model_100.save("lda_model_100")

#### COMBINE TOPICS WITH SENTIMENT

# determine endogenous and exogenous fiscal policy topics:

fis_endo = [3,6,22,27,45,63,67,79,91]
fis_exo = [7,14,15,16,17,24,28,31,34,37,38,40,44,47,50,53,54,58,59,64,70,72,90,96,97]


# extract topic weights from lda model
topic_dist_lda = lda_model_100.get_document_topics(corpus, minimum_probability = 0)
topicweights = []
for i in range(0,len(topic_dist_lda)):
    topicweights.append(topic_dist_lda[i])
dct = [dict(topicweights[i]) for i in range(0,len(topicweights))]

df = pd.DataFrame.from_dict(dct)
cols=df.columns.tolist()
cols.sort()

df_topicweights = df[cols]
df_topicweights = df_topicweights.fillna(0)

if not os.path.exists("LDA_100_Topic_Weights/"):
        os.makedirs("LDA_100_Topic_Weights/")
        
df_topicweights.to_csv("LDA_100_Topic_Weights/lda_topicweights.csv")
df_topicweights.to_pickle("LDA_100_Topic_Weights/lda_topicweights.pkl")

# limit documents as of 1970, # as of index 20440
df_topicweights_1970 = df_topicweights.iloc[20440:,].reset_index(drop=True)

df_topicweights_1970.to_csv("LDA_100_Topic_Weights/lda_topicweights_1970.csv")
df_topicweights_1970.to_pickle("LDA_100_Topic_Weights/lda_topicweights_1970.pkl")


# load sentiment obtained from doc2vec model (not aggregated)
filename = "results/doc2vec_10Years_RollingWindow.csv"
df_sentiment = pd.read_csv(filename, parse_dates=['date'])
df_sentiment['Year'] = df_sentiment['date'].dt.year
df_sentiment['Quarter'] = df_sentiment['date'].dt.quarter
df_sentiment['sentiment'] = df_sentiment.similarity_expansive - data.similarity_restrictive

# exogenous topic-sentiment
df_fis_exo = pd.DataFrame(fis_exo)
df_exo_topic_weights = pd.merge(df_topicweights_1970.T,df_fis_exo, left_index = True, right_on = 0)
df_exo_topic_weights = df_exo_topic_weights.rename(columns = {0:'exo_topics'}).set_index('exo_topics')
sum_df_exo_topic_weights = pd.Series(df_exo_topic_weights.sum(axis=0)) #exogenous topic weights for each doc

df_sentiment_1970 = data_all_lp[data_all_lp['date'] >= '1970-01-01'].reset_index(drop=True)
df_sentiment_1970['sum_exo_topic_weights'] = list(sum_df_exo_topic_weights)
df_sentiment_1970['exo_sentiment_weight'] = list(df_sentiment_1970.sentiment.multiply(df_sentiment_1970.sum_exo_topic_weights))

# endogenous topic-sentiment
df_fis_endo = pd.DataFrame(fis_endo)
df_endo_topic_weights = pd.merge(df_topicweights_1970.T, df_fis_endo, left_index=True, right_on=0)
df_endo_topic_weights = df_endo_topic_weights.rename(columns={0:'endo_topics'}).set_index('endo_topics')
# endogenous topic weights for each doc
sum_df_endo_topic_weights = pd.Series(df_endo_topic_weights.sum(axis=0))

df_sentiment_1970['sum_endo_topic_weights'] = list(sum_df_endo_topic_weights)
df_sentiment_1970['endo_sentiment_weight'] = list(df_sentiment_1970.sentiment.multiply(df_sentiment_1970.sum_endo_topic_weights))

# quarterly grouped
df_sentiment_1970_grouped = df_sentiment_1970[['endo_sentiment_weight','exo_sentiment_weight']].groupby(df_sentiment_1970['date'].dt.to_period('Q')).mean()

# quarterly sentiment scores and sentiment exogenous and sentiment endogenous
df_sentiment_1970_grouped.to_csv('results/quarterly_sentiment_decomposed.csv')