This script contains functions to use for text preprocessing.

####
def convert_umlauts(dataframe,textcolumn):
    """
    Convert German umlauts in a text column of a DataFrame to a dot-free notation.

    This function replaces the German umlauts in the specified text column with 
    equivalent ASCII representations.

    Parameters:
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing the text column to process.
    textcolumn : str
        The name of the column in the DataFrame where the text needs to be converted.

    Returns:
    -------
    pandas.DataFrame
        The DataFrame with the converted text column.
    """
    dataframe[textcolumn].replace('Ä','AE',regex=True, inplace = True)
    dataframe[textcolumn].replace('ä','ae',regex=True, inplace = True)
    dataframe[textcolumn].replace('Ü','UE',regex=True, inplace = True)
    dataframe[textcolumn].replace('ü','ue',regex=True, inplace = True)
    dataframe[textcolumn].replace('Ö','OE',regex=True, inplace = True)
    dataframe[textcolumn].replace('ö','oe',regex=True, inplace = True)
    dataframe[textcolumn].replace('ß','ss',regex=True, inplace = True)

    return dataframe

####

def convert_umlauts_strings(texts):

	 """
    Convert German umlauts in a list of strings to a dot-free notation.

    This function replaces German umlauts with equivalent ASCII representations
    in each string within the provided list.

    Parameters:
    ----------
    texts : list of str
        A list of strings where the umlauts need to be converted.

    Returns:
    -------
    list of str
        A list of strings with converted umlauts.
    """
	
    mapping = {ord(u"Ü"): u"Ue", ord(u"ü"): u"ue", ord(u"ß"): u"ss", ord(u"ä"): u"ae", ord(u"Ä"): u"Ae",
               ord(u"ö"): u"oe", ord(u"Ö"): u"Oe"}
    converted_texts = [i.translate(mapping) for i in texts]

    return converted_texts
####

############
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

def sent_to_words(sentences):
	"""

    This function tokenizes and preprocesses each sentence, excluding tokens with one 
    character and those with more than 30 characters.

    Parameters:
    ----------
    sentences : list of str
        A list of sentences/documents to be tokenized and processed.

    Yields:
    ------
    list of str
        A generator yielding lists of words for each sentence/document.
    """
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True,max_len = 30))  
        #exclude tokens with one character and those with more than 30 characters


############
def count_tokens(dataframe, textcolumn, returncolumn):

	"""
    Count the number of tokens in a specified text column and store the result in a new column.

    This function calculates the number of tokens in each entry of the specified text column and 
    appends this information to a new column.

    Parameters:
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing the text data.
    textcolumn : str
        The name of the column containing the text to be analyzed.
    returncolumn : str
        The name of the column to store the token counts.

    Returns:
    -------
    pandas.DataFrame
        The DataFrame with the new column containing token counts.
    """
	
    no_of_tokens = []
    for i in range(0,len(dataframe)):
        no_of_tokens.append(len(dataframe[textcolumn][i]))

    dataframe[returncolumn] = no_of_tokens
    return dataframe
############

def get_space_token(dataframe, tokenizedcolumn, new_columnname):

	"""
    Transform tokenized lists into a space-separated string and store it in a new column.

    This function converts lists of tokens in the specified column into a single string 
    with tokens separated by spaces and stores the result in a new column.

    Parameters:
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing the tokenized text.
    tokenizedcolumn : str
        The name of the column containing tokenized text (lists of strings).
    new_columnname : str
        The name of the column to store the space-separated token strings.

    Returns:
    -------
    pandas.DataFrame
        The DataFrame with the new column containing space-separated token strings.
    """
	
    space_tokens = []
    for i in dataframe[tokenizedcolumn]:
            a = ','.join(i)
            b= a.replace(',',' ')
            space_tokens.append(b)

    dataframe[new_columnname] = space_tokens
    return dataframe
###########

import collections
from collections import Counter
import itertools
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

def get_token_statistics(dataframe,tokencolumn,spacetokencolumn,savepath):

	"""
    Calculate token statistics including frequency and TF-IDF scores and save to a CSV file.

    This function computes the frequency and TF-IDF scores for tokens in the given column 
    and stores the results in a CSV file.

    Parameters:
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing tokenized text data.
    tokencolumn : str
        The name of the column containing the list of tokens.
    spacetokencolumn : str
        The name of the column containing space-separated tokens.
    savepath : str
        The path to save the CSV file with token statistics.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with token statistics including frequency and TF-IDF scores.
    """
	
    #Top tokens
    flat = itertools.chain.from_iterable(dataframe[tokencolumn])

    corpus = list(flat)
    mc = collections.Counter(corpus).most_common() #if integer is passed here, then e.g. top 1000 tokens are calculated

    list_mc_tokens = [list(ele) for ele in mc]
    list_mc_tokens_correct = list(itertools.chain.from_iterable(list_mc_tokens))
    del list_mc_tokens_correct[1::2] #delete every second element strating from index 1
    top_tokens = list_mc_tokens_correct
    table = list_mc_tokens
    df_top_tokens = pd.DataFrame(table)
    df_top_tokens.columns= ['Vocabulary','Frequency']

    #idf-scores
    v = TfidfVectorizer()
    x = v.fit_transform(dataframe[spacetokencolumn]) #tokens must be separated by a space and not by a comma

    v.vocabulary_

    feature_names = v.get_feature_names()
    idfwert = v.idf_

    df_idf = pd.DataFrame()
    df_idf['Vocabulary'] = feature_names
    df_idf['idf_score'] = idfwert

    df_token_statistics = pd.merge(df_top_tokens,df_idf, how = 'inner', on = 'Vocabulary')
    df_token_statistics.to_csv(savepath)
    return df_token_statistics

############
from nltk.util import ngrams

#n = 2 bigram, n=3 trigram
def ngramconvert(dataframe,n,space_token, outputtoken):

	"""
    Convert space-separated tokens into n-grams and store them in a new column.

    This function generates n-grams from space-separated tokens and stores the resulting 
    n-grams in a specified column of the DataFrame.

    Parameters:
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing space-separated tokens.
    n : int
        The number of elements in each n-gram (e.g., 2 for bigrams, 3 for trigrams).
    space_token : str
        The name of the column containing space-separated tokens.
    outputtoken : str
        The name of the column to store the n-grams.

    Returns:
    -------
    pandas.DataFrame
        The DataFrame with the new column containing n-grams.
		
	"""
	
    docs_ngram_tuples = dataframe[space_token].apply(lambda sentence: list(ngrams(sentence.split(), n)))
    preprocessed_bigram_list = []
    for i in range(0,len(dataframe)):
        preprocessed_bigram_list.append(list(map('_'.join, docs_ngram_tuples[i])))

    dataframe[outputtoken] = preprocessed_bigram_list
    return dataframe
###########

from nltk.stem.snowball import SnowballStemmer
def snowballstem_tokens(dataframe, text_token_column):

	"""
    Apply Snowball stemming to tokens in a specified column and store results in a new column.

    This function uses the Snowball stemmer to process each token in the specified column 
    and stores the stemmed tokens in a new column.

    Parameters:
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing the tokenized text.
    text_token_column : str
        The name of the column containing the tokenized text (list of strings).

    Returns:
    -------
    pandas.DataFrame
        The DataFrame with a new column containing stemmed tokens.
    """
	
    Stemmer=SnowballStemmer("german")

    stemmed_docs = []
    for i in dataframe[text_token_column]:
        stemmed_doc = []
        for j in i:
            stemmed_doc.append(Stemmer.stem(j))
        stemmed_docs.append(stemmed_doc)

    dataframe['stemmed_token'] = stemmed_docs
    return dataframe

###########

def create_custombigram(custom_bigramlist,dataframe,tokencolumn,outputcolumn):

	"""
	Merge specified bigrams in a tokenized text column of a DataFrame.
	
	Parameters:
	-----------
	custom_bigramlist : list of str
		Bigrams to merge, e.g. ['word1_word2']
	dataframe : pandas.DataFrame
		DataFrame with the tokenized text data.
	tokencolumn : str
		Column name with tokenized text.
	outputcolumn : str
		Column name for output with merged bigrams.
	
	
	Returns:
	--------
	pandas.DataFrame: DataFrame with a new column containing merged bigrams.
	"""
	
    bigrams_set = set(custom_bigramlist)
    bigram_corpus = []
    for doc in dataframe[tokencolumn]:
        bigram_doc = []
        for j in range(len(doc)):
            if (j>0 and doc[j-1]+"_"+doc[j] in bigrams_set):
                bigram_doc.pop()
                bigram_doc.append(doc[j-1]+"_"+doc[j])
            else:
                bigram_doc.append(doc[j])

        bigram_corpus.append(bigram_doc)

    dataframe[outputcolumn] = bigram_corpus
    return dataframe
	
############
def gensim_filter_extremes(no_below, no_above, keep_n, id2word):
    import numpy as np
    """
    adjusted gensim function to make it similar to sklearn

    id2word: gensim.corpora.Dictionary
    no_below: fraction of total corpus
    no_above: "-"
	keep_n: int

    Filter out tokens that appear in

    1. less than `no_below` documents (fraction of total corpus)
    2. more than `no_above` documents (fraction of total corpus size 
    3. after (1) and (2), keep only the first `keep_n` most frequent tokens (or
       keep all if `None`).

    After the pruning, shrink resulting gaps in word ids.

    **Note**: Due to the gap shrinking, the same word may have a different
    word id before and after the call to this function!
    """
    no_above_abs = int(
        no_above * id2word.num_docs)  # convert fractional threshold to absolute threshold #sklearn (?) schauen, ob ich hier auch aufrunden sollte
    no_below_abs = int(np.ceil(no_below * id2word.num_docs))  # da in sklearn aufgerundet wird np.ceil nutzen

    # determine which tokens to keep
    good_ids = (
        v for v in iter(id2word.token2id.values())
        if no_below_abs <= id2word.dfs.get(v, 0) <= no_above_abs)
    good_ids = sorted(good_ids, key=id2word.dfs.get, reverse=True)
    if keep_n is not None:
        good_ids = good_ids[:keep_n]
    bad_words = [(id2word[id], id2word.dfs.get(id, 0)) for id in set(id2word).difference(good_ids)]

    # do the actual filtering, then rebuild dictionary to remove gaps in ids
    id2word.filter_tokens(good_ids=good_ids)
    return id2word
##########

def remove_stopwords(texttoken,stopwordslist):
	"""
    Remove stopwords from a list of tokenized text documents.

    Parameters:
	-----------
    texttoken : list of list of str
		A list where each element is a list of tokens (words) representing a document.
    stopwordslist: list of str
		A list of stopwords to be removed from the text documents.

    Returns:
	--------
    list of list of str: A list of tokenized documents with stopwords removed.

    return [[word for word in simple_preprocess(str(doc)) if word not in stopwordslist] for doc in texttoken]

##########
def make_bigrams(texts,bigram_mod):
	"""
    Apply a gensim bigram model to a list of tokenized text documents to create bigrams.

    Parameters:
	-----------
    texts: list of list of str
		A list where each element is a list of tokens (words) representing a document.
    bigram_mod: gensim.models.phrases.Phraser
		A trained bigram model from gensim used to detect bigrams in the text.

    Returns:
	--------
    list of list of str: A list of tokenized documents with bigrams included.
    return [bigram_mod[doc] for doc in texts]

##########

import re
from tqdm import tqdm
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import date
import time
import pandas as pd
from tqdm import tqdm
import numpy as np


clean_texts = lambda x: re.sub("[^abcdefghijklmnopqrstuvwxyzäöüß& ']", " ", str(x).lower()).strip()


def n_elements_tokens(texts, n = 2):
    tokens = []
    for text in tqdm(texts):
        n_elements = [i for i in text.split() if len(i)==n]
        tokens.append(n_elements)

    tokens = [item for sublist in tokens for item in sublist]
    return np.unique(tokens)


def keep_word_length(texts, word_min=2, word_max =20):
    """
    This function removes too short/long words.
    :param texts: a list containing the texts.
    :param word_min: an integer defining the minimum length of the words.
    :param word_max: an integer defining the maximum length of the words.
    :return: texts containing containing the words of the defined length.
    """
    short_words_removed = []
    for text in tqdm(texts):
        text = list(filter(lambda word: len(word.lower()) > word_min, text.split()))
        text = list(filter(lambda word: len(word.lower()) < word_max, text))
        #stops_removed = list(filter(lambda word: word.lower() not in stopwords_merged, text))
        short_words_removed.append(' '.join(text))
    return short_words_removed


def lemmatize_texts(texts, spacy_model):
    '''
    This function lemmatizes texts.
    :param texts: texts to be lemmatized.
    :param spacy_model: spacy model to be used, e.g. "en_core_web_sm", "de_core_news_sm". The language models should be downloaded first. 
    :return: a list containing lemmatized texts.
    '''
    import spacy
    nlp_model = spacy.load(spacy_model)
    pp = []
    print('Start lemmatization...')
    t0 = time.time()
    for i in tqdm(range(len(texts))):
        text = " ".join([token.lemma_ for token in nlp_model(texts[i])])#
        pp.append(text)
    t1 = time.time()
    print('Finished lemmatization. Process took', t1 - t0, 'seconds')
    return pp


def remove_words(texts,words):
    '''
    This function removes specified words.
    :param texts: texts to be processed.
    :param words: a list defining terms to be removed.
    :return: texts without specified words.
    '''
    docs = []
    for doc in tqdm(texts):
        doc_cleaned = list(filter(lambda word: word.lower() not in words, doc.split()))
        docs.append(' '.join(doc_cleaned))
    return docs


def remove_tags(texts,language,
                tags_de =['DET','CONJ','SCONJ', 'PRON', 'ADP', 'ADV','PROPN'],
                tags_en=['DET','CCONJ','SCONJ', 'PRON','ADP', 'ADV', 'PROPN']):

    '''
    This function removes specific words based on their syntactic characteristics.
    For this method, use cleaned but NOT lemmatized texts. Do lemmatization after removing specific tags.
    :param texts: a list containing texts.
    :param language: a list defining the language of the the texts.
    :param tags_de: a list defining which parts of speech should be removed for the German language.
    :param tags_en: a list defining which parts of speech should be removed for the English language.
    :return: a list containing the texts without specified tags and a list of identified stopwords tags.
    '''
    assert len(texts) == len(language)
    cleaned_docs = []
    stops = []
    for i in tqdm(range(len(texts))):
        if language[i]=='de':
            tags = tags_de
            doc = nlp_de(texts[i])
            for t in doc:
                if t.pos_ not in tags:
                    cleaned_doc.append(t.orth_)
                elif t.pos_ in tags:
                    stops.append(t.orth_)
        elif language[i]=='en':
            tags = tags_en
            doc = nlp_en(texts[i])
            for t in doc:
                if t.pos_ not in tags:
                    cleaned_doc.append(t.orth_)
                elif t.pos_ in tags:
                    stops.append(t.orth_)
        cleaned_docs.append(' '.join(cleaned_doc))
    return cleaned_docs, stops

def learn_bigrams(docs, count=10, quantile = 0.95):
    '''
    This functions uses gensim's module Phrases to learn bigrams in a given text corpus.
    1. For given count, find all possible bigrams.
    2. Consider the scores of all possible bigrams, set the threshold at the defined quantile.

    :param docs: a list of documents.
    :param count: an iteger defining how many times a bigram should at least occur.
    :param quantile: a float defining the threshold for bigrams to be considered.
    :return: a list of learned bigrams as well as a threshold value (score).
    '''
    from gensim.test.utils import datapath
    #from gensim.models.word2vec import Text8Corpus
    from gensim.models.phrases import Phrases, Phraser
    # set threshold as 95% quantile of all the score values by the given min_count value, 134.23485847703284
    phrases = Phrases(docs, min_count=count, threshold=0.0001)
    bigram = Phraser(phrases)
    score_values = list(bigram.phrasegrams.values())
    learned_bigrams = []
    for b in bigram.phrasegrams.items():
        learned_bigrams.append(b)
    collocations = [learned_bigrams[i][0][0].decode("utf-8") + ' ' + learned_bigrams[i][0][1].decode("utf-8")
                   for i in range(len(learned_bigrams))]
    coll_len = len([collocations[i] for i in np.where(score_values > np.quantile(score_values, quantile))[0]])
    print('There are '+str(coll_len)+' collocations.')
    coll_final = ', '.join([collocations[i] for i in np.where(score_values > np.quantile(score_values, quantile))[0]])
    threshold_value = np.quantile(score_values, quantile)
    return coll_final, threshold_value


def popularity_based_prefiltering(texts, min_df, max_df):
    '''
    This function removes too rare/common words and returns preprocessed texts.
    :param texts: a list containing texts to be preprocessed.
    :param min_df: the lower threshold.
    :param max_df: the upper threshold.
    :return: a list containing preprocessed texts.
    '''

    tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df)
    # fit on dataset
    tfidf.fit([x.lower() for x in texts])
    # get vocabulary
    vocabulary = set(tfidf.vocabulary_.keys())
    print(len(vocabulary), 'words in the vocabulary')
    pp = []
    for text in texts:
        rare_removed = list(filter(lambda word: word.lower() in vocabulary, text.split()))
        string = ' '.join(rare_removed)
        pp.append(string.lower())
    return pp



