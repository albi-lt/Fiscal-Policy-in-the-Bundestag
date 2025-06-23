import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import collections
from collections import Counter
import itertools
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams
from nltk.stem.snowball import SnowballStemmer
import re
from tqdm import tqdm
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import date
import time
import pandas as pd
from tqdm import tqdm
import numpy as np

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
        # exclude tokens with one character and those with more than 30 characters

def remove_stopwords(texttoken, stopwordslist):
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
    list of strings: A list of tokenized documents with stopwords removed.
    """
    stops_removed = [[word for word in simple_preprocess(doc) if word not in stopwordslist] for doc in texttoken]
    return [' '.join(doc) for doc in stops_removed]

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




