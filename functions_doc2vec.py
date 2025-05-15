def find_terms_vectors(terms_list, model, vocabulary):
    '''
    :param terms_list: a list of relevant terms embeddings of which should be found in the model.
    :param model: a Doc2Vec or Word2Vec gensim model.
    :param vocabulary: vocabulary of the model.
    :return: a list of terms found in the vocabulary and a list of corresponding vectors.
    '''
    wv = []
    terms = []
    for i in terms_list:
        if len(i.split()) == 1:
            if i in vocabulary:
                wv.append(model.wv[i])
                terms.append(i)
        elif len(i.split()) == 2:
            if (i.split()[0] in vocabulary) & (i.split()[1] in vocabulary):
                wv.append(model.wv[i.split()[0]]+model.wv[i.split()[1]])
                terms.append(i)
    return terms, wv