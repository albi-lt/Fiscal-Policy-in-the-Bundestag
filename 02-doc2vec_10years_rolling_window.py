import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import multiprocessing
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from functions_doc2vec import *
# Get the number of CPU cores for multiprocessing
cores = multiprocessing.cpu_count()

# Import data
data = pd.read_pickle('data\df_bundestag_speeches_1960_preprocessed.pkl')
data.index = pd.PeriodIndex(data.date, freq='Q')

# Create a list of periods and sub-periods for the analysis
periods = pd.PeriodIndex(pd.date_range('1960-01-01', periods=247, freq='Q'))
subperiods = []
forecast_periods = []
for i in range(0, 207, 1):
    subperiods.append(periods[i:i + 40])
    forecast_periods.append(periods[i + 40])

# Add columns for similarity metrics to the data frame
data['similarity_expansive'] = np.nan
data['similarity_restrictive'] = np.nan

# Set parameters for the Doc2Vec model
set_window = 5
set_epochs = 100
set_vector_size = 100
dictionary = 'extended'

# Define the directory to save the trained models
models_directory = f'results/doc2vec_models/{set_window}Window_{set_epochs}Epochs_{set_vector_size}Size'
if not os.path.exists(models_directory):
    os.mkdir(models_directory)

# Load fiscal policy terms
expansive = pd.read_csv('expansionary_terms_preprocessed.csv')['expansionary_terms'].values
restrictive = pd.read_csv('contractionary_terms_preprocessed.csv')['contractionary_terms'].values

# Track the total start time
starttime = time.time()

# Loop through each subperiod
for i, period in enumerate(subperiods):
    print(f"Train model for the period {period[0]}-{period[-1]}.")
    start = time.time()

    # Train model based on ten years of data
    subset = data[data.index.isin(list(period))].reset_index(drop=True)
    sentences = subset.text_preprocessed_lemmatized
    sentences = [i.split() for i in sentences]
    documents = [TaggedDocument(doc, [id_]) for id_, doc in enumerate(sentences)]

    # Create and train the Doc2Vec model
    model = Doc2Vec(documents,
                    workers=cores - 1,
                    alpha=0.025,
                    vector_size=set_vector_size,
                    min_count=1,
                    epochs=set_epochs,
                    window=set_window,
                    dbow_words=1)
    print(f'The estimation took {time.time() - start} seconds.')

    # Save the trained model
    model.save(f'{models_directory}/doc2vec_subset_{period[0]}_{period[-1]}')

    print(
        f'Calculate similarities to fiscal policy vectors for the period {forecast_periods[i]}.')

    # Calculate "expansive" and "restrictive" vectors for the current period
    vocabulary = [word for index, word in enumerate(model.wv.index_to_key)]
    expansive_terms, expansive_wv = find_terms_vectors(expansive, model, vocabulary)
    restrictive_terms, restrictive_wv = find_terms_vectors(restrictive, model, vocabulary)
    vector_expansive = sum(expansive_wv) / len(expansive_wv)
    vector_restrictive = sum(restrictive_wv) / len(restrictive_wv)

    # Forecast for the next quarter
    forecast_df = data[data.index == forecast_periods[i]]
    document_vectors = [model.infer_vector(text.split()) for text in forecast_df.text_preprocessed_lemmatized]

    # Calculate similarities and store them in the DataFrame
    data.loc[forecast_periods[i], 'similarity_expansive'] = [cosine_similarity([vec], [vector_expansive])[0][0] for vec
                                                             in document_vectors]
    data.loc[forecast_periods[i], 'similarity_restrictive'] = [cosine_similarity([vec], [vector_restrictive])[0][0] for
                                                               vec in document_vectors]

# Track the total end time
endtime = time.time()
print(f'Overall the process took {(endtime - starttime) / 60} minutes.')

# Save the DataFrame to a CSV file
data.to_csv(f'results/doc2vec_10Years_RollingWindow.csv', index=False)
