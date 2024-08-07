# Fiscal-Policy-in-the-Bundestag

This project contains all the necessary code and instructions to replicate the results presented in our study.


## About the Study

This study explores the influence of parliamentary debates on fiscal policy in Germany, using a novel dataset of speeches from the Bundestag.


### Objective:

Analyze the impact of fiscal sentiment in parliamentary debates on government spending and macroeconomic outcomes using data from 1960 to 2021.

### Methodology:

*Embedding-Based Approach:*

Represents words and documents in a shared vector space to measure fiscal sentiment from contractionary to expansionary.


*Vector Autoregressive Models:*

Analyzes how changes in fiscal sentiment affect government spending and other macroeconomic variables.


### Key Findings:

- 📈 **Changes in fiscal sentiment influence government spending.**
- 📈 **Fiscal sentiment has significant macroeconomic effects.**
- 📈 **Parliamentary debates provide insights into government spending shocks.**

## Replication Data

The underlying dataset on which all analyses are based can be downloaded here.

bundestag_speeches = pd.read_csv('data/all_bundestag_speeches_replication_data.csv', index_col = 'Unnamed: 0')

## Prerequisites

Use the provided `requirements.txt` file to install all necessary dependencies.

## Step 1: Preprocessing
### File: `01-preprocessing.py`

This script performs standard preprocessing on the Bundestag speech corpus (filtering relevant speeches, tokenization, stop-word removal etc.). Relevant functions used in this script are outsourced in `functions_text_preprocessing.py`.

## Step 2: Fiscal Policy Dictionary

### Files:
- `expansionary_terms_preprocessed.csv`
- `contractionary_terms_preprocessed.csv`

These csv. files contain the dictionary of fiscal-policy related words, categorized into expansionary and contractionary terms.

## Step 3: Train Doc2Vec Models

### File: `02-doc2vec_10years_rolling_window.py`

This script trains Doc2Vec models over defined 10-year training periods. It also infers vectors for forecast periods and construct fiscal policy vectors.

## Step 4: Construct Fiscal Sentiment Indices

### File: `03-doc2vec_fiscal_indices.py`

This script constructs fiscal sentiment indices using the vectors generated in step 3.

## Step 5: Calculate Dictionary-Based Alternatives

### File: `04-dictionary_based_fiscal_indices.py`

This script provides an alternative method to construct fiscal indices using a dictionary-based approach.

## Step 6: Topic Modelling to obtain endogenous/exogenous fiscal-policy related sentiment

### File: `05-topic_modelling.py`
This script trains an LDA topic model to obtain endogenous/exogenous fiscal-policy-related sentiment. Indices generated in Step 4 are thus split into exogenous and endogenous sentiment.

## Step 7: VAR Analysis

### Folder: `06-EER-replication_of_VAR_results`
The folder reproduces all results from the VAR Analyis.


