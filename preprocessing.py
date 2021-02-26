from os import stat_result
import pandas as pd
import numpy as np
import html
import spacy
from typing import List
from joblib import Parallel, delayed

nlp = spacy.load('en_core_web_trf')

def lemmatize_pipe(doc):
    return [token.lemma_ for token in doc]

def ent_pipe(doc):
    return [ent.text if ent.label_=='PRODUCT' else ' ' for ent in doc.ents ]

def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos+chunksize] for pos in range(0, total_length, chunksize))

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def lemma_chunk(texts):
    lemma_pipe = []
    for doc in nlp.pipe(texts, batch_size=20):
        lemma_pipe.append(lemmatize_pipe(doc))
    return lemma_pipe

def ent_chunk(texts):
    ent_pipe = []
    for doc in nlp.pipe(texts, batch_size=20):
        ent_pipe.append(ent_pipe(doc))
    return ent_chunk

def preprocess(texts, task, chunksize=100):
    executor = Parallel(n_jobs=-1, backend='multiprocessing', prefer='processes')
    do = delayed(task)
    tasks = (do(chunk) for chunk in chunker(texts, len(texts), chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)






# def prep(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     This function takes in train or test dataset and performs these cleaning and transformations:
#     1. Fix quotes and double quotes
#     2. Tokenisation
#     3. Entity recognition

#     Args:
#         df (pd.DataFrame): dataframe to be preprocessed

#     Returns:
#         pd.DataFrame: preprocessed dataframe 
#     """
#     df['review'] = df['review'].apply(html.unescape).str.strip('"')
#     df['review_lemma'] = df['review'].apply(_tokenize)
#     df['review_ent'] = df['review'].apply(_ne)
#     return df

# def _tokenize(text: str) -> List:
#     """
#     Tokenize and lemmatize given text

#     Args:
#         text (str): review text

#     Returns:
#         List: list of lemmatized tokens
#     """
#     doc = nlp(text)
#     return [word.lemma_ for word in doc]

# def _ne(text: str) -> List:
#     """
#     Return drug related entity given review text

#     Args:
#         text (str): review text

#     Returns:
#         List: list of entities extracted
#     """
#     doc = nlp(text)
#     return [ent.text for ent in doc.ents if ent.label_=='PRODUCT']