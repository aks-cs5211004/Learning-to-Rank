import argparse
import os
import re
import string
import xml.etree.ElementTree as ET
from collections import Counter
from math import log10
from typing import Dict, List

import nltk
import numpy as np
from gensim.models import KeyedVectors  # Change from Word2Vec to KeyedVectors
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
from scipy.spatial.distance import cosine, pdist, squareform

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from constants_gen import (DELIMITERS, DIGITS, DIRICHILET_MU, LOWERCASE,
                           PUNCTUATIONS, STEMMING, STOPWORDS_ELIMINATION,
                           TOP_K, UNK_PERCENTAGE, W2V_LAMBDA)
from lm import (CombinedLanguageModel, DocLanguageModel, KL_divergence,
                SimpleTokenizer, preprocess_text)


def KL_divergence_query(document_model: DocLanguageModel, query_model: Dict[str, float]) -> float:
    accumulator = 0
    for term in query_model:
        if query_model[term] > 0:
            accumulator += query_model[term] * log10(query_model[term] / document_model.get_term_prob(term))
    return accumulator


# Function to load GloVe embeddings from a file
def load_glove_embeddings(file_path):
    glove_model = {}
    with open(file_path, 'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            glove_model[word] = embedding
    return glove_model


# Main script that handles GloVe embeddings
def main():
    # --------- FILE HANDLING ------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='LM with GloVe Embeddings')
    parser.add_argument('--query_file', type=str, help='Query file')
    parser.add_argument('--top_100_file', type=str, help='Top 100 file')
    parser.add_argument('--collection_file', type=str, help='Collection file')
    parser.add_argument('--glove_embeddings_file', type=str, help='GloVe embeddings file')
    parser.add_argument('--output_file', type=str, help='Output directory for vectors')
    parser.add_argument('--expansions_file', type=str, help='Output file for query expansions')
    args = parser.parse_args()

    query_file_path = args.query_file
    top_100_file_path = args.top_100_file
    collection_file_path = args.collection_file
    output_file_path = args.output_file
    expansions_file_path = args.expansions_file
    glove_file_path = args.glove_embeddings_file

    query_file = open(query_file_path, 'r')
    top_100_file = open(top_100_file_path, 'r')
    collection_file = open(collection_file_path, 'r')
    output_file = open(output_file_path, 'w')
    expansions_file = open(expansions_file_path, 'w')

    print('GloVe embeddings file path is:', glove_file_path)

    # ----- Parse queries, top 100 results, and doc contents from the files and set of doc ids ---------------
    queries = {}
    top_100_results = {}
    doc_ids = set()
    doc_contents = {}
    for line in query_file:
        if line.startswith('[query_id]'):
            continue
        qid, query = line.strip().split('\t')
        queries[qid] = query
    for line in top_100_file:
        parts = line.strip().split('\t')
        if line.startswith('[query_id]'):
            continue
        query_number, doc_id, score = parts
        if query_number not in top_100_results:
            top_100_results[query_number] = []
        top_100_results[query_number].append([doc_id, score])
    for query in top_100_results:
        for doc_id, _ in top_100_results[query]:
            doc_ids.add(doc_id)
    ct = 0
    for line in collection_file:
        if len(line.strip().split('\t')) != 4:
            continue
        doc_id, url, title, body = line.strip().split('\t')
        if doc_id in doc_ids:
            # print(f"Processing collection file count: {ct}")
            ct += 1
            doc_contents[doc_id] = (title, body)

    # ---------- Load GloVe Embeddings -------------------------------------------------------------------
    glove_model = load_glove_embeddings(glove_file_path)

    for query_number, query_text in queries.items():
        print(f"Processing query {query_number}")
        # ----------------- Get QUERY TOKENS -------------------------------------------------
        query_text = preprocess_text(query_text, LOWERCASE, PUNCTUATIONS, DIGITS, STEMMING, STOPWORDS_ELIMINATION)
        query_tokens = SimpleTokenizer().tokenize(query_text)
        expansions_file.write(f"{query_number}: ")

        # ----------------- Get TOP 100 DOCS TOKENS ------------------------------------------
        doc_ids_query = {doc_id for doc_id, _ in top_100_results[query_number]}
        combined_text = ""
        for doc_id in doc_ids_query:
            combined_text += f"{doc_contents[doc_id][0]} {doc_contents[doc_id][1]} "
        combined_text = preprocess_text(combined_text, LOWERCASE, PUNCTUATIONS, DIGITS, STEMMING, STOPWORDS_ELIMINATION)

        # # 1. Get all words from the model
        # glove_words = list(glove_model.keys())
        # vocab_size = len(glove_words)
        # # 2. Create a binary vector for query terms
        # query_vector = np.zeros(vocab_size)
        # query_counter = Counter(query_tokens)
        # for i, word in enumerate(glove_words):
        #     query_vector[i] = query_counter[word]
        # # 3. Calculate the similarity matrix using pairwise cosine distance
        # word_vectors = np.array([glove_model[word] for word in glove_words])  # Convert to array for efficient indexing
        # similarity_matrix = 1 - squareform(pdist(word_vectors, metric='cosine'))
        # # 4. Calculate query similarity with all terms
        # query_similarity = np.dot(similarity_matrix, query_vector)
        # # 5. Flatten the similarity scores to get a 1D array
        # query_similarity = query_similarity.flatten()
        
        # -----------------Get CORE-------------------------------------------------- 
        vocab_size = len(glove_model)
        glove_words = list(glove_model.keys())
        embeddings = np.array([glove_model[word] for word in glove_words])
        query_vector = np.zeros(vocab_size)
        print(query_tokens)
        query_counter = Counter(query_tokens)
        for i, word in enumerate(glove_words):
            query_vector[i] = query_counter[word]
        query_similarity = np.dot(embeddings, np.dot(embeddings.T, query_vector))
        query_similarity = query_similarity.flatten()
                
        # ----------------- Get EXPANSIONS ---------------------------------------------------
        sorted_indices = np.argsort(query_similarity)[::-1]
        top_indices = [(idx, query_similarity[idx]) for idx in sorted_indices[:TOP_K]]
        expanded_query = {}
        for idx, score in top_indices:
            word = glove_words[idx]
            expanded_query[word] = score
        # print(expanded_query)
        expansions_file.write(", ".join(expanded_query.keys()))
        expansions_file.write("\n")

        # ----------------- Get DOC LMS ------------------------------------------------------
        doc_lms = {}
        for doc_id in doc_ids_query:
            doc_lms[doc_id] = DocLanguageModel(doc_id, doc_contents[doc_id])
        combined_lm = CombinedLanguageModel()
        for doc_id in doc_ids_query:
            combined_lm.add_doc_lm(doc_lms[doc_id])
        combined_lm.add_unks(UNK_PERCENTAGE)
        for doc_id in doc_ids_query:
            doc_lms[doc_id].set_background_lm(combined_lm)

        # ----------------- Get RELEVANCE MODEL PROBABILITIES ---------------------------------
        original_query_counter = {}
        for x in query_tokens:
            x = x if x in combined_lm.term_freq.keys() else '<UNK>'
            if x not in original_query_counter:
                original_query_counter[x] = 0
            original_query_counter[x] += 1
        expanded_query_counter = {}
        for x, y in expanded_query.items():
            x = x if x in combined_lm.term_freq else '<UNK>'
            if x not in expanded_query_counter:
                expanded_query_counter[x] = 0
            expanded_query_counter[x] = y / sum(expanded_query.values())
        relevance_model_probabilities = {}
        for term in combined_lm.term_freq.keys():
            relevance_model_probabilities[term] = 0
            relevance_model_probabilities[term] += W2V_LAMBDA * (expanded_query_counter.get(term, 0) / sum(expanded_query_counter.values()))
            relevance_model_probabilities[term] += (1 - W2V_LAMBDA) * (original_query_counter.get(term, 0) / sum(original_query_counter.values()))

        # ----------------- Get RESULTS ------------------------------------------------------
        results = []
        for doc_id in doc_ids_query:
            kl_div_q = KL_divergence_query(doc_lms[doc_id], relevance_model_probabilities)
            results.append((doc_id, 1 - kl_div_q))

        results.sort(key=lambda x: x[1], reverse=True)
        for idx, result in enumerate(results):
            doc_id, score = result
            output_file.write(f"{query_number} Q0 {doc_id} {idx + 1} {score:.4f} runid1\n")

    query_file.close()
    top_100_file.close()
    collection_file.close()
    expansions_file.close()
    output_file.close()


if __name__ == '__main__':
    main()
