import argparse
import csv
import json
import os
import re
import string
import xml.etree.ElementTree as ET
from math import log10
from typing import Dict, List

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import RegexpTokenizer

from constants import (DELIMITERS, DIGITS, DIRICHILET_MU, LOWERCASE,
                       PUNCTUATIONS, STEMMING, STOPWORDS_ELIMINATION, TOP_K,
                       UNK_PERCENTAGE, W2V_LAMBDA)


def preprocess_text(text: str, lowercase: bool, punctuations: bool, digits: bool, stemming: bool, stopword_elimination: bool)->str:
    if lowercase:
        text = text.lower()
    if punctuations:
        text = text.translate(str.maketrans('', '', string.punctuation))
    if digits:
        digits_pattern = r"\d+(\.\d+)?"
        text = re.sub(digits_pattern, "<NUM>", text)
    if stemming:
        stemmer = SnowballStemmer("english")
        text = " ".join([stemmer.stem(token) for token in text.split()])
    if stopword_elimination:
        stop_words = set(nltk.corpus.stopwords.words('english'))
        text = " ".join([token for token in text.split() if token not in stop_words])
    return text
    
class SimpleTokenizer:
    def __init__(self):
        self.delimiters = DELIMITERS
        pattern = "[" + re.escape(''.join(self.delimiters)) + "]+"
        self._tokenizer_ = RegexpTokenizer(pattern=pattern, gaps=True)
    def tokenize(self, text: str)->List[str]:
        tokens = self._tokenizer_.tokenize(text.lower())
        return tokens
    
class DocLanguageModel:
    def __init__(self, id: str, contents: tuple) -> None:
        self.id = id
        self._tokenizer_ = SimpleTokenizer()
        text_list= list(contents)
        self.text = ' '.join(text_list)
        self.text = preprocess_text(self.text, LOWERCASE, PUNCTUATIONS, DIGITS, STEMMING, STOPWORDS_ELIMINATION)
        self.tokens = self._tokenizer_.tokenize(self.text)
        self.doc_length = len(self.tokens)
        self.term_freq = {}
        for token in self.tokens:
            if token in self.term_freq:
                self.term_freq[token] += 1
            else:
                self.term_freq[token] = 1
        self.background_lm = None
        self.mu = DIRICHILET_MU
        
    def set_background_lm(self, background_lm):
        self.background_lm = background_lm
        
    def get_term_prob(self, term: str):
        term_freq = self.term_freq.get(term, 0)
        return (term_freq + self.mu * self.background_lm.get_term_prob(term)) / (self.doc_length + self.mu)
        
    
class CombinedLanguageModel:
    def __init__(self) -> None:
        self.total_count = 0
        self.term_freq = {}
    
    def add_doc_lm(self, doc_lm: DocLanguageModel):
        for term,count in doc_lm.term_freq.items():
            if term in self.term_freq:
                self.term_freq[term] += count
            else:
                self.term_freq[term] = count
        self.total_count += doc_lm.doc_length
    
    def get_term_prob(self, term: str):
        term_freq = self.term_freq.get(term, 0)
        return term_freq / self.total_count
    
    def add_unks(self, unk_percentage: float):
        self.term_freq['<UNK>'] = int((unk_percentage/100) * sum(self.term_freq.values()))
        
        
def KL_divergence(p: DocLanguageModel, q: Dict[str, float])->float:
    p_keys = set(p.term_freq.keys())
    q_keys = set(q.keys())
    keys = p_keys.intersection(q_keys)
    p_values = np.array([p.get_term_prob(key) for key in keys])
    q_values = np.array([q[key] for key in keys])
    return np.sum(p_values * np.log(p_values / q_values))
        
# ------------------------------------------------------------------------------ CODE -------------------------------------
# -------------------------------------------------------------------------------------------------------------------------         
        
 
def main():
    parser = argparse.ArgumentParser(description='LM')
    parser.add_argument('--query_file', type=str, help='Query file')
    parser.add_argument('--top_100_file', type=str, help='Top 100 file')
    parser.add_argument('--collection_file', type=str, help='Collection file')
    parser.add_argument('--output_file', type=str, help='Output file')
    args = parser.parse_args()
    query_file_path = args.query_file
    top_100_file_path = args.top_100_file
    collection_file_path= args.collection_file
    output_file_path = args.output_file

    query_file = open(query_file_path, 'r')
    top_100_file = open(top_100_file_path, 'r')
    collection_file = open(collection_file_path, 'r')
    output_file = open(output_file_path, 'w')
    
    print("Processing files...")
    print(f"Query file: {query_file_path}")
    print(f"Top 100 file: {top_100_file_path}")
    print(f"Collection file: {collection_file_path}")
    print(f"Output file: {output_file_path}")
    
    # parse queries
    queries = {}
    for line in query_file:
        if line.startswith('[query_id]'):
            continue
        qid, query = line.strip().split('\t')
        queries[qid] = query
        
    # parse top 100 results file query_id doc_id score
    top_100_results = {}
    print("Processing top 100 results...")
    for line in top_100_file:
        parts = line.strip().split('\t')
        if line.startswith('[query_id]'):
            continue
        query_number, doc_id, score = parts
        if query_number not in top_100_results:
            top_100_results[query_number] = list()
        top_100_results[query_number].append([doc_id, score])
        
    # get the set of all doc_ids
    doc_ids = set()
    for query in top_100_results:
        for doc_id, _ in top_100_results[query]:
            doc_ids.add(doc_id)
            
    print("Processing collection file...")
    print(len(doc_ids))
    
    # now go over the whoel collections and get the contents of the matching doc_ids
    # make it faster by storing the contents in a dictionar
    doc_contents = {}
    ct=0
    for line in collection_file:
        if len(line.strip().split('\t')) != 4:
            continue
        doc_id, url, title, body = line.strip().split('\t')
        if doc_id in doc_ids:
            print(f"Processed {ct} docs\r")
            ct+=1
            doc_contents[doc_id] = (url, title, body)
                
    for query in queries:
        query_text = queries[query]
        query_text = preprocess_text(query_text, LOWERCASE, PUNCTUATIONS, DIGITS, STEMMING, STOPWORDS_ELIMINATION)
        query_tokens = SimpleTokenizer().tokenize(query_text)
        print(f"Processing Query Id: {query}")
        
        doc_ids_query = set()
        for doc_id, _ in top_100_results[query]:
            doc_ids_query.add(doc_id)
        
        doc_lms = {}
        for doc_id in doc_ids_query:
            doc_lms[doc_id] = DocLanguageModel(doc_id, doc_contents[doc_id])
            
        print("Len of doc_lms", len(doc_lms))
        
        background_lm = CombinedLanguageModel()
        for doc_id in doc_ids_query:
            background_lm.add_doc_lm(doc_lms[doc_id])
        background_lm.add_unks(UNK_PERCENTAGE)
        
        for doc_id in doc_ids_query:
            doc_lms[doc_id].set_background_lm(background_lm)
        
        query_probability_per_language_model = dict()
        for doc_id in doc_ids_query:
            query_prob = 1
            for query_term in query_tokens:
                query_prob *= doc_lms[doc_id].get_term_prob(query_term)
            query_probability_per_language_model[doc_id] = query_prob
        query_prob = sum(query_probability_per_language_model.values()) / len(doc_lms)
        
        
        # get relevance_model_probabilities
        relevance_model_probabilities = {}
        for term in background_lm.term_freq:
            relevance_model_probabilities[term] = 0
            for doc_id in doc_ids_query:
                p = doc_lms[doc_id].get_term_prob(term)
                q = query_probability_per_language_model[doc_id]
                relevance_model_probabilities[term] += p * q
            relevance_model_probabilities[term] /= len(doc_ids_query)
            relevance_model_probabilities[term] /= query_prob
        
        results = []
        for doc_id in doc_ids_query:
            kl_div_q = KL_divergence(doc_lms[doc_id], relevance_model_probabilities)
            results.append((doc_id, 1 - kl_div_q))
            
        results.sort(key=lambda x: x[1], reverse=True)
        for idx, result in enumerate(results):
            doc_id, score = result
            output_file.write(f"{query_number} Q0 {doc_id} {idx + 1} {score:.4f} runid1\n")

                
    query_file.close()
    top_100_file.close()
    collection_file.close()
    output_file.close()
    
    
if __name__ == "__main__":
    main()