#!/usr/bin/python
# -*- coding: utf-8 -*-

from nltk.tokenize import RegexpTokenizer
from datetime import datetime
import numpy as np
import stop_words
import json
import os
import tempfile

from db_layer import db_layer
import utils


class feature_extractor(object):
    def __init__(self, config_file="conf/config.json"):
        self.config_file = config_file
        self.config = utils.get_configuration(config_file)
        self.db = db_layer(config_file)

    def train(self, urls):
        pass

    def compute(self, author):

        if not type(author) == dict:
            author = self.db.get_author(author)

        author = self.compute_features(author)
        self.db.update_author(author)

        return author

    def compute_features(author):
        return author


class concat_fe(feature_extractor):
    def __init__(self, config_file="conf/config.json", children=[]):
        self.children = list(children)
        super(concat_fe, self).__init__(config_file)

    def compute_features(self, author):
        for ch in self.children:
            author = ch.compute_features(author)

        return author

    def train(self, urls):
        for ch in self.children:
            ch.train(urls)


class clear_fe(feature_extractor):
    def compute_features(self, author):
        return self.db.clear_features(author, commit=False)

class num_tokens_fe(feature_extractor):
    def __init__(self, config_file="conf/config.json"):
        super(num_tokens_fe, self).__init__(config_file)
        self.tokenizer = RegexpTokenizer(r'\w+')

    def compute_features(self, author):
        documents = [self.tokenizer.tokenize(d) for d in author["corpus"]]
        unique_tokens = [list(set(t)) for t in documents]
        
        # Number of tokens per document
        ntokens = map(len, documents)
        author = self.db.set_feature(author, "tokens_avg", np.mean(ntokens))
        author = self.db.set_feature(author, "tokens_min", np.min(ntokens))
        author = self.db.set_feature(author, "tokens_max", np.max(ntokens))

        # Number of unique tokens per document (binary occurence)
        n_unique_tokens = [float(len(u)) / t \
                            for u, t in zip(unique_tokens, ntokens)]
        author = self.db.set_feature(author, "unique_tokens_avg",
                                     np.mean(n_unique_tokens))
        author = self.db.set_feature(author, "unique_tokens_min",
                                     np.min(n_unique_tokens))
        author = self.db.set_feature(author, "unique_tokens_max",
                                     np.max(n_unique_tokens))

        return author

class stop_words_fe(feature_extractor):
    def __init__(self, config_file="conf/config.json"):
        def get_stop_words(lang):
            try:
                mapped_lang = self.config["languages"][lang]
                return stop_words.get_stop_words(mapped_lang)
            except:
                return []
            
        super(stop_words_fe, self).__init__(config_file)
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stopwords = {ln: get_stop_words(ln) \
                            for ln in self.db.get_languages()}

    def compute_features(self, author):
        lang = self.db.get_author_language(author["id"])
        stopwords = self.stopwords[lang]
        documents = [self.tokenizer.tokenize(d) for d in author["corpus"]]
        
        # Occurrences of the stop-words in the text
        ntokens = map(len, documents)
        stop_tokens = [[x for x in d if x in stopwords] for d in documents]
        n_sw = [float(len(sw)) / t for sw, t in zip(stop_tokens, ntokens)]
        author = self.db.set_feature(author, "stop_words_avg",
                                     np.mean(n_sw))
        author = self.db.set_feature(author, "stop_words_min",
                                     np.min(n_sw))
        author = self.db.set_feature(author, "stop_words_max",
                                     np.max(n_sw))

        # Binary (unique) occurrences of the stop-words in the text
        unique_tks = [list(set(x)) for x in documents]
        n_unique_tokens = map(len, unique_tks)
        sw_unique = [[x for x in d if x in stopwords] for d in unique_tks]
        n_sw_unique = [float(len(sw)) / t for sw, t in zip(sw_unique,
                                                           n_unique_tokens)]
        author = self.db.set_feature(author, "unique_stop_words_avg",
                                     np.mean(n_sw_unique))
        author = self.db.set_feature(author, "unique_stop_words_min",
                                     np.min(n_sw_unique))
        author = self.db.set_feature(author, "unique_stop_words_max",
                                     np.max(n_sw_unique))
        
        #TODO: include a BoW encoding the occurrences of each stop-word

        return author

class punctuation_fe(feature_extractor):
    def compute_features(self, author):
        def avg_min_max_char(documents, char):
            l = [len(filter(lambda x: x == char, d)) for d in documents]
            if len(l) == 0:
                l = [0]

            return np.mean(l), np.min(l), np.max(l)

        def set_avg_min_max(author, punctuation, name, char):
            avg_char, min_char, max_char = avg_min_max_char(punctuation, char)
            author = self.db.set_feature(author, "puntuation_" + name + "_avg",
                                         avg_char)
            author = self.db.set_feature(author, "puntuation_" + name + "_min",
                                         min_char)
            author = self.db.set_feature(author, "puntuation_" + name + "_max",
                                         max_char)
            return author

        documents = author["corpus"]
        punctuation = [filter(lambda x: not x.isalnum() and not x.isspace(),
                              d) for d in documents]

        len_punctuation = [len(x) for x in punctuation]

        author = self.db.set_feature(author, "puntuation_avg",
                                     np.mean(len_punctuation))
        author = self.db.set_feature(author, "puntuation_min",
                                     np.min(len_punctuation))
        author = self.db.set_feature(author, "puntuation_max",
                                     np.max(len_punctuation))
        
        author = set_avg_min_max(author, punctuation, "points", '.')
        author = set_avg_min_max(author, punctuation, "commas", ',')
        author = set_avg_min_max(author, punctuation, "semi_colon", ';')
        author = set_avg_min_max(author, punctuation, "question", '?')
        author = set_avg_min_max(author, punctuation, "exclamation", '!')
        author = set_avg_min_max(author, punctuation, "double_quote", '"')
        author = set_avg_min_max(author, punctuation, "single_quote", '\'')

        return author
