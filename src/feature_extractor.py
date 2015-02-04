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
    def compute_features(self, article):
        return self.db.clear_features(author, commit=False)

class num_tokens_fe(feature_extractor):
    def __init__(self, config_file="conf/config.json"):
        super(num_tokens_fe, self).__init__(config_file)
        self.tokenizer = RegexpTokenizer(r'\w+')

    def compute_features(self, author):
        tokens = [self.tokenizer.tokenize(d) for d in author["corpus"]]
        unique_tokens = [list(set(t)) for t in tokens]
        
        ntokens = [len(t) for t in tokens]
        author = self.db.set_feature(author, "avg_n_tokens", np.mean(ntokens))
        author = self.db.set_feature(author, "min_n_tokens", np.min(ntokens))
        author = self.db.set_feature(author, "max_n_tokens", np.max(ntokens))

        n_unique_tokens = [float(len(u)) / t \
                            for u, t in zip(unique_tokens, ntokens)]
        author = self.db.set_feature(author, "avg_rate_unique_tokens",
                                     np.mean(n_unique_tokens))
        author = self.db.set_feature(author, "min_rate_unique_tokens",
                                     np.min(n_unique_tokens))
        author = self.db.set_feature(author, "max_rate_unique_tokens",
                                     np.max(n_unique_tokens))

        return author

class stop_words_fe(feature_extractor):
    def compute_features(self, author):
        language = self.db.get_author_language(author["id"])
        print stop_words.get_stop_words(self.config["languages"][language])
        return author