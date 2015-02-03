#!/usr/bin/python
# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np

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
