#!/usr/bin/python
# -*- coding: utf-8 -*-

from nltk.tokenize import RegexpTokenizer
from textblob import TextBlob
from gensim import corpora, models
from datetime import datetime
import numpy as np
import stop_words
import json
import os
import tempfile
import re
from collections import Counter

from db_layer import db_layer
import utils


class feature_extractor(object):
    def __init__(self, config_file="conf/config.json"):
        self.config_file = config_file
        self.config = utils.get_configuration(config_file)
        self.db = db_layer(config_file)
        self.paragraph_re = r'.+\n'
        self.language = None

    def get_paragraphs(self, documents):
        ret = [list(re.findall(self.paragraph_re, d + '\n')) \
                   for d in documents]
        ret = utils.flatten(ret)
        ret = [p for p in ret if len(p.split()) > 0]
        return ret

    def get_sentences(self, documents):
        ret = utils.flatten([TextBlob(d).sentences for d in documents])
        return [t.string for t in ret]

    def train(self, authors):
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

    def train(self, authors):
        for ch in self.children:
            ch.train(authors)


class clear_fe(feature_extractor):
    def compute_features(self, author):
        return self.db.clear_features(author, commit=False)


class num_tokens_fe(feature_extractor):
    def __init__(self, config_file="conf/config.json"):
        super(num_tokens_fe, self).__init__(config_file)
        self.tokenizer = RegexpTokenizer(r'\w+')

    def get_features(self, author, corpus, prefix):
        corpus = [self.tokenizer.tokenize(d) for d in corpus]
        unique_tokens = [list(set(t)) for t in corpus]

        # Number of tokens per document
        ntokens = map(len, corpus)
        author = self.db.set_feature(author, prefix + "tokens_avg",
                                     np.mean(ntokens))
        author = self.db.set_feature(author, prefix + "tokens_min",
                                     np.min(ntokens))
        author = self.db.set_feature(author, prefix + "tokens_max",
                                     np.max(ntokens))

        # Number of unique tokens per document (binary occurence)
        n_unique_tokens = [float(len(u)) / max(1, t) \
                            for u, t in zip(unique_tokens, ntokens)]
        author = self.db.set_feature(author, prefix + "unique_tokens_avg",
                                     np.mean(n_unique_tokens))
        author = self.db.set_feature(author, prefix + "unique_tokens_min",
                                     np.min(n_unique_tokens))
        author = self.db.set_feature(author, prefix + "unique_tokens_max",
                                     np.max(n_unique_tokens))
        return author

    def compute_features(self, author):
        author = self.get_features(author, author["corpus"],
                                   "document::")
        author = self.get_features(author,
                                   self.get_paragraphs(author["corpus"]),
                                   "paragraph::")
        author = self.get_features(author,
                                   self.get_sentences(author["corpus"]),
                                   "sentence::")
        return author


class structure_fe(feature_extractor):
    def compute_features(self, author):
        documents = [TextBlob(d) for d in author["corpus"]]
        d_sentences = [d.sentences for d in documents]
        d_nsentences = [len(d) for d in d_sentences]

        paragraphs = [TextBlob(d) \
                        for d in self.get_paragraphs(author["corpus"])]
        p_sentences = [d.sentences for d in paragraphs]
        p_nsentences = [len(d) for d in p_sentences]

        author = self.db.set_feature(author, "document::sentences_min",
                                     np.min(d_nsentences))
        author = self.db.set_feature(author, "document::sentences_max",
                                     np.max(d_nsentences))
        author = self.db.set_feature(author, "document::sentences_avg",
                                     np.mean(d_nsentences))

        author = self.db.set_feature(author, "paragraph::sentences_min",
                                     np.min(p_nsentences))
        author = self.db.set_feature(author, "paragraph::sentences_max",
                                     np.max(p_nsentences))
        author = self.db.set_feature(author, "paragraph::sentences_avg",
                                     np.mean(p_nsentences))

        paragraphs_per_document = [len(self.get_paragraphs([d]))\
                                    for d in author["corpus"]]
        author = self.db.set_feature(author, "document::paragraph_min",
                                     np.min(paragraphs_per_document))
        author = self.db.set_feature(author, "document::paragraph_max",
                                     np.max(paragraphs_per_document))
        author = self.db.set_feature(author, "document::paragraph_avg",
                                     np.mean(paragraphs_per_document))

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

    def get_features(self, author, corpus, prefix):
        corpus = [self.tokenizer.tokenize(d) for d in corpus]
        unique_tokens = [list(set(t)) for t in corpus]

        lang = self.db.get_author_language(author["id"])
        stopwords = self.stopwords[lang]
        documents = list(corpus)

        # Occurrences of the stop-words in the text
        ntokens = map(len, documents)
        stop_tokens = [[x for x in d if x in stopwords] for d in documents]
        n_sw = [float(len(sw)) / max(1, t)\
                  for sw, t in zip(stop_tokens, ntokens)]
        author = self.db.set_feature(author, prefix + "stop_words_avg",
                                     np.mean(n_sw))
        author = self.db.set_feature(author, prefix + "stop_words_min",
                                     np.min(n_sw))
        author = self.db.set_feature(author, prefix + "stop_words_max",
                                     np.max(n_sw))

        # Binary (unique) occurrences of the stop-words in the text
        unique_tks = [list(set(x)) for x in documents]
        n_unique_tokens = map(len, unique_tks)
        sw_unique = [[x for x in d if x in stopwords] for d in unique_tks]
        n_sw_unique = [float(len(sw)) / max(1, t)\
                        for sw, t in zip(sw_unique, n_unique_tokens)]
        author = self.db.set_feature(author, prefix + "unique_stop_words_avg",
                                     np.mean(n_sw_unique))
        author = self.db.set_feature(author, prefix + "unique_stop_words_min",
                                     np.min(n_sw_unique))
        author = self.db.set_feature(author, prefix + "unique_stop_words_max",
                                     np.max(n_sw_unique))

        ##TODO: include a BoW encoding the occurrences of each stop-word

        return author

    def compute_features(self, author):
        author = self.get_features(author, author["corpus"],
                                   "document::")
        author = self.get_features(author,
                                   self.get_paragraphs(author["corpus"]),
                                   "paragraph::")
        author = self.get_features(author,
                                   self.get_sentences(author["corpus"]),
                                   "sentence::")
        return author

class punctuation_fe(feature_extractor):
    def avg_min_max_char(self, documents, char):
        l = [d.count(char) for d in documents]
        if len(l) == 0:
            l = [0]

        return np.mean(l), np.min(l), np.max(l)

    def set_avg_min_max(self, author, punctuation, name, char):
        avg_char, min_char, max_char = self.avg_min_max_char(punctuation, char)
        author = self.db.set_feature(author,
                                     "punctuation_" + name + "_avg",
                                     avg_char)
        author = self.db.set_feature(author,
                                     "punctuation_" + name + "_min",
                                     min_char)
        author = self.db.set_feature(author,
                                     "punctuation_" + name + "_max",
                                     max_char)
        return author

    def get_features(self, author, corpus, prefix):
        punctuation = [filter(lambda x: not x.isalnum() and \
                                        not x.isspace(),
                              d) for d in corpus]
        len_punctuation = [len(x) for x in punctuation]

        author = self.db.set_feature(author, prefix + "punctuation_avg",
                                     np.mean(len_punctuation))
        author = self.db.set_feature(author, prefix + "punctuation_min",
                                     np.min(len_punctuation))
        author = self.db.set_feature(author, prefix + "punctuation_max",
                                     np.max(len_punctuation))

        author = self.set_avg_min_max(author, punctuation,
                                      prefix + "points", '.')
        author = self.set_avg_min_max(author, punctuation,
                                      prefix + "commas", ',')
        author = self.set_avg_min_max(author, punctuation,
                                      prefix + "semi_colon", ';')
        author = self.set_avg_min_max(author, punctuation,
                                      prefix + "question", '?')
        author = self.set_avg_min_max(author, punctuation,
                                      prefix + "open_question", u'¿')
        author = self.set_avg_min_max(author, punctuation,
                                      prefix + "exclamation", '!')
        author = self.set_avg_min_max(author, punctuation,
                                      prefix + "open_exclamation", u'¡')
        author = self.set_avg_min_max(author, punctuation,
                                      prefix + "double_quote", '"')
        author = self.set_avg_min_max(author, punctuation,
                                      prefix + "single_quote", '\'')

        return author

    def compute_features(self, author):
        author = self.get_features(author,
                                   author["corpus"],
                                   "document::")
        author = self.get_features(author,
                                   self.get_paragraphs(author["corpus"]),
                                   "paragraph::")
        author = self.get_features(author,
                                   self.get_sentences(author["corpus"]),
                                   "sentence::")
        return author





class char_distribution_fe(feature_extractor):
    def train(self, authors):
        self.chars = [self.db.get_author(a)["corpus"] for a in authors]
        self.chars = utils.flatten(utils.flatten(self.chars))
        self.chars = filter(lambda x: x.isalnum(), self.chars)
        self.chars = list(set([x.lower() for x in self.chars]))
        self.chars.sort()

    def compute_features(self, author):
        def get_distribution(document):
            distribution = filter(lambda x: x.isalnum(), document)
            distribution = [ch.lower() for ch in distribution]
            document_length = len(distribution)
            distribution = Counter(distribution)

            ret = []
            for ch in self.chars:
                ret.append(float(distribution[ch]) / document_length)

            return ret

        # Bag-of-Words
        author_chars = [get_distribution(d) for d in author["corpus"]]
        author_chars = np.divide(np.sum(author_chars, axis=0),
                                 len(author_chars))

        for id_ch, (char, value) in enumerate(zip(self.chars, author_chars)):
            author = self.db.set_feature(author,
                                         "BoW::abc::" + char,
                                         value)

        # Digits, uppercase, lowercase
        doc_length = [len(d) for d in author["corpus"]]

        digits = [filter(lambda x: x.isdigit(), d) for d in author["corpus"]]
        digits = [float(len(x)) / y for (x, y) in zip(digits, doc_length)]
        author = self.db.set_feature(author, "num_digits_avg", np.mean(digits))
        author = self.db.set_feature(author, "num_digits_min", np.min(digits))
        author = self.db.set_feature(author, "num_digits_max", np.max(digits))

        upper = [filter(lambda x: x.isupper(), d) for d in author["corpus"]]
        upper = [float(len(x)) / y for (x, y) in zip(upper, doc_length)]
        author = self.db.set_feature(author, "uppercase_avg", np.mean(upper))
        author = self.db.set_feature(author, "uppercase_min", np.min(upper))
        author = self.db.set_feature(author, "uppercase_max", np.max(upper))

        lower = [filter(lambda x: x.islower(), d) for d in author["corpus"]]
        lower = [float(len(x)) / y for (x, y) in zip(lower, doc_length)]
        author = self.db.set_feature(author, "lower_case_avg", np.mean(lower))
        author = self.db.set_feature(author, "lower_case_min", np.min(lower))
        author = self.db.set_feature(author, "lower_case_max", np.max(lower))

        return author


class word_distribution_fe(feature_extractor):
    def __init__(self, config_file="conf/config.json"):
        def get_stop_words(lang):
            try:
                mapped_lang = self.config["languages"][lang]
                return stop_words.get_stop_words(mapped_lang)
            except:
                return []

        super(word_distribution_fe, self).__init__(config_file)
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stopwords = {ln: get_stop_words(ln) \
                                for ln in self.db.get_languages()}

    def train(self, authors):
        languages = list(set([self.db.get_author_language(a) for a in authors]))
        stopword_list = [x[lang] for lang in languages]
        stopword_list = utils.flatten(stopword_list)
        self.words = [self.db.get_author(a)["corpus"] for a in authors]
        self.words = utils.flatten(self.words)
        self.words = map(lambda x: self.tokenizer.tokenize(x), self.words)
        self.words = utils.flatten(self.words)
        self.words =  list(set([x.lower() for x in self.words]))
        self.words = filter(lambda x: x not in stopword_list, self.words)
        self.words.sort()

    def compute_features(self, author):
        def get_distribution(document):
            distribution = self.tokenizer.tokenize(document)
            distribution = [x.lower for x in distribution]
            document_length = len(distribution)
            distribution = Counter(distribution)

            ret = []
            for w in self.words:
                ret.append(float(distribution[w]) / document_length)

            return ret

        # Bag-of-Words
        author_words = [get_distribution(d) for d in author["corpus"]]
        author_words = np.divide(np.sum(author_words, axis=0),
                                 len(author_words))

        for id_w, (word, value) in enumerate(zip(self.words, author_words)):
            author = self.db.set_feature(author,
                                         "BoW::word::" + word,
                                         value)
        return author


class stopword_distribution_fe(feature_extractor):
    def __init__(self, config_file="conf/config.json"):
        def get_stop_words(lang):
            try:
                mapped_lang = self.config["languages"][lang]
                return stop_words.get_stop_words(mapped_lang)
            except:
                return []

        super(stopword_distribution_fe, self).__init__(config_file)
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stopwords = {ln: get_stop_words(ln) \
                                for ln in self.db.get_languages()}

    def train(self, authors):
        languages = list(set([self.db.get_author_language(a) for a in authors]))
        stopword_list = [x[lang] for lang in languages]
        stopword_list = utils.flatten(stopword_list)
        self.words = [self.db.get_author(a)["corpus"] for a in authors]
        self.words = utils.flatten(self.words)
        self.words = map(lambda x: self.tokenizer.tokenize(x), self.words)
        self.words = utils.flatten(self.words)
        self.words = list(set([x.lower() for x in self.words]))
        self.words = filter(lambda x: x in stopword_list, self.words)
        self.words.sort()

    def compute_features(self, author):
        def get_distribution(document):
            distribution = self.tokenizer.tokenize(document)
            distribution = [x.lower() for x in distribution]
            document_length = len(distribution)
            distribution = Counter(distribution)

            ret = []
            for w in self.words:
                ret.append(float(distribution[w]) / document_length)

            return ret

        # Bag-of-Words
        author_words = [get_distribution(d) for d in author["corpus"]]
        author_words = np.divide(np.sum(author_words, axis=0),
                                 len(author_words))

        for id_w, (word, value) in enumerate(zip(self.words, author_words)):
            author = self.db.set_feature(author,
                                         "BoW::word::" + word,
                                         value)
        return author



class word_topics_fe(feature_extractor):
    def __init__(self, config_file="conf/config.json"):
        def get_stop_words(lang):
            try:
                mapped_lang = self.config["languages"][lang]
                return stop_words.get_stop_words(mapped_lang)
            except:
                return []
        super(word_topics_fe, self).__init__(config_file)
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stopwords = {ln: get_stop_words(ln) \
                                for ln in self.db.get_languages()}

    def train(self, authors):
        # get set of languages used in training set corpus
        languages = list(set([self.db.get_author_language(a) for a in authors]))
        # create a full stopwod list of all the languages used
        stopword_list = [x[lang] for lang in languages]
        stopword_list = utils.flatten(stopword_list)
        # transform corpus into a list of preprocessed documents
        documents = [self.db.get_author(a)["corpus"] for a in authors]
        documents = utils.flatten(documents)
        documents = map(lambda x: self.tokenizer.tokenize(x), documents)
        documents = [map(lambda x: x.lower(), d) for d in documents]
        documents = [filter(lambda x: x not in stopword_list, d) for d in documents]
        # build topic model
        self.dictionary = corpora.Dictionary(documents)
        documents = map(lambda x: self.dictionary.doc2bow(x), documents)
        self.model = models.LdaModel(documents, num_topics=10, id2word=self.dictionary, iterations=1000)

    def compute_features(self, author):
        # join all documents as single document
        document = " ".join(author["corpus"])
        # preprocess single document
        lang = db.get_author_language(author)
        document = self.tokenizer.tokenize(document)
        document = filter(lambda x: x not in self.stopwords[lang], document)
        topics = self.model[self.dictionary.doc2bow(document)]
        # topics are abstract, index number will be the identifier
        for (index, prop) in topics:
            self.db.set_feature(author,
                                "LDA::word::" + index,
                                prop)

        return author


class stopword_topics_fe(feature_extractor):
    def __init__():
        def get_stop_words(lang):
            try:
                mapped_lang = self.config["languages"][lang]
                return stop_words.get_stop_words(mapped_lang)
            except:
                return []
        super(word_topics_fe, self).__init__(config_file)
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stopwords = {ln: get_stop_words(ln) \
                                for ln in self.db.get_languages()}

    def train(self, authors):
        # get set of languages used in training set corpus
        languages = list(set([self.db.get_author_language(a) for a in authors]))
        # create a stopwod list of all the languages used
        stopword_list = [x[lang] for lang in languages]
        stopword_list = utils.flatten(stopword_list)
        # transform corpus into a list of preprocessed documents
        documents = [self.db.get_author(a)["corpus"] for a in authors]
        documents = utils.flatten(documents)
        documents = map(lambda x: self.tokenizer.tokenize(x), documents)
        documents = [map(lambda x: x.lower(), d) for d in documents]
        documents = [filter(lambda x: x in stopword_list, d) for d in documents]
        # build topic model
        self.dictionary = corpora.Dictionary(documents)
        documents = map(lambda x: self.dictionary.doc2bow(x), documents)
        self.model = models.LdaModel(documents, num_topics=10, id2word=self.dictionary, iterations=1000)


    def compute_features(self, author):
        # join all documents as single document
        document = " ".join(author["corpus"])
        # preprocess single document
        lang = db.get_author_language(author)
        document = self.tokenizer.tokenize(document)
        document = filter(lambda x: x in self.stopwords[lang], document)
        topics = self.model[self.dictionary.doc2bow(document)]
        # topics are abstract, index number will be the identifier
        for (index, prop) in topics:
            self.db.set_feature(author,
                                "LDA::word::" + index,
                                prop)

        return author


