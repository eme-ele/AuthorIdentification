from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from src.db_layer import *
import utils
import copy
import math


class classifier:
    def __init__(self, config, language):
        self.config_file = config
        self.config = utils.get_configuration(config)
        self.db = db_layer(config)
        self.language = language
        self.feature_list = []
        self.fe = self.db.get_feature_extractor(language)

    def train(self, authors):
        pass

    def predict(self, author):
        return 0.5

    def accuracy(self, authors):
        ret = 0.0
        gt = self.db.get_ground_truth(self.language)

        for author in authors:
            if self.predict(author) == gt[author]:
                ret += 1.0

        return ret * 100.0 / len(authors)

    def auc(self, authors):
        return 0.0  # TODO
    
    def c_at_one(self, authors):
        return 0.0  # TODO

    def get_matrix(self, authors, known=True):
        self.feature_list = [a["features"].keys() for a in authors]
        self.feature_list = list(set([f for fts in self.feature_list \
                                        for f in fts ]))
        self.feature_list.sort()

        samples = [[s["features" if known else "unknown_features"].get(f,
                                                                       np.nan)\
                    for f in self.feature_list] for s in authors]

        return np.asarray(samples)


class weighted_distance_classifier(classifier):
    def __init__(self, config, language):
        classifier.__init__(self, config, language)
        self.weights = {}
        self.threshold = 0.0

    def normal_p(self, mu, sigma, x):
        diff_x_mu = x - mu
        sigma2 = sigma * sigma

        if sigma2 == 0.0:
            if x == mu:
                return 1.0
            else:
                return 0.0

        return np.exp(- diff_x_mu * diff_x_mu / (2 * sigma2)) / \
               math.sqrt(2.0 * np.pi * sigma2)

    def distance(self, weights, descriptor, unknown):
        return sum([w * (abs(x - d) ** 2 + 1) / (abs(x - m) ** 2 + 1) \
                            for (w, m, d, x) in zip(weights, self.mean,
                                                    descriptor, unknown)])

    def train(self, authors_id):
        authors = [self.db.get_author(a, True) for a in authors_id]
        samples = self.get_matrix(authors)
        
        self.scaler = None
        self.scaler = MinMaxScaler()
        self.scaler.fit(samples)

        self.pca = None
        self.pca = PCA(n_components=100)
        self.pca.fit(samples)

        if self.scaler:
            samples = self.scaler.transform(samples)
        if self.pca:
            samples = self.pca.transform(samples)

        self.mean = np.mean(samples, axis=0)
        self.std = np.std(samples, axis=0)

        print samples.shape
        distances = []

        gt = self.db.get_ground_truth(self.language)
        
        values = []
        
        for id_, (author, descriptor) in enumerate(zip(authors_id, samples)):
            self.train_weights(author, descriptor)
            
            unknown = self.db.get_unknown_document(author)
            unknown_descriptor = self.get_matrix([authors[id_]], False)
            
            if self.scaler:
                unknown_descriptor = self.scaler.transform(unknown_descriptor)
            if self.pca:
                unknown_descriptor = self.pca.transform(unknown_descriptor)

            unknown_descriptor = unknown_descriptor[0]

            target = gt[author]

            values.append((self.distance(self.weights[author],
                                        descriptor, unknown_descriptor),
                           target)
                         )

        values.sort()

        best_threshold = 0
        best_accuracy = len(filter(lambda (_, t): t < 0.5, values))
        next_accuracy = best_accuracy

        for i, (v, t) in enumerate(values):
            if t > 0.5:
                next_accuracy += 1
            else:
                next_accuracy -= 1

            if next_accuracy >= best_accuracy:
                best_accuracy = next_accuracy
                best_threshold = i

        self.threshold = values[best_threshold][0]

        #print best_threshold, self.threshold, \
              #best_accuracy * 100.0 / len(values)

    def predict(self, author_id):
        author = self.db.get_author(author_id, reduced=True)

        if self.weights.get(author_id) is None:
            descriptor = self.get_matrix([author], True)
            
            if self.scaler:
                descriptor = self.scaler.transform(descriptor)
            if self.pca:
                descriptor = self.pca.transform(descriptor)

            descriptor = descriptor[0]

            self.train_weights(author_id, descriptor)

        unknown_descriptor = self.get_matrix([author], False)
        
        if self.scaler:
            unknown_descriptor = self.scaler.transform(unknown_descriptor)
        if self.pca:
            unknown_descriptor = self.pca.transform(unknown_descriptor)

        unknown_descriptor = unknown_descriptor[0]

        if self.distance(self.weights[author_id],
                         descriptor, unknown_descriptor) < self.threshold:
            return 1.0
        else:
            return 0.0

        return 0.5

    def train_weights(self, author, descriptor):
        bounded_d = [min(max(mu - 2 * sigma, d), mu + 2 * sigma) \
                        for (d, mu, sigma) in zip(descriptor,
                                                  self.mean, self.std)]

        #self.weights[author] = [abs(d - m) ** 2 / (2 * s + 1e-7) + 1.0\
                                    #for (d, m, s) in zip(bounded_d,
                                                      #self.mean, self.std)]
        self.weights[author] = [2.0 - self.normal_p(m, s, d)\
                                    for (d, m, s) in zip(bounded_d,
                                                         self.mean, self.std)]
        total_w = sum(self.weights[author])
        self.weights[author] = [x / total_w for x in self.weights[author]]
