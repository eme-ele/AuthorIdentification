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

    def predict(self, author, document):
        return 0.5

    def auc(self, authors):
        return 0.0  # TODO
    
    def c_at_one(self, authors):
        return 0.0  # TODO

    def get_matrix(self, authors):
        self.feature_list = [a["features"].keys() for a in authors]
        self.feature_list = list(set([f for fts in self.feature_list \
                                        for f in fts ]))
        self.feature_list.sort()

        samples = [[s["features"].get(f, np.nan) for f in self.feature_list] \
                    for s in authors]

        return np.asarray(samples)

class weighted_distance_classifier(classifier):
    def __init__(self, config, language):
        classifier.__init__(self, config, language)
        self.weights = {}
        self.threshold = 0.0

    def normal_p(mu, sigma, x):
        diff_x_mu = x - mu
        sigma2 = sigma * sigma

        if sigma2 == 0.0:
            if x == mu:
                return 1.0
            else:
                return 0.0

        return np.exp(- diff_x_mu * diff_x_mu / (2 * sigma2)) / \
               math.sqrt(2.0 * np.pi * sigma2)

    def train(self, authors_id):
        authors = [self.db.get_author(a, True) for a in authors_id]
        samples = self.get_matrix(authors)
        
        self.mean = np.mean(samples, axis=0)
        self.std = np.std(samples, axis=0)

        print samples.shape
        distances = []

        gt = self.db.get_ground_truth(self.language)
        
        values = []
        
        for id_, (author, descriptor) in enumerate(zip(authors_id, samples)):
            bounded_d = [min(max(mu - 2 * sigma, d), mu + 2 * sigma) \
                            for (d, mu, sigma) in zip(descriptor,
                                                      self.mean, self.std)]

            self.weights[author] = [abs(d - mu) / (2.0 * s + 1e-7) \
                                        for (d, s) in zip(bounded_d, self.std)]
            total_w = sum(self.weights[author])
            self.weights[author] = [x / total_w for x in self.weights[author]]

            unknown = self.db.get_unknown_document(author)
            
            print dummy_author
            target = gt[author]

            print target
            #for i in range(len(descriptor)):
                #print descriptor[i], self.mean[i], self.std[i]
            #print

        import os; os.sys.exit(0)

    def predict(self, author, document):
        return 0.5
