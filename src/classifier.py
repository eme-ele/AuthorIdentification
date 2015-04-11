import numpy as np


class classifier:
    def __init__(self, config):
        self.config_file = config_file
        self.config = utils.get_configuration(config_file)
        self.db = db_layer(config_file)

    def train(self, authors):
        pass

    def predict(self, author, document):
        return 0.5

    def auc(self, authors):
        return 0.0  # TODO
    
    def c_at_one(self, authors):
        return 0.0  # TODO


class weighted_distance_classifier(classifier):
    def __init__(self, config):
        self.config_file = config_file
        self.config = utils.get_configuration(config_file)
        self.db = db_layer(config_file)

    def train(self, authors):
        pass

    def predict(self, author, document):
        return 0.5
