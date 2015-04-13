from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GMM, DPGMM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from scipy.stats import multivariate_normal

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

    def distance(self, weights, a, b):
        return math.sqrt(sum([w * (x - y) * (x - y) \
                                for (w, x, y) in zip(weights, a, b)]))

    def train(self, authors_id):
        authors = [self.db.get_author(a, True) for a in authors_id]
        samples = self.get_matrix(authors)
        
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
        print best_threshold, self.threshold, \
              best_accuracy * 100.0 / len(values)

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

        self.weights[author] = [abs(d - mu) / (2.0 * s + 1e-7) + 1.0 \
                                    for (d, s) in zip(bounded_d, self.std)]
        total_w = sum(self.weights[author])
        self.weights[author] = [x / total_w for x in self.weights[author]]
        
        
        



class ubm(classifier):
    def __init__(self, config, language, fe):
        self.config_file = config
        self.config = utils.get_configuration(config)
        self.db = db_layer(config)
        self.language = language
        self.feature_list = []
        self.fe = fe
        self.weights = {}
        self.threshold = 0.0
        

    def train(self, authors_id):
        def gen_hex_colour_code():
           return '#' + ''.join([random.choice('0123456789ABCDEF') \
                                    for x in range(6)])
           
        def make_ellipses(gmm, ax, colors):
            for n in range(0, len(gmm.weights_)):
                color = colors[n]
                v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
                u = w[0] / np.linalg.norm(w[0])
                angle = np.arctan2(u[1], u[0])
                angle = 180 * angle / np.pi  # convert to degrees
                v *= 9
                ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                          180 + angle, color=color)
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(0.5)
                ax.add_artist(ell)
                
        def mvnpdf(mean, covar, samples):
            multivariate_normal.pdf(samples, mean=mean, cov=covar)
            return multivariate_normal.pdf(samples, mean=mean, cov=covar) #allow_singular=True
    
        def alfa(n, r):
            return n/(n+r)
        # x is a vector of samples
        def em(weights, means, covars,  samples, r):
            pr = np.array(
                [
                    weights[i]*mvnpdf(means[i],covars[i],samples) + 0.1**7 \
                        for i in range(0,len(means))
                ]
            ).T
            
            if len(samples) == 1:
                pr = np.array([pr])
                
            # print map(sum, pr)
            pr = np.array([p/s \
                            for (p, s) in zip(pr,map(sum, pr))]).T
            # print map(sum,pr.T)
            ns = map(sum, pr)
            
            # print pr
            # print samples
            new_means = [sum([p*s for p,s in zip(ps,samples)])/ns[i] \
                                for i, ps in enumerate(pr)]
            new_covars = [sum([p*(s**2) for p,s in zip(ps,samples)])/ns[i] \
                                for i, ps in enumerate(pr)]
            alfas = [alfa(n, r) for n in ns]
            # Bayesian adaptation
            t = len(samples)
            adapted_weights = [a*n/t + (1-a)*w \
                            for a, n, w in zip(alfas, ns, weights)]
            adapted_weights = adapted_weights/sum(adapted_weights)
            
            adapted_means = [a*nm + (1-a)*m \
                            for a, nm, m in zip(alfas, new_means, means)]
            adapted_means = np.array(adapted_means)
            
            adapted_covars = [a*nc + ((1-a)*(c+m**2) - am**2) for a, nc, c, am, m \
                                in zip(alfas, new_covars, covars, adapted_means, means)]
            adapted_covars = np.array(adapted_covars)
            
            # print
            # print 'weights'
            # print weights
            # print adapted_weights
            # print sum(adapted_weights)
            #
            # print
            #
            # print 'means'
            # print means
            # print adapted_means
            #
            # print
            # print 'covars'
            # print covars
            # print adapted_covars
            return adapted_weights, adapted_means, adapted_covars

                    
        authors = [self.db.get_author(a, True) for a in authors_id]
        samples = self.get_matrix(authors)
        
        self.scaler = MinMaxScaler()
        self.scaler.fit(samples)

        self.pca = None
        self.pca = PCA(n_components=40)
        self.pca.fit(samples)

        if self.scaler:
            samples = self.scaler.transform(samples)
        if self.pca:
            samples = self.pca.transform(samples)
            
        # self.scaler = MinMaxScaler()
        # samples = self.scaler.fit_transform(samples)

        self.mean = np.mean(samples, axis=0)
        self.std = np.std(samples, axis=0)
    
        # print samples.shape
        distances = []

        gt = self.db.get_ground_truth(self.language)
        
        components=100
        n_classes = len(np.unique(gt.values()))    
        classifiers = dict((covar_type, GMM(n_components=components,
                            covariance_type=covar_type))
                           for covar_type in ['diag']) 
                           #'spherical', 'diag', 'tied', 'full'])
                           
        n_classifiers = len(classifiers)

        for index, (name, classifier) in enumerate(classifiers.items()):
            fig = plt.figure()
            
            classifier.fit(samples)

            # colors = 'rgbcmyk'# [gen_hex_colour_code() for i in  range(0,components)]
            
            # h = fig.add_subplot(111, aspect='equal')
            # make_ellipses(classifier, h, colors)
            # samples_pred = classifier.predict(samples)
            # l = zip(samples_pred, samples)
            # for p in np.unique(samples_pred):
            #     s = [i[1] for i in l if i[0] == p]
            #     plt.scatter([i[0] for i in s], \
            #                 [i[1] for i in s], 1.5, color=colors[p])
            break
            
            plt.xticks(())
            plt.yticks(())  
            plt.show()
        
        tp = 'diag'
        # print classifiers
        c = classifiers[tp]

        gms = []
        for i in samples:
            # print 'Sample'
            # print i
            # print 'Adaptation'
            agm = GMM(n_components=components,covariance_type=tp)
            agm.weights_, agm.means_, agm.covars_ = \
                            em(c.weights_, c.means_, c.covars_,  [i], 16)
            gms.append(agm)
            a, b = c.score(i), agm.score(i)
            # print a<b, a,' < ', b
        
        values = []              
        for id_, (author, descriptor, gm) in enumerate(zip(authors_id, samples, gms)):

            unknown = self.db.get_unknown_document(author)
            unknown_descriptor = self.get_matrix([authors[id_]], False)

            if self.scaler:
                unknown_descriptor = self.scaler.transform(unknown_descriptor)
            if self.pca:
                unknown_descriptor = self.pca.transform(unknown_descriptor)
            ud = unknown_descriptor[0]

            target = gt[author]
            # print target
            # print ud
            values.append((agm.score(ud)/c.score(ud),target))

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
        print best_threshold, self.threshold, \
              best_accuracy * 100.0 / len(values)

    def predict(self, author_id):
        print author_id
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

        self.weights[author] = [abs(d - mu) / (2.0 * s + 1e-7) + 1.0 \
                                    for (d, s) in zip(bounded_d, self.std)]
        total_w = sum(self.weights[author])
        self.weights[author] = [x / total_w for x in self.weights[author]]
