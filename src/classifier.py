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

#test
import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl


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
        
    def mvnpdf(self, mean, covar, samples):
        multivariate_normal.pdf(samples, mean=mean, cov=covar)
        return multivariate_normal.pdf(samples, mean=mean, cov=covar) #allow_singular=True

    def alfa(self, n, r):
        return n/(n+r)
        
    # x is a vector of samples
    def em(self, weights, means, covars,  samples, r):
        
        for i in range(0,1):
            pr = np.array(
                [
                    weights[i]*self.mvnpdf(means[i],covars[i],samples) + 0.1**7 \
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
            alfas = [self.alfa(n, r) for n in ns]
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
            
            weights = adapted_weights
            means = adapted_means
            covars = adapted_covars
        
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
        
    def train(self, authors_id):
        def plot_test():
            # Number of samples per component
            n_samples = 70

            # Generate random sample, two components
            np.random.seed(0)
            C = np.array([[0., -0.1], [1.7, .4]])
            X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
                      .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
            X = np.vstack((X, np.array([[5,4], [5.1,4.1],  [5.1,4]])))
            print X
            # Fit a mixture of Gaussians with EM using five components
            gmm = GMM(n_components=2, covariance_type='full')
            gmm.fit(X)
    
            agm = GMM(n_components=2, covariance_type='full')
            agm.weights_, agm.means_, agm.covars_ = \
                    self.em(gmm.weights_, gmm.means_, gmm.covars_,  [X[len(X)-1]], 10)
    
            # Fit a Dirichlet process mixture of Gaussians using five components
            dpgmm = DPGMM(n_components=2, covariance_type='full')
            dpgmm.fit(X)

            color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

            for i, (clf, title) in enumerate([(gmm, 'GMM'),
                                              (agm, 'Dirichlet Process GMM')]):
                splot = plt.subplot(2, 1, 1 + i)
                Y_ = clf.predict(X)
                for i, (mean, covar, color) in enumerate(zip(
                        clf.means_, clf._get_covars(), color_iter)):
                    v, w = linalg.eigh(covar)
                    u = w[0] / linalg.norm(w[0])
                    # as the DP will not use every component it has access to
                    # unless it needs it, we shouldn't plot the redundant
                    # components.
                    if not np.any(Y_ == i):
                        continue
                    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

                    # Plot an ellipse to show the Gaussian component
                    angle = np.arctan(u[1] / u[0])
                    angle = 180 * angle / np.pi  # convert to degrees
                    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
                    ell.set_clip_box(splot.bbox)
                    ell.set_alpha(0.5)
                    splot.add_artist(ell)

                plt.xlim(-10, 10)
                plt.ylim(-3, 6)
                plt.xticks(())
                plt.yticks(())
                plt.title(title)

            plt.show()
            exit(-1)

        #Comentar aqui para funcionamiento normal
        test = True
        if test:
            plot_test()
                    
        authors = [self.db.get_author(a, True) for a in authors_id]
        samples = self.get_matrix(authors)
        
        n_samples = 500

        # Generate random sample, two components
        fig = plt.figure()
        np.random.seed(0)
        C = np.array([[0., -0.1], [1.7, .4]])
        X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
                  .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
                  
        print x
        print multivariate_normal.pdf(x, mean=mean, cov=covar)
        exit(-1)
        
        self.scaler = MinMaxScaler()
        self.scaler.fit(samples)

        self.pca = None
        self.pca = PCA(n_components=40)
        self.pca.fit(samples)

        if self.scaler:
            samples = self.scaler.transform(samples)
        if self.pca:
            samples = self.pca.transform(samples)
            
        self.mean = np.mean(samples, axis=0)
        self.std = np.std(samples, axis=0)
    
        # print samples.shape
        distances = []

        gt = self.db.get_ground_truth(self.language)
        
        components=len(authors_id)/2
        n_classes = len(np.unique(gt.values()))    
        
        classifiers = dict((covar_type, GMM(n_components=components,
                            covariance_type=covar_type))
                           for covar_type in ['diag']) 
                           #'spherical', 'diag', 'tied', 'full'])
                           
        n_classifiers = len(classifiers)

        for index, (name, classifier) in enumerate(classifiers.items()):            
            classifier.fit(samples)
            
        
        tp = 'diag'
        self.c = classifiers[tp]

        gms = []
        for i in samples:
            # print 'Sample'
            # print i
            # print 'Adaptation'
            agm = GMM(n_components=components,covariance_type=tp)
            agm.weights_, agm.means_, agm.covars_ = \
                    self.em(self.c.weights_, self.c.means_, self.c.covars_,  [i], 16)
            gms.append(agm)
            a, b = self.c.score(i), agm.score(i)
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

            values.append((agm.score(ud)/self.c.score(ud),target))

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

        descriptor = self.get_matrix([author], True)
        if self.scaler:
            descriptor = self.scaler.transform(descriptor)
        if self.pca:
            descriptor = self.pca.transform(descriptor)
        descriptor = descriptor[0]
        
        agm = GMM(n_components=len(self.c.means_),covariance_type='diag')
        agm.weights_, agm.means_, agm.covars_ = \
            self.em(self.c.weights_, self.c.means_, self.c.covars_, [descriptor], 16)
            
        unknown_descriptor = self.get_matrix([author], False)
        
        if self.scaler:
            unknown_descriptor = self.scaler.transform(unknown_descriptor)
        if self.pca:
            unknown_descriptor = self.pca.transform(unknown_descriptor)
        ud = unknown_descriptor[0]
        
        a, b = self.c.score(unknown_descriptor), agm.score(unknown_descriptor)
        
        if agm.score(ud)/self.c.score(ud) < self.threshold:
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
