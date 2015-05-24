import sys
import os
import re
import optparse
import numpy as np
import json
import commands as cmd

from src.utils import *
from src.importer import *
from src.db_layer_simple import *
from src.feature_extractor import *
from src.classifier import *


parser = optparse.OptionParser()
parser.add_option('-i', help='path to training corpus', dest='documents_path', type='string')
parser.add_option('-o', help='path to output directory', dest='model_path', type='string')
(opts, args) = parser.parse_args()

mandatories = ['documents_path', 'model_path']
for m in mandatories:
    if not opts.__dict__[m]:
        print "mandatory option is missing"
        parser.print_help()
        exit(-1)

documents_path = opts.documents_path
model_path = opts.model_path
contents_path = os.path.join(opts.documents_path, 'contents.json')

with open(contents_path) as contents_file:
    contents_dict = json.load(contents_file)

language = contents_dict["language"]
authors = contents_dict["problems"]
config_file = "conf/config_run.json"
config = get_configuration(config_file)

db = db_layer(language, authors, config_file, documents_path)

print "Language:", language
print "Number of examples in training set:", len(authors)

ln = db.get_language()

## features
fe = concat_fe(config_file, db,
               [
                   clear_fe(config_file, db),
                   #pos_fe(config_file),
                   hapax_fe(config_file, db),
                   word_distribution_fe(config_file, db),
                   num_tokens_fe(config_file, db),
                   stop_words_fe(config_file, db),
                   punctuation_fe(config_file, db),
                   structure_fe(config_file, db),
                   char_distribution_fe(config_file, db),
                   spacing_fe(config_file, db),
                   punctuation_ngrams_fe(config_file, db),
                   stopword_topics_fe(config_file, db),
                   word_topics_fe(config_file, db)
               ])


print "Clearing features..."
for id_author, author in enumerate(authors):
    db.clear_features(author, commit=True)

    if id_author % 10 == 0:
        print "%0.2f%%\r" % (id_author * 100.0 / len(authors)),
        os.sys.stdout.flush()

print "Training features..."
fe.train(authors)
db.store_feature_extractor(fe, ln)

print "Computing features..."
for id_author, author in enumerate(authors):
    author = fe.compute(author, known=True)
    author = fe.compute(author, known=False)

    if (id_author + 1) % 10 == 0:
        print "%0.2f%%\r" % ((id_author + 1) * 100.0 / len(authors)),
        os.sys.stdout.flush()
print

print "Training model..."
w_clf = rf_classifier(config_file, db, ln)
models = [
          ("Weights", weighted_distance_classifier(config_file, db, ln)),
          ("reject-RF", reject_classifier(config_file, db, ln,
                                          rf_classifier(config_file, db,
                                                        ln))),
          ("adj-RF",  adjustment_classifier(config_file, db, ln,
                                            rf_classifier(config_file, db,
                                            ln))),
          ("rej-adj-RF",
           adjustment_classifier(config_file, db, ln,
                                 reject_classifier(config_file, db, ln,
                                           rf_classifier(config_file, db,
                                           ln)))),
          ("RF", rf_classifier(config_file, db, ln)),
          ("UBM", ubm(config_file, db, ln, fe,  n_pca=5, \
                             n_gaussians=2, r=8, normals_type='diag')),
         ]
model = model_selector(config_file, db, ln, [x[1] for x in models])
model.train(authors)

print "Storing model on", opts.model_path, "..."
db.store_model(opts.model_path, model)
