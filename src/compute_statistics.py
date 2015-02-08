#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import re
import argparse
import numpy as np

from utils import *
from db_layer import *
from feature_extractor import *

if __name__ != '__main__':
    os.sys.exit(1)

parser = argparse.ArgumentParser(\
    description="Train and compute the authors' features.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--language', metavar="lang", nargs='?',
                    default=["DU", "EN", "GR", "SP"],
                    help='Language')
parser.add_argument('--config', metavar="conf", nargs='?',
                    default="conf/config.json", help='Configuration file')
args = parser.parse_args()

config = get_configuration(args.config)
db = db_layer(args.config)

if type(args.language) == str:
    args.language = [args.language]

for ln in args.language:
    print "Language:", ln

    authors = db.get_authors(ln)
    authors = [db.get_author(a, reduced=True) for a in authors]

    features = [a["features"].keys() for a in authors]
    features = list(set(utils.flatten(features)))
    features.sort()

    for feature in features:
        utils.graph_feature_quality(feature, authors)
