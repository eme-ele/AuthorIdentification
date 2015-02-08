#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import json
import math
import os
from matplotlib import pyplot as plt

def get_configuration(filename):
    try:
        f = open(filename)
        ret = json.loads(f.read())
        f.close()
        return ret
    except:
        return {}


def flatten(l):
    return [item for sublist in l for item in sublist]


def graph_feature_quality(feature, authors):
    def distance(a, b):
        return abs(a - b)

    values = [a["features"][feature] for a in authors]
    values.sort()

    all_distances = []
    for v in values:
        all_distances += [distance(v, w) for w in values]
    all_distances = list(set(all_distances))

    distance_error = {d: 0 for d in all_distances}

    for v in values:
        for d in all_distances:
            next_distances = [distance(v, w) for w in values]
            next_distances = filter(lambda x: x <= d, next_distances)
            distance_error[d] += 1.0 - float(len(next_distances) - 1) / \
                                       len(values)
    
    xs = [d for d in distance_error]
    xs.sort()
    x_max_val = max(xs)
    x_min_val = min(xs)

    ys = [distance_error[d] for d in xs]
    y_max_val = max(ys)
    y_min_val = min(ys)
    
    plt.plot([(x - x_min_val) / (x_max_val - x_min_val + 1e-6) for x in xs],
             [(y - y_min_val) / (y_max_val - y_min_val + 1e-6) for y in ys])

    plt.title(feature)
    plt.show()
