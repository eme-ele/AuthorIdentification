#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import json
import math
import os


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
