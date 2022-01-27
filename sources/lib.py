# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 14:56:14 2022

@author: quang
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from pyparsing import col
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import classification_report

import pickle

#from sympy import ordered
from lib import *
from data_exploratory_test import *


import timeit
