# %% codecell
import numpy as np
import Plot as plot
import math
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
import time
import datetime
import pickle
import os
import scipy.spatial
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib as plt
import logging
from scipy import signal
import learningAlgs as classImportLA
import dataManipulation as dataMan
from itertools import permutations
import importlib
from datetime import timedelta
from multiprocessing import Pool
import multiprocessing
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cluster import KMeans
import timeIntervalPlotter as intervalPlotter
import pysal
import warnings
import OldLumping as oldLumping
import boundaryFull_SS_WeightedLumping as WLumping
from importlib import reload
from scipy.stats import rayleigh
import dataManipulation as dataMan
import matplotlib.pyplot as plt
import processData as processData
import tensorflow as tf
warnings.filterwarnings('always')
# <codecell>

#*******************************************************************************
# %%codecell
reload(classImportLA)
dataFrame = processData.processingData()
#address is: /home/netlab/Desktop/thesis/data/1node1-3-5/
#/home/netlab/Desktop/thesis/data/500f80271400/
data = dataFrame.copy() #copying the dataFrame to have a copy of not edited data

print("hello")
# <codecell>

#*******************************************************************************
# %% codecell
print(data)
data = processData.dataFrameManipulation(data)
numberOfStates = 255
cuTrans = processData.markovianTransitionMatrixDegree1(data, numberOfStates)
normalizedCuTrans = processData.normalizingTransMatrix(cuTrans)
cuTrans_cpy = normalizedCuTrans.copy()

arrExtra = [0]
def particling(x):
    if (x["col1"] - 1 not in data["col1"]) or (x["col1"] - 1 < 0):
        arrExtra[0] = x["time"]
        return 0

    elif x["timeIndex"] != data["timeIndex"][x["col1"] - 1]:
        arrExtra[0] = x["time"]
        return 0

    else:
        return ((x["time"] - arrExtra[0]).seconds)/6

data["periodParticle"] = data.apply(lambda x: particling(x), axis = 1) #numbering each 6 seconds in the dataFrame

print(cuTrans_cpy[0])
# <codecell>
