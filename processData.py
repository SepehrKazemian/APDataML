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
warnings.filterwarnings('always')
# <codecell>


def processingData():
    address = input("the address of the collected data files (not alligned files or CSV files): ")
    fileNameArr = os.listdir(address)

    extraArr = []
    for i in range(len(fileNameArr)):
        if ".txt" not in fileNameArr[i]:
            extraArr.append(fileNameArr[i])

    for i in range(len(extraArr)):
        fileNameArr.remove(extraArr[i])

    print(fileNameArr)



    timerInMinute = 30 #chunking files to 30 minutes
    timeInterval = 6 #seconds

    CU_FileChunks = None
    fileNameArr = ["500f80271400.txt"] #for test we are giving a ready file

    LA = classImportLA.learningAlgs() #calling an object of the class

    #*************checking if we have the processed version of file in our CSV or not, if not we are gonna process data otherwise just we gonna read it
    pathFile = address + "/CSV/" + str(fileNameArr[0]) + ".csv"
    filePath = address + "/" + str(fileNameArr[0])
    importlib.reload(dataMan) #reload the class if it has cache (.pyc) to load the class from scratch
    if os.path.isfile(pathFile) == False:
            print("we do not have processed data for file " + str(fileNameArr[0]) + " so we are making it")
            numOfLines = int(os.popen('wc -l < ' + str(filePath)).read().split()[0])
            print("number of lines is: " + str(numOfLines))
            dataMan.normalDataSplitting(fileNameArr[0], 0, 0, timeInterval, address)

    # print(fileNameArr[0])
    # print(address)
    stat, data = LA.csvChecker(fileNameArr[0], 0, address)

    print(data.head())
    print("now we have the processed data from pandas")
    data["CU/255"] = data["CU"] / 255 #we add a column to our pandas table

    return data

def dataFrameManipulation(data):
    minutes = int(input("please enter how long would be the chunk minutes? "))
    data["time"] = data["time"].apply(lambda x: x + timedelta(hours = -7)) #changing time from UTC to Mountain

    #making data ready for assigning an Index to each 30 minutes interval
    secondsPerChunk = int(60 * minutes) #for 30 minutes interval we have these many seconds
    data["timeIndex"] = -1
    startIndex = 0
    timeIndexVal = 1
    data["timeIndex"] = data["time"].apply(lambda x: math.floor(((x.second + (x.minute * 60) + (x.hour * 3600)) / secondsPerChunk)))
    numberOfChunks = max(data["timeIndex"]) + 1
    dataSet = data.copy() #copying the dataFrame before manipulating it again
    print("removing weekends from the data")
    data["weekDay"] = data["time"].apply(lambda x: x.weekday()) #printing the weekdays numbers
    data = data.drop(data["weekDay"].loc[(data["weekDay"] == 6) | ((data["weekDay"] == 5))].index, axis=0)
    return data


def markovianTransitionMatrixDegree1(data, numberOfClasses):
    dataCpy = data.copy()
    if numberOfClasses > 255:
        return 0
    classCoeff = 255 / (numberOfClasses)
    numberOfChunks = max(data["timeIndex"])
    cuTrans = np.zeros(shape=(numberOfChunks, numberOfClasses, numberOfClasses))
    start = -1
    next = -1
    prevChunkVal = -1
    newChunkVal = -1
    firstIndexOfChunk = -1

    start = -1
    next = -1
    prevChunkVal = -1
    newChunkVal = -1
    firstIndexOfChunk = -1
    counter = 0
    maxValuePlusOne = 27

    for x in range(numberOfChunks):
        iterPandas = data.loc[data["timeIndex"] == x]
        start = -1
        next = -1

        for index, row in iterPandas.iterrows():
            start = next
            next = math.floor(row["CU"] / classCoeff)
            if start != -1:
                cuTrans[x, math.floor(start), math.floor(next)] += 1
    return cuTrans

def normalizingTransMatrix(cuTrans):
    number_of_samples = np.zeros(shape=(cuTrans.shape[0], cuTrans.shape[1]))
    ans = cuTrans.copy()
    for x in range(cuTrans.shape[0]):
        for i in range(cuTrans.shape[1]):
            sum = 0
            sum = np.sum(ans[x][i])
            number_of_samples[x][i] = int(sum)
            if sum != 0:
                ans[x][i] = ans[x][i]/sum
    return ans
