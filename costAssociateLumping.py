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

#*******************************************************************************
# %%codecell
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

dataFrame = processingData()
#address is: /home/netlab/Desktop/thesis/data/1node1-3-5/
#/home/netlab/Desktop/thesis/data/500f80271400/
data = dataFrame.copy() #copying the dataFrame to have a copy of not edited data

print("hello")
# <codecell>

#*******************************************************************************
# %% codecell
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

data = dataFrameManipulation(data)

def markovianTransitionMatrixDegree1(data, coeffBool):
    dataCpy = data.copy()
#     dataCpy = data
    numberOfChunks = max(data["timeIndex"]) + 1
    cuTrans = np.zeros(shape=(numberOfChunks, 255, 255))
    start = -1
    next = -1
    prevChunkVal = -1
    newChunkVal = -1
    firstIndexOfChunk = -1



#     cuTrans = np.zeros(shape=(numberOfChunks, 26, 26))
#     cuDifTrans = np.zeros(shape=(numberOfChunks, 26, 26))

    start = -1
    next = -1
    prevChunkVal = -1
    newChunkVal = -1
    firstIndexOfChunk = -1
#     indexesOfChunks = dataCpy.index[dataCpy["timeIndex"] == i]
    counter = 0
    maxValuePlusOne = 27

    for x in range(numberOfChunks):
        iterPandas = data.loc[data["timeIndex"] == x]
        start = -1
        next = -1

        for index, row in iterPandas.iterrows():
            start = next
            next = row["CU"]
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
    return ans, number_of_samples

##For 30 minutes chunk
cuTrans = markovianTransitionMatrixDegree1(data, 0)
normalizedCuTrans, number_of_samples = normalizingTransMatrix(cuTrans)
cuTrans_cpy = normalizedCuTrans.copy()
# <codecell>
#*******************************************************************************

#*******************************************************************************
# %% codecell
def steadyState(transitionMatrix):
    ss_transitionMatrix = np.zeros(shape=(transitionMatrix.shape[0]))
    ss_transitionMatrix = abs(pysal.spatial_dynamics.ergodic.
                                 steady_state(transitionMatrix))
    return ss_transitionMatrix


def bandwidthPercentage(vectorMatrix):
    percentageIncreament = (100 / vectorMatrix.shape[0])
    percentageMatrix = np.zeros(shape=(vectorMatrix.shape[0]))
    maxPercentage = 0
    for j in range(vectorMatrix.shape[0]):
        maxPercentage += percentageIncreament
        percentageMatrix[j] = maxPercentage
    return percentageMatrix


def assymetricDistrib(percentageVector, arrayOfStatesPercentage, predictedStateIndex):
    xAxisPoints = np.linspace(rayleigh.ppf(0.01), rayleigh.ppf(0.99), 338)
    #number of overal datapoints must stay the same all the time
    maxState = 338

    inverseDistrib = max(rayleigh.pdf(xAxisPoints)) - rayleigh.pdf(xAxisPoints)
    minState = np.argmin(inverseDistrib)

    avgUnderUtilizedSum = 0
    numberOfUnderUtilizedStates = minState - 0
    for i in range(predictedStateIndex + 1, len(arrayOfStatesPercentage)):
        underUtilizedPercentage = percentageVector[i] - percentageVector[predictedStateIndex]
        index_On_UnderUtilized_Distribution = math.floor((underUtilizedPercentage *
                                                          numberOfUnderUtilizedStates) / 100)
        index_On_UnderUtilized_Distribution = minState - index_On_UnderUtilized_Distribution
        avgUnderUtilizedSum += inverseDistrib[index_On_UnderUtilized_Distribution] * arrayOfStatesPercentage[i]

    if predictedStateIndex != len(arrayOfStatesPercentage) - 1:
        avgUnderUtilizedSum /= (len(arrayOfStatesPercentage) - (predictedStateIndex + 1))



    avgOverUtilizedSum = 0
    numberOfOverUtilizedStates = maxState - minState
    for i in range(0, predictedStateIndex):
        overUtilizedPercentage = percentageVector[predictedStateIndex] - percentageVector[i]
        index_On_OverUtilized_Distribution = math.floor((overUtilizedPercentage *
                                                         numberOfOverUtilizedStates) / 100)
        index_On_OverUtilized_Distribution += minState
        avgOverUtilizedSum += inverseDistrib[index_On_OverUtilized_Distribution] * arrayOfStatesPercentage[i]

    if predictedStateIndex != 0:
        avgOverUtilizedSum /= (predictedStateIndex)

    avgPenalty = avgOverUtilizedSum + avgUnderUtilizedSum

    avgPenalty *= 1-(arrayOfStatesPercentage[predictedStateIndex])
    return avgPenalty

def statePenaltyValue(steadyStateVectorArray, percentageMatrix):
    ssPenaltyValue = np.zeros(shape = (steadyStateVectorArray.shape[0]))
    for j in range(steadyStateVectorArray.shape[0]):
        ssPenaltyValue[j] = assymetricDistrib(percentageMatrix,
                                              steadyStateVectorArray, j)

    return ssPenaltyValue

# statePenaltyNumpy = statePenaltyValue(steadyStateVectorArray, percentageMatrix)


def neighbor_states_differences(statePenaltyNumpy):
    neighbor_difference_array = np.zeros(shape = (statePenaltyNumpy.shape[0] - 1))
    for j in range(statePenaltyNumpy.shape[0] - 1):
        neighbor_difference_array[j] = abs(statePenaltyNumpy[j] -
                                           statePenaltyNumpy[j + 1])

    return neighbor_difference_array

# neighbor_difference_array = nieghbor_states_differences(statePenaltyNumpy)

def mergingStates(normalizedCuTrans, min_diff):
    extra_Normalized_Transition_Matrix = np.zeros(shape = (normalizedCuTrans.shape[1] - 1,
                                                           normalizedCuTrans.shape[1] - 1))

    normalizedCuTrans[:, min_diff] = normalizedCuTrans[:, min_diff] + normalizedCuTrans[:, min_diff + 1]
    normalizedCuTrans[min_diff, :] = normalizedCuTrans[min_diff, :] + normalizedCuTrans[min_diff + 1, :]
    normalizedCuTrans[min_diff, :] = normalizedCuTrans[min_diff, :] /2
    extraNumpy = np.delete(normalizedCuTrans, min_diff + 1, 0)
    extra_Normalized_Transition_Matrix = np.delete(extraNumpy, min_diff + 1, 1)

    return extra_Normalized_Transition_Matrix

def updatePercentage(percentageMatrix, min_diff):
    newPercentageMatrix = np.zeros(shape = (percentageMatrix.shape[0] - 1))

    newPercentageMatrix = np.delete(percentageMatrix, min_diff, 0)

    return newPercentageMatrix
# <codecell>
#*******************************************************************************


def clustering(normalizedTransitionMatrix):

    for i in range(normalizedTransitionMatrix.shape[0]):

        percentageMatrix = bandwidthPercentage(normalizedTransitionMatrix[i])
        # a percentage Matrix for each time interval

        mergedStates = normalizedTransitionMatrix[i]
        #running for each time interval independently
        while True:

            steadyStateVector = steadyState(mergedStates)
            # every time interval becomes a transition matrix vector

            statePenaltyNumpy = statePenaltyValue(steadyStateVector, percentageMatrix)
            # finding penalty for each state in the vector

            neighbor_difference_array = neighbor_states_differences(statePenaltyNumpy)
            # difference of neighbours' penlty

            extra_diff_arr = neighbor_difference_array.copy()

            constraint_counter = 0
            while constraint_counter < extra_diff_arr.shape[0]:
                min_index_in_penalty = np.argmin(extra_diff_arr)

                if min_index_in_penalty != 0:
                    if 25 < percentageMatrix[min_index_in_penalty + 1] - percentageMatrix[min_index_in_penalty - 1]:
                        extra_diff_arr[min_index_in_penalty] = 100
                        constraint_counter += 1
                        continue

                elif percentageMatrix[min_index_in_penalty + 1] > 25:
                    extra_diff_arr[min_index_in_penalty] = 100
                    constraint_counter += 1
                    continue

                mergedStates = mergingStates(mergedStates, min_index_in_penalty)
                # merging states with closest difference w.r.t penalty


                percentageMatrix = updatePercentage(percentageMatrix, min_index_in_penalty)
                # updating bandwidth percentage after merging states

                break


            if constraint_counter == extra_diff_arr.shape[0]:
                print("the value of i is: ", i)
                print(mergedStates)
                print(percentageMatrix)
                break




transExtra = cuTrans_cpy.copy()
clustering(transExtra)

min_diff = np.argmin(neighbor_difference_array) #minimum of difference is for x and x + 1

cuTransCopy = cuTrans.copy()


cuTransCopy[20][:, min_diff] = cuTransCopy[20][:, min_diff] + cuTransCopy[20][:, min_diff - 1]
cuTransCopy[20][min_diff, :] = cuTransCopy[20][min_diff, : ] + cuTransCopy[20][min_diff - 1, : ]

print(cuTrans[20][2])
print(cuTrans[20][3])
print(np.sum(cuTrans[20][2]))
print(np.sum(cuTrans[20][3]))

print(neighbor_difference_array)

# neighbor_difference_list = [0 for i in range(statePenaltyNumpy.shape[1])]
#
# for i in range(statePenaltyNumpy.shape[1]):
#     if i == 0:
#         neighbor_difference_list[i] = (np.inf, np.abs(statePenaltyNumpy[20][i] - statePenaltyNumpy[20][i + 1]))
#
#     elif i == (statePenaltyNumpy.shape[1] - 1):
#         neighbor_difference_list[i] = (np.abs(statePenaltyNumpy[20][i] -
#                                             statePenaltyNumpy[20][i - 1]), np.inf)
#
#     else:
#         neighbor_difference_list[i] = (np.abs(statePenaltyNumpy[20][i] -
#                                             statePenaltyNumpy[20][i - 1])
#                                     , np.abs(statePenaltyNumpy[20][i] -
#                                           statePenaltyNumpy[20][i + 1]))
#
# neighbor_difference_list
print(statePenaltyNumpy[20])
