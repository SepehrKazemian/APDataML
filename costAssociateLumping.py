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
warnings.filterwarnings('always')
# <codecell>

#*******************************************************************************
# %%codecell

dataFrame = processData.processingData()
#address is: /home/netlab/Desktop/thesis/data/1node1-3-5/
#/home/netlab/Desktop/thesis/data/500f80271400/
data = dataFrame.copy() #copying the dataFrame to have a copy of not edited data

print("hello")
# <codecell>

#*******************************************************************************
# %% codecell

data = processData.dataFrameManipulation(data)
numberOfStates = 255
##For 30 minutes chunk
cuTrans = processData.markovianTransitionMatrixDegree1(data, numberOfStates)
normalizedCuTrans = processData.normalizingTransMatrix(cuTrans)
cuTrans_cpy = normalizedCuTrans.copy()
print(cuTrans_cpy[0])
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

    # print("sum of the min_diff: ", np.sum(normalizedCuTrans[min_diff]))
    # print("sum of the min_diff + 1: ", np.sum(normalizedCuTrans[min_diff + 1]))
    if np.sum(normalizedCuTrans[min_diff, :]) != 0 and np.sum(
        normalizedCuTrans[min_diff + 1, :]) != 0:
        normalizedCuTrans[min_diff, :] = normalizedCuTrans[min_diff, :] + normalizedCuTrans[min_diff + 1, :]
        normalizedCuTrans[min_diff, :] = normalizedCuTrans[min_diff, :] /2
    else:
        normalizedCuTrans[min_diff, :] = normalizedCuTrans[min_diff, :] + normalizedCuTrans[min_diff + 1, :]

    normalizedCuTrans[:, min_diff] = normalizedCuTrans[:, min_diff] + normalizedCuTrans[:, min_diff + 1]
    extraNumpy = np.delete(normalizedCuTrans, min_diff + 1, 0)
    extra_Normalized_Transition_Matrix = np.delete(extraNumpy, min_diff + 1, 1)
    # print("sum of this row now is: ", np.sum(extra_Normalized_Transition_Matrix[min_diff]))

    return extra_Normalized_Transition_Matrix

def updatePercentage(percentageMatrix, min_diff):
    newPercentageMatrix = np.zeros(shape = (percentageMatrix.shape[0] - 1))

    newPercentageMatrix = np.delete(percentageMatrix, min_diff, 0)

    return newPercentageMatrix
# <codecell>
#*******************************************************************************


def clustering(normalizedTransitionMatrix):

    overal_merged_states = []
    overal_bandwidth_vector = []

    for i in range(normalizedTransitionMatrix.shape[0]):
    # for i in range(1):

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
                overal_merged_states.append(mergedStates)
                overal_bandwidth_vector.append(percentageMatrix)
                break

    return overal_merged_states, overal_bandwidth_vector




transExtra = cuTrans_cpy.copy()
overal_merged_states, overal_bandwidth_vector = clustering(transExtra)
