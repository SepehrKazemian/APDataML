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
import lumping_traditional as oldLumping
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
# print(data)
data = processData.dataFrameManipulation(data)
numberOfStates = 255
cuTrans = processData.markovianTransitionMatrixDegree1(data, numberOfStates)
normalizedCuTrans = processData.normalizingTransMatrix(cuTrans)
cuTrans_cpy = normalizedCuTrans.copy()

arrExtra = [0]
print("done")
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
#*******************************************************************************

def bandwidthPercentage(vectorMatrix):
    percentageIncreament = (100 / vectorMatrix.shape[0])
    percentageMatrix = []
    maxPercentage = 0
    for j in range(vectorMatrix.shape[0]):
        maxPercentage += percentageIncreament
        percentageMatrix.append([[j], maxPercentage, False])

    return percentageMatrix

def reduceMatrixAllInOne(transitionMatrix):
    reload(oldLumping)
    percentageMatrix_list = bandwidthPercentage(transitionMatrix)
    zero_cols_rows = []
    #*************removing zeros from columns and rows (matrix reduction) ****************
    for i in range(len(transitionMatrix)):
        if (np.sum(transitionMatrix[i]) == 0) and (np.sum(transitionMatrix[:,i]) == 0):
            zero_cols_rows.append(i)


    irreducible_matrix = transitionMatrix.copy()
    for i in range(len(zero_cols_rows) - 1, -1, -1):
        irreducible_matrix = np.delete(irreducible_matrix, zero_cols_rows[i], axis = 0)
        irreducible_matrix = np.delete(irreducible_matrix, zero_cols_rows[i], axis = 1)
        percentageMatrix_list[zero_cols_rows[i]][2] = True


    for i in range(len(percentageMatrix_list) - 1, 0, -1):
        if (percentageMatrix_list[i][2] == True) and (percentageMatrix_list[i - 1][2] == True):
            percentageMatrix_list[i - 1][0].extend(percentageMatrix_list[i][0])
            percentageMatrix_list[i - 1][1] = percentageMatrix_list[i][1]
            del percentageMatrix_list[i]

    result = oldLumping.lumping(irreducible_matrix, percentageMatrix_list, False)



def reduceMatrixOneByOne(transitionMatrix):
    reload(oldLumping)
    percentageMatrix_list = bandwidthPercentage(transitionMatrix)
    zero_cols_rows = []
    #*************removing zeros from columns and rows (matrix reduction) ****************
    for i in range(len(transitionMatrix)):
        if (np.sum(transitionMatrix[i]) == 0) and (np.sum(transitionMatrix[:,i]) == 0):
            zero_cols_rows.append(i)


    irreducible_matrix = transitionMatrix.copy()
    for i in range(len(zero_cols_rows) - 1, -1, -1):
        irreducible_matrix = np.delete(irreducible_matrix, zero_cols_rows[i], axis = 0)
        irreducible_matrix = np.delete(irreducible_matrix, zero_cols_rows[i], axis = 1)
        percentageMatrix_list[zero_cols_rows[i]][2] = True


    for i in range(len(percentageMatrix_list) - 1, 0, -1):
        if (percentageMatrix_list[i][2] == True) and (percentageMatrix_list[i - 1][2] == True):
            percentageMatrix_list[i - 1][0].extend(percentageMatrix_list[i][0])
            percentageMatrix_list[i - 1][1] = percentageMatrix_list[i][1]
            del percentageMatrix_list[i]


    savedResult = 0
    while True:

        result = oldLumping.lumping(irreducible_matrix, percentageMatrix_list, True)
        print(irreducible_matrix)

        if result == None:
            return savedResult, percentageMatrix_list
            break
        else:
            savedResult = result

        min_degree, min_error, irreducible_matrix, best_sectors = result[0], result[1], result[2], result[3]

        zero_cols_rows = []
        for i in range(len(irreducible_matrix)):
            if (np.sum(irreducible_matrix[i]) == 0) and (np.sum(irreducible_matrix[:,i]) == 0):
                zero_cols_rows.append(i)

        if len(zero_cols_rows) != 0:
            for i in range(len(zero_cols_rows) - 1, -1, -1):
                irreducible_matrix = np.delete(irreducible_matrix, zero_cols_rows[i], axis = 0)
                irreducible_matrix = np.delete(irreducible_matrix, zero_cols_rows[i], axis = 1)

            percentageMatrix_list = reduceMatrix(percentageMatrix_list, zero_cols_rows)



        # print("min_degree is: ", min_degree)
        # print("min_error is: ", min_error)
        # print("best sectoring is: ", best_sectors)
        # print("best lumped is: ", irreducible_matrix)
        print("length of best lumped is: ", len(irreducible_matrix))

        percentageMatrix_list = mergeStates(best_sectors, percentageMatrix_list)
        # print("percentageMatrix is: ", percentageMatrix_list)

        count = 0
        for i in range(len(percentageMatrix_list)):
            if percentageMatrix_list[i][2] == False:
                count += 1
        print("percentageMatrix length is: ", count)

def reduceMatrix(percentageMatrix, zero_cols_rows):
    index_in_percentage_matrix = np.inf
    counter = 0
    for x in range(len(zero_cols_rows)):
        for i in range(len(percentageMatrix)):
            if percentageMatrix[i][2] == False:
                if zero_cols_rows[x] == counter:
                    index_in_percentage_matrix = i
                    break
                counter += 1
        percentageMatrix[index_in_percentage_matrix][2] = True
        if index_in_percentage_matrix == (len(percentageMatrix) - 1):
            if percentageMatrix[index_in_percentage_matrix - 1][2] == True:
                percentageMatrix[index_in_percentage_matrix - 1][0].extend(
                    percentageMatrix[index_in_percentage_matrix][0])

                percentageMatrix[index_in_percentage_matrix - 1][1] = percentageMatrix[index_in_percentage_matrix][1]
                del percentageMatrix[index_in_percentage_matrix]

        elif index_in_percentage_matrix == 0:
            if percentageMatrix[index_in_percentage_matrix + 1][2] == True:
                percentageMatrix[index_in_percentage_matrix][0].extend(
                    percentageMatrix[index_in_percentage_matrix + 1][0])

                percentageMatrix[index_in_percentage_matrix][1] = percentageMatrix[index_in_percentage_matrix + 1][1]
                del percentageMatrix[index_in_percentage_matrix + 1]

        else:
            if (percentageMatrix[index_in_percentage_matrix - 1][2] == True) and (
                percentageMatrix[index_in_percentage_matrix + 1][2] == True):
                leftside_bandwidth_val = 0
                rightside_bandwidth_val = 0
                if (index_in_percentage_matrix - 1) != 0:
                    leftside_bandwidth_val = percentageMatrix[index_in_percentage_matrix][1] - percentageMatrix[index_in_percentage_matrix - 2][1]
                else:
                    leftside_bandwidth_val = percentageMatrix[index_in_percentage_matrix][1]
                rightside_bandwidth_val = percentageMatrix[index_in_percentage_matrix + 1][1] - percentageMatrix[index_in_percentage_matrix - 1][1]

                if leftside_bandwidth_val <= rightside_bandwidth_val:
                    percentageMatrix[index_in_percentage_matrix - 1][0].extend(
                        percentageMatrix[index_in_percentage_matrix][0])

                    percentageMatrix[index_in_percentage_matrix - 1][1] = percentageMatrix[index_in_percentage_matrix][1]
                    del percentageMatrix[index_in_percentage_matrix]

                else:
                    percentageMatrix[index_in_percentage_matrix][0].extend(
                        percentageMatrix[index_in_percentage_matrix + 1][0])

                    percentageMatrix[index_in_percentage_matrix][1] = percentageMatrix[index_in_percentage_matrix + 1][1]
                    del percentageMatrix[index_in_percentage_matrix + 1]

            elif (percentageMatrix[index_in_percentage_matrix - 1][2] == True) and (
                percentageMatrix[index_in_percentage_matrix + 1][2] == False):

                percentageMatrix[index_in_percentage_matrix - 1][0].extend(
                    percentageMatrix[index_in_percentage_matrix][0])

                percentageMatrix[index_in_percentage_matrix - 1][1] = percentageMatrix[index_in_percentage_matrix][1]
                del percentageMatrix[index_in_percentage_matrix]

            elif (percentageMatrix[index_in_percentage_matrix - 1][2] == False) and (
                percentageMatrix[index_in_percentage_matrix + 1][2] == True):

                percentageMatrix[index_in_percentage_matrix][0].extend(
                    percentageMatrix[index_in_percentage_matrix + 1][0])

                percentageMatrix[index_in_percentage_matrix][1] = percentageMatrix[index_in_percentage_matrix + 1][1]
                del percentageMatrix[index_in_percentage_matrix + 1]

    return percentageMatrix




def mergeStates(best_sectors, percentageMatrix):
    False_state_in_percentageMatrix = np.inf
    for i in range(1, len(best_sectors)):
        if best_sectors[i] - best_sectors[i - 1] == 2:
            False_state_in_percentageMatrix = best_sectors[i]
            #it means states number (whole_subset[i] - 1) and (whole_subset[i] - 2) should merge
            #it is also the (whole_subset[i]) False and (whole_subset[i] - 1) False in percentageMatrix
            break

    index1_in_percentage_matrix = np.inf
    index2_in_percentage_matrix = np.inf
    counter = 0
    for i in range(len(percentageMatrix)):
        if percentageMatrix[i][2] == False:
            counter += 1
            if counter == (False_state_in_percentageMatrix - 1):
                index1_in_percentage_matrix = i
            elif counter == False_state_in_percentageMatrix:
                index2_in_percentage_matrix = i
                break

    # print("indices are: ", index1_in_percentage_matrix, index2_in_percentage_matrix)
    # print("values of indices: ", percentageMatrix[index1_in_percentage_matrix])
    # print("values of indices: ", percentageMatrix[index2_in_percentage_matrix])
    if (index2_in_percentage_matrix) == (index1_in_percentage_matrix + 1):
        percentageMatrix[index1_in_percentage_matrix][0].extend(
            percentageMatrix[index2_in_percentage_matrix][0])

        percentageMatrix[index1_in_percentage_matrix][1] = percentageMatrix[
            index2_in_percentage_matrix][1]

        del percentageMatrix[index2_in_percentage_matrix]

    elif (index2_in_percentage_matrix) == (index1_in_percentage_matrix + 2):
        percentageMatrix[index1_in_percentage_matrix + 1][0].extend(
            percentageMatrix[index2_in_percentage_matrix][0])

        percentageMatrix[index1_in_percentage_matrix][0].extend(
            percentageMatrix[index1_in_percentage_matrix + 1][0])

        percentageMatrix[index1_in_percentage_matrix][1] = percentageMatrix[
            index2_in_percentage_matrix][1]

        del percentageMatrix[index2_in_percentage_matrix]
        del percentageMatrix[index1_in_percentage_matrix + 1]

    return percentageMatrix

lumpedResult = matrix_irreducibility(cuTrans_cpy[0])
print(lumpedResult[0][2])

#
#     sizeOfNewArr = len(deleteList)
# #     print(deleteList)
#     newArr = np.zeros(shape=(26-sizeOfNewArr, 26-sizeOfNewArr))
