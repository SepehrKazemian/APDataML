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
data = processData.dataFrameManipulation(data)
numberOfStates = 255
reload(processData)
cuTrans = processData.markovianTransitionMatrixDegree1(data, numberOfStates, "CU")
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



def preparingMatrixForLumping(transitionMatrix):
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

    for i in range(len(irreducible_matrix)):
        if np.sum(irreducible_matrix[i], dtype = np.float32) != 1.0:
            print(np.sum(irreducible_matrix[i], dtype = np.float32))

    return percentageMatrix_list, irreducible_matrix

def lumpingStatesOneByOne(percentageMatrix_list, irreducible_matrix):

    # savedResult = 0
    result = oldLumping.lumping(irreducible_matrix, percentageMatrix_list, True)

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

    print("length of best lumped is: ", len(irreducible_matrix))

    percentageMatrix_list = mergeStates(best_sectors, percentageMatrix_list)

    for i in range(len(irreducible_matrix)):
        if (np.sum(irreducible_matrix[i], dtype = np.float32) != 1.0):

            print(i, np.sum(irreducible_matrix[i], dtype = np.float32))

    for i in range(irreducible_matrix.shape[0]):
        sum = 0
        sum = np.sum(irreducible_matrix[i])
        if sum != 0:
            irreducible_matrix[i] = irreducible_matrix[i]/sum

    for i in range(len(irreducible_matrix)):
        if (np.sum(irreducible_matrix[i], dtype = np.float32) != 1.0):

            print(i, np.sum(irreducible_matrix[i], dtype = np.float32))

    count = 0
    for i in range(len(percentageMatrix_list)):
        if percentageMatrix_list[i][2] == False:
            count += 1
    print("percentageMatrix length is: ", count)

    return(percentageMatrix_list, irreducible_matrix)

    # if len(irreducible_matrix) == 26:
        # break
    # result = oldLumping.lumping(irreducible_matrix, percentageMatrix_list, False)
    # return result

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

lumpedAllInOne = reduceMatrixAllInOne(cuTrans_cpy[0])
cuTrans_cpy.shape

for i in range(cuTrans_cpy.shape[0]):
    percentageMatrix_list, irreducible_matrix = preparingMatrixForLumping(cuTrans_cpy[i])
    result = oldLumping.lumping(irreducible_matrix, percentageMatrix_list, False)
    name = "/home/netlab/Desktop/thesis/APDataML/pickles/normal_lumping_result_" + str(i) + ".pickle"
    with open(name, 'wb') as handle:
        pickle.dump(result, handle)

    # with open(name, 'rb') as handle:
    #     b = pickle.load(handle)



name = "/home/netlab/Desktop/thesis/APDataML/pickles/normal_lumping_result_0.pickle"

scores = {}
if os.path.getsize(name) > 0:
    with open(name, 'rb') as handle:
        b = pickle.Unpickler(handle)
        scores = unpickler.load()

scores

for i in range(60):
    percentageMatrix_list, irreducible_matrix = lumpingStatesOneByOne(percentageMatrix_list, irreducible_matrix)

c = 0
for i in range(len(percentageMatrix_list)):
    if percentageMatrix_list[i][2] == False:
        c += 1
        print(percentageMatrix_list[i][1], c, i)

print(percentageMatrix_list[-1][1])

reload(oldLumping)
result = oldLumping.lumping(irreducible_matrix, percentageMatrix_list, False)
print(percentageMatrix_list)
print(lumpedResult)
print(result[2])
percentages = mergePercentages(percentageMatrix_list, result[-1])
print(percentages)


name = "/home/netlab/Desktop/thesis/APDataML/pickles/lumping_result_" + str(0) + ".pickle"
with open(name, 'wb') as handle:
    pickle.dump(result, handle)

with open(name, 'rb') as handle:
    b = pickle.load(handle)

b
result
for i in range(len(percentageMatrix_list)):
    if percentageMatrix_list[i][2] == False:
        print(percentageMatrix_list[i][1])

def mergePercentages(percentageMatrix, whole_subset):
    start = 0
    counter = 0
    list_of_percentages = []
    for i in range(1, len(whole_subset) - 1):
        for j in range(start, len(percentageMatrix)):
            start += 1
            if percentageMatrix[j][2] == False:
                counter += 1
                if counter == whole_subset[i]:
                    list_of_percentages.append((percentageMatrix[j][1]))
                    break
    return list_of_percentages

percentages.append(100)

XArraysForLearning, YArraysForLearning, XArraysForTesting, YArraysForTesting, boundaries = simpleLogisticRegression(data, 48, 30, percentages)

Y = np.argmax(YArraysForLearning, 1)
Y_test = np.argmax(YArraysForTesting, 1)
print(Y.shape)
reg = "l2"
solvers = "lbfgs"
clf = LogisticRegression(penalty = reg, max_iter = 100000, random_state = 0,
                         solver = solvers , multi_class = 'multinomial')
clf.fit(XArraysForLearning, Y)
Ypred = clf.predict(XArraysForTesting)
acc = accuracy_score(Y_test, Ypred)
print(acc)
saveResults.append((x, acc))


def classifying(CU, boundaries):
    occupiedBandwidth = (CU / 255) * 100
    for i in range(len(boundaries)):
        if occupiedBandwidth <= boundaries[i]:
            return i

def simpleLogisticRegression(data, numberOfTimeIntervals, minuteSplit, boundaries):

    warnings.filterwarnings('always')
    reg = "l2"
    solvers = "lbfgs"
    clf = LogisticRegression(penalty = reg, max_iter = 100000, random_state = 0,
                             solver = solvers , multi_class = 'multinomial')
    accuracyValue = 0
    numOfElements = 0
    f1scoreValue = 0
    precisionValue = 0
    recallValue = 0
    prevRowTrain = np.inf
    prevCU = np.inf

    sampleIntervals = 6 #seconds
    minuteSplit = 30 #minutes
    numOfSamples = minuteSplit * 60 / sampleIntervals
    # numberOfDays = len(numOfDays)
    days = np.zeros(7)
    numOfThirtyMinsPerDay = np.zeros(int((24 * 60) / minuteSplit)) #in this case 48
    which6SecondsPerPeriod = np.zeros(int(minuteSplit * 60 / sampleIntervals)) #in this case 300
    prevRowTrain = np.inf
    prevCU = np.inf

    for x in range(0, 1):
        XArraysForLearning = []
        YArraysForLearning = []
        XArraysForTesting = []
        YArraysForTesting = []

        x = 0
        iterPandas = data.loc[(data["timeIndex"] == x)].copy()

        trainingDataFrame = iterPandas.iloc[:int(np.floor(0.8 * len(iterPandas)))].copy()

        trainingTransitionMatrix = processData.markovianTransitionMatrixDegree1(trainingDataFrame, 255)
        normalizedTrainTrans = processData.normalizingTransMatrix(trainingTransitionMatrix)
        normalizedTrainTransition = normalizedTrainTrans.copy()

        trainingDataFrame["cuClass"] = trainingDataFrame["CU"].apply(lambda x: classifying(x, boundaries))

        stackCounter = 0
        prevCU = 0
        for index, row in trainingDataFrame.iterrows():
            lastCU = np.zeros(len(boundaries))
            lastCU[prevCU] = 1
            which6SecondsPerPeriod[int(row["periodParticle"])] = 1
            XArray = lastCU
            XArray = np.append(XArray, which6SecondsPerPeriod)

            if stackCounter == 0:
                XArraysForLearning = XArray
            else:
                XArraysForLearning = np.vstack((XArraysForLearning, XArray))

            recentCU = np.zeros(len(boundaries))
            recentCU[row["cuClass"]] = 1

            if stackCounter == 0:
                YArraysForLearning = recentCU
                stackCounter += 1
            else:
                YArraysForLearning = np.vstack((YArraysForLearning, recentCU))

            which6SecondsPerPeriod[int(row["periodParticle"])] = 0
            prevCU = row["cuClass"]


        weights = np.random.randn(XArraysForLearning.shape[1])
        print("Testing")
        #********************LR testing********************


        testingDataFrame = iterPandas.iloc[int(np.floor(0.8 * len(iterPandas))):].copy()

        testingDataFrame["cuClass"] = testingDataFrame["CU"].apply(lambda x: classifying(x, boundaries))


        stackCounter = 0
        prevCU = 0
        for index, row in testingDataFrame.iterrows():

            lastCU = np.zeros(len(boundaries))
            lastCU[prevCU] = 1
            which6SecondsPerPeriod[int(row["periodParticle"])] = 1
            XArray = lastCU
            XArray = np.append(XArray, which6SecondsPerPeriod)

            if stackCounter == 0:
                XArraysForTesting = XArray

            else:
                XArraysForTesting = np.vstack((XArraysForTesting, XArray))
    #         print(XArraysForTesting)

            recentCU = np.zeros(len(boundaries))
            recentCU[row["cuClass"]] = 1

            if stackCounter == 0:
                YArraysForTesting = recentCU
                stackCounter += 1
            else:
                YArraysForTesting = np.vstack((YArraysForTesting, recentCU))
    #         print(YArraysForTesting)

            which6SecondsPerPeriod[int(row["periodParticle"])] = 0
            prevCU = row["cuClass"]

        print(XArraysForLearning.shape)
        print(YArraysForLearning.shape)

        return XArraysForLearning, YArraysForLearning, XArraysForTesting, YArraysForTesting, boundaries
