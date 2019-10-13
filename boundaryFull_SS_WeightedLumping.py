from itertools import permutations
import itertools
from functools import partial
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
# import pysal
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib as plt
# from discreteMarkovChain import markovChain
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
import pysal
import warnings
from itertools import permutations
from itertools import combinations
import pysal
import decimal
import itertools
from functools import partial
warnings.filterwarnings('always')


def sectoring(permutedArr, ss_weight, sectorNumber):
#     print(ss_weight)
#     print("aaaa")
    import decimal
    deci = decimal.Decimal
    numOfSectorsIndex = []
    minLumpedError = math.inf
    bestLumped = [0]
    bestSectoring = []
    numbOfChunks = len(sectorNumber) - 1
#     print("here we look")
#     print(sectorNumber)

    
    
    sector = []
    indices = []
    lumpedArr = [[0 for i in range(numbOfChunks)] for i in range(numbOfChunks)]
    indicesArr = [[0 for i in range(numbOfChunks)] for i in range(numbOfChunks)]
    lumpedArrVal = [[0 for i in range(numbOfChunks)] for i in range(numbOfChunks)]
    quasiLumpedArr = [[0 for i in range(numbOfChunks)] for i in range(numbOfChunks)]
    fullLumpedArrIndices = [[0 for i in range(numbOfChunks)] for i in range(numbOfChunks)]
#     print(lumpedArr)
#     [[0 for i in range(len(permutedArr))] for i in range(len(permutedArr))]

#     print("initialized lumped Array is: " + str(lumpedArr))
    for x in range(len(sectorNumber)):
        index = x
        lumpedArr[x-1][x-1] = permutedArr[sectorNumber[x - 1] : sectorNumber[x], sectorNumber[x - 1] : sectorNumber[x]]
#         print("just this")
        a = np.arange(sectorNumber[x - 1], sectorNumber[x])
        b = np.arange(sectorNumber[x - 1], sectorNumber[x])
        A,B = np.meshgrid(a,b)
        AB=np.array([A.flatten(),B.flatten()]).T
        indicesArr[x - 1][x - 1] = AB
        sector.append(permutedArr[sectorNumber[x - 1] : sectorNumber[x], sectorNumber[x - 1] : sectorNumber[x]])
        index -= 1
        while index - 1 >= 0:
            lumpedArr[index - 1][x - 1] = permutedArr[sectorNumber[index - 1] : sectorNumber[index], sectorNumber[x - 1] : sectorNumber[x]]
#             print("just this")
            a = np.arange(sectorNumber[index - 1], sectorNumber[index])
            b = np.arange(sectorNumber[x - 1], sectorNumber[x])
            A,B = np.meshgrid(a,b)
            AB=np.array([A.flatten(),B.flatten()]).T
            indicesArr[index - 1][x - 1] = AB
            indices.append(AB)
#             fullLumpedArr[sectorNumber[index - 1] : sectorNumber[index], sectorNumber[x - 1] : sectorNumber[x]] = permutedArr[sectorNumber[index - 1] : sectorNumber[index], sectorNumber[x - 1] : sectorNumber[x]]
            #                     print("lumped Array is: " + str(lumpedArr))
#             print(fullLumpedArr)
            lumpedArr[x - 1][index - 1] = permutedArr[sectorNumber[x - 1] : sectorNumber[x], sectorNumber[index - 1] : sectorNumber[index]]
#             print("just this")
            a = np.arange(sectorNumber[x - 1], sectorNumber[x])
            b = np.arange(sectorNumber[index - 1], sectorNumber[index])
            A,B = np.meshgrid(a,b)
            AB=np.array([A.flatten(),B.flatten()]).T
            indicesArr[x - 1][index - 1] = AB
            indices.append(AB)
#             fullLumpedArr[sectorNumber[x - 1] : sectorNumber[x], sectorNumber[index - 1] : sectorNumber[index]] = permutedArr[sectorNumber[x - 1] : sectorNumber[x], sectorNumber[index - 1] : sectorNumber[index]]
#             print(fullLumpedArr)
#                     print("lumped Array is: " + str(lumpedArr))
            sector.append(permutedArr[sectorNumber[index - 1] : sectorNumber[index], sectorNumber[x - 1] : sectorNumber[x]])
            sector.append(permutedArr[sectorNumber[x - 1] : sectorNumber[x], sectorNumber[index - 1] : sectorNumber[index]])
            index -= 1
            
#     print(sectorNumber)
#     print("indices are:")
#     print(indicesArr)
#     print("lumped array is:")
#     print(lumpedArr)
    

    numOfRows = 0
    numOfRows = len(permutedArr)
    
    quasiError = np.zeros(shape=(sectorNumber[-1], sectorNumber[-1]))
#     print("quasiError is:")
#     print(quasiError)
    errorArr = [[0 for i in range(len(lumpedArr[0]))] for j in range(numOfRows)]
    semiLumpedArr = [[0 for i in range(len(lumpedArr[0]))] for j in range(numOfRows)]
    
#     print("error dimensions")
#     print(numOfRows)
#     print(len(lumpedArr[0]))
#     print(errorArr)
            
#     time.sleep(5)
#     print("we are here")
    probIter  = 0
    prob = [0 for i in range(len(permutedArr))]
    hitTimes = [0 for i in range(len(permutedArr))]
#     print("lumped arr is: " + str(lumpedArr))
#     print(len(lumpedArr))
    for l in range(len(lumpedArr)): #gets the numpy arrays
        sum = 0
        for x in range(len(lumpedArr[l])): #gets each numpy array
            probIter = 0
            for q in range(len(hitTimes)): #check the correct place of full matrix
                    if hitTimes[q] != len(permutedArr):
#                                 print("hit is " + str(hitTimes))
                        probIter = q
                        break
            quasiError = [0 for i in range(len(lumpedArr[l][x]))]
            
            if len(lumpedArr[l][x]) == 1:
                rowVal = 0
                for rows in range(0, l):
                    rowVal += len(lumpedArr[rows][0])
#                 print("rowVal is: " + str(rowVal))
                errorArr[rowVal][x] = 0
#                 print("errorArr is: " + str(errorArr))
                semiLumpedArr[rowVal][x] = np.float16(np.sum(lumpedArr[l][x][0]))
#                 print("semiLumpedArr is: " + str(semiLumpedArr))
            
            else:
                minVal = math.inf
                for p in range(len(lumpedArr[l][x])):
                    if np.float16(np.sum(lumpedArr[l][x][p])) < minVal:
                        minVal = np.float16(np.sum(lumpedArr[l][x][p]))
#                 print("minVal is")
#                 print(minVal)
                
                for p in range(len(lumpedArr[l][x])):
                    rowVal = 0
                    for rows in range(0, l):
                        rowVal += len(lumpedArr[rows][0])
                    rowVal += p
                    
#                     print("rowVal is: " + str(rowVal))
#                     print(p)
#                     print(np.float16(np.sum(lumpedArr[l][x][p])))
#                     print(minVal)
#                     print(np.float16(np.sum(lumpedArr[l][x][p])) - minVal)
#                     print("sector number of x is: " + str(sectorNumber[x]))
#                     print(sectorNumber)
                    errorArr[rowVal][x] = (np.float16(np.sum(lumpedArr[l][x][p])) - minVal) * (ss_weight[x])
#                     print("errorArr is: " + str(errorArr))
                    semiLumpedArr[rowVal][x] = np.float16(np.sum(lumpedArr[l][x][p]))
#                     print("semiLumpedArr is: " + str(semiLumpedArr))
                    
                    
            for p in range(len(lumpedArr[l][x])): #read rows              
                
                quasiError[p] = np.float16(np.sum(lumpedArr[l][x][p]))
#                 print(quasiError)

                prob[probIter] += np.float16(np.sum(lumpedArr[l][x][p]))
#                 print("prob is: " + str(prob))
                
                hitTimes[probIter] += len(lumpedArr[l][x][p])
#                 print(hitTimes)
                
                probIter += 1
#             time.sleep(10)

#                     print(quasiError)
            minOfMaxErrs = math.inf
            lowErrIndex = math.inf
            if len(quasiError) > 1:
#                 print("quasi Error is: " + str(quasiError))
                for first in range(len(quasiError)):
                    maxErr = 0
                    for sec in range(len(quasiError)):
                        if math.fabs(quasiError[first] - quasiError[sec]) > maxErr:
                            maxErr = math.fabs(quasiError[first] - quasiError[sec])
                    if maxErr < minOfMaxErrs:
                        lowErrIndex = first
                        minOfMaxErrs = maxErr
            else:
                lowErrIndex = 0
                minOfMaxErrs = 0
            lumpedArrVal[l][x] = np.float16(np.sum(lumpedArr[l][x][lowErrIndex]))
#             print(minOfMaxErrs)
            quasiLumpedArr[l][x] = minOfMaxErrs
#             print(quasiLumpedArr)
    
#     print("arrays:")
#     print(errorArr)
#     print(semiLumpedArr)
    
#     print("error is: ")
    maxError = - math.inf
    for i in range(len(errorArr)):
        if np.sum(errorArr[i]) > maxError:
            maxError = np.sum(errorArr[i])
#         print(np.sum(semiLumpedArr[i]))
#     print(maxError)

    return np.float16(maxError), lumpedArrVal, sectorNumber



def combinationEnum(npArray, ss_weight):
    arr = []
    arrayResults = []
    minErr = math.inf
    lumpedMat = []
    sector = []
    permutationSave = []
    permuteMatrix = []
    indexList = []
    numOfSectorsIndex = []
    error = math.inf
    lumpedMatrix = []
    sectoringModel = []
    permutationModel = []
    
    
#     for i in range(len(classifier)):
#         arr.append(i)
    
#     p = 0
#     for x in range(len(classifierArr)):
#         indexList.append([])
#         for j in range(len(classifierArr[x])):
#             indexList[x].append(p)
#             p += 1
 

    matrixPlace = []
    newMatrixPlace = []
#     listIndex = list(subset)
    colEx = npArray


#             time.sleep(5)
    numOfSectorsIndex = []
#             print("even here")
    for i in range(1, len(npArray) - 1): #we dont care about the first and the last raw and column because they are already in
        numOfSectorsIndex.append(i)
#             print("and here")
#             print(numOfSectorsIndex)
    listOfSectors = [] #we use it to prevent repitative sectors to operate
#             print(len(classifier))
    for i in range(2, 5):
#                 print("i is: " + str(i) )
#                 print("num of sector indices are: " + str(numOfSectorsIndex))
        extraArr = []
        for subset in combinations(numOfSectorsIndex, i): #drawing line for every possible index untill 5 classes
#                     print("i is: " + str(i) )
            if sorted(list(subset)) not in extraArr:
#                 print("subset list is: " + str(list(subset)))
#                         print("arrays of subset is: " + str(extraArr))
                extraArr.append(sorted(list(subset)))
#                     print("extraArr is: " + str(extraArr))
                sectorNumber = [0]
                for j in range(len(sorted(list(subset)))):
#                         print(extraArr[j])
#                         print("extraArr j is: " + str(extraArr[j]))                        
#                         for l in range(len(extraArr[j])):
                    sectorNumber.append(sorted(list(subset))[j])
#                         sectorNumber.extend(extraArr[j])
                sectorNumber.append(len(colEx))
#                 print(sectorNumber)
#                     print("sector number is: " + str(sectorNumber))
#                     print("list of sectors are: " + str(listOfSectors))
                if sectorNumber not in listOfSectors:
#                             print("injam")
#                     sectoring(colEx, sectorNumber)
#                     time.sleep(5)
                    listOfSectors.append(sectorNumber)
        
    func = partial(sectoring, colEx, ss_weight)
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    result = p.map(func, listOfSectors)
    p.close()
    p.join()
    minError = np.inf
    bestLumped = []
    bestClustering = []
    resultArr = []
    for i in range(len(result)):
#         print(result[i])
#         print("\n")
#         max([j-i for i, j in zip(result[i][2][:-1], result[i][2][1:])]) <= 14
        bool25Percent = True
        for x in range(1, len(result[i][2])):
            if (result[i][2][x] - result[i][2][x - 1] >= 7) or (result[i][2][x] - result[i][2][x - 1] == 1):
#                 print(result[i][2])
                bool25Percent = False
                break

        minVal = np.inf
        if result[i][0] <= minError and bool25Percent == True:
            minError = result[i][0]
            resultArr = result[i]
#             arr.append(result[i])
    arr = resultArr
#             print(result[i][0], result[i][1], result[i][2] )
#         if result[i][0] < minError and result[i][0] != 0:
#             minError = result[i][0]
#             bestLumped = result[i][1]
#             bestClustering = result[i][2]
#         print(minError)
#         print(bestLumped)
#         print(bestClustering)
#     return minError, bestLumped, bestClustering
    return arr


def boundaryFull_SS_WeightedLumping(transitionMatrix, numberOfStates):
    numberOfChunks = numberOfStates
    cuTrans_cpy = transitionMatrix
    rowArg = []
    colArg = []
    y = 0
    z = 0
    x = 28
    errorsArr = []
    lumpedArr = []
    sectorsArr = []
    remainedClasses = []
    results = []

    numberOfChunks = max(data["timeIndex"]) + 1

    #*****for finding the classes that are remained untouched in the process of reducing the transition matrix *****
    for i in range(len(cuTrans_cpy[0])):
        remainedClasses.append(i)

    # savedStat = [0 for i in range(len(allHours))]

    counter = 0
    for x in range(numberOfChunks):
        print(x)
    #     x = 21
        rowArg = []
        colArg = []

        for i in range(len(cuTrans_cpy[x])):
            if np.sum(cuTrans_cpy[x][i]) == 0:
                rowArg.append(i)
            if np.sum((cuTrans_cpy[x].T)[i] == 0):
                colArg.append(i)

        deleteList = list(set(rowArg) & set(colArg))
        sizeOfNewArr = len(deleteList)
    #     print(deleteList)
        newArr = np.zeros(shape=(26-sizeOfNewArr, 26-sizeOfNewArr))

    #     print(x)
    #     print(cuTrans_cpy[x].shape)
        z = 0
        y = 0
        for i in range(len(cuTrans_cpy[x])):
            if i not in deleteList: 
                for j in range(26):
                    if j not in deleteList:
                        newArr[z][y] = cuTrans_cpy[x][i][j]
    #                     print(i, j)
                        y += 1
                y = 0
                z += 1
    #     print("the transition matrix is: " + str(newArr))
    #     print("shape of matrix is: " + str(newArr.shape))


    #     print(deleteList)
        uncommonList = list(set(remainedClasses) - set(deleteList))
        print(uncommonList)
    #     listOfMerge = [6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42]

        classifierArr = []
        extraVar = math.inf
        prevIndex = 0

    #     print(index, checkList[index], newArr.shape)
    #     uncommonList = [0, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 44]

    #     newArr = np.array([[0.2, 0.28, 0.1, 0.21, 0.11, 0.1],[0.29, 0.1, 0.2, 0.05, 0.31, 0.05],[0.15, 0.2, 0.24, 0.12, 0.2, 0.09],[0.27, 0.18, 0.22, 0.18, 0.01, 0.14], [0.18, 0.2, 0.3, 0.31, 0.01, 0], [0, 0.25, 0.43, 0.07, 0.08, 0.17]])
    #     newArr = np.array([[0.2, 0.28, 0.1],[0.29, 0.1, 0.2],[0.15, 0.2, 0.24]])
        print("steadyState")
        ss = abs(pysal.spatial_dynamics.ergodic.steady_state(newArr))
        ss_as_weight = [np.float64(i) for i in ss]
    #     print(ss_as_weight)


        result = combinationEnum(newArr, ss_as_weight)
    #     print(result)
    #     [print(result[i][2]) for i in range(len(result))]


        extraList = []
        result = list(result)
        start_point = -1
        end_point = -1
        extraList_index = -1
        for j in range(len(result[2])):
            if start_point == -1 and end_point == -1:
                end_point = 0
                continue
            else:
                start_point = end_point
                end_point = result[2][j]
                extraList.append([])
                extraList_index += 1
                while start_point <= end_point - 1:
                    extraList[extraList_index].append(uncommonList[start_point])
                    start_point += 1
        result[2] = extraList
        result = tuple(result)

        #filling the gaps values in a result list of classes
        extraList = []
        result = list(result)
        result[2][-1][-1] = 25 #last element should be 25 since we started from 0 and have 26 classes in total

        lastElemCluster = -1
        firstElemCluster = -1
        for j in range(len(result[2])):
            firstElemCluster = result[2][j][0]
            if lastElemCluster != -1:             
                while lastElemCluster + 1 < firstElemCluster:
                    result[2][j - 1].append(lastElemCluster + 1)
                    lastElemCluster += 1 

            if len(result[2][j]) != 1:
                first = -1
                second = -1
                for x in range(len(result[2][j])):
                    if first == -1 and second == -1:
                        second = result[2][j][x]
                        continue
                    else:
                        first = second
                        second = result[2][j][x]
                        while first + 1 < second:
                            result[2][j].append(first + 1)
                            result[2][j].sort()
                            first += 1
            lastElemCluster = result[2][j][-1]
        result = tuple(result)
        print(result)

    #     print("aaaaaaaaaaaaaaaaaaaaaaaaa\n\n\n\n")
    #     print("a")
    #     [print(result[i][0], result[i][2]) for i in range(len(result))]
    #     print(np.int8(result))
    #     result = [ print(str(elem) + "aaaaa\n") for elem in result ]

    #     print(decimal.Decimal(result))
        results.append(result)
    return results


def lumpingOneState(transitionMatrix, state):
    cuTrans_cpy = transitionMatrix
    remainedClasses = []
    for i in range(len(cuTrans_cpy[0])):
        remainedClasses.append(i)
    x = state
    rowArg = []
    colArg = []
    for i in range(len(cuTrans_cpy[x])):
        if np.sum(cuTrans_cpy[x][i]) == 0:
            rowArg.append(i)
        if np.sum((cuTrans_cpy[x].T)[i] == 0):
            colArg.append(i)

    deleteList = list(set(rowArg) & set(colArg))
    sizeOfNewArr = len(deleteList)
#     print(deleteList)
    newArr = np.zeros(shape=(26-sizeOfNewArr, 26-sizeOfNewArr))

#     print(x)
#     print(cuTrans_cpy[x].shape)
    z = 0
    y = 0
    for i in range(len(cuTrans_cpy[x])):
        if i not in deleteList: 
            for j in range(26):
                if j not in deleteList:
                    newArr[z][y] = cuTrans_cpy[x][i][j]
#                     print(i, j)
                    y += 1
            y = 0
            z += 1
#     print("the transition matrix is: " + str(newArr))
#     print("shape of matrix is: " + str(newArr.shape))


#     print(deleteList)
    uncommonList = list(set(remainedClasses) - set(deleteList))
    print(uncommonList)
#     listOfMerge = [6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42]

    classifierArr = []
    extraVar = math.inf
    prevIndex = 0

#     print(index, checkList[index], newArr.shape)
#     uncommonList = [0, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 44]

#     newArr = np.array([[0.2, 0.28, 0.1, 0.21, 0.11, 0.1],[0.29, 0.1, 0.2, 0.05, 0.31, 0.05],[0.15, 0.2, 0.24, 0.12, 0.2, 0.09],[0.27, 0.18, 0.22, 0.18, 0.01, 0.14], [0.18, 0.2, 0.3, 0.31, 0.01, 0], [0, 0.25, 0.43, 0.07, 0.08, 0.17]])
#     newArr = np.array([[0.2, 0.28, 0.1],[0.29, 0.1, 0.2],[0.15, 0.2, 0.24]])
    print("steadyState")
    ss = abs(pysal.spatial_dynamics.ergodic.steady_state(newArr))
    ss_as_weight = [np.float64(i) for i in ss]
#     print(ss_as_weight)


    result = combinationEnum(newArr, ss_as_weight)
    print(result)
    return result
