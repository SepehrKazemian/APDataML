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
warnings.filterwarnings('always')


def sectoring(permutedArr, sectorNumber):
#     print("aaaa")
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
    for x in range(1, len(sectorNumber)):
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
            
#     print(indicesArr)
#     print(lumpedArr)

    numOfRows = 0
    numOfRows = len(permutedArr)
    
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
                errorArr[rowVal][x] = 0
                semiLumpedArr[rowVal][x] = np.sum(lumpedArr[l][x][0])
            
            else:
                minVal = math.inf
                for p in range(len(lumpedArr[l][x])):
                    if np.sum(lumpedArr[l][x][p]) < minVal:
                        minVal = np.sum(lumpedArr[l][x][p])
                
                for p in range(len(lumpedArr[l][x])):
                    rowVal = 0
                    for rows in range(0, l):
                        rowVal += len(lumpedArr[rows][0])
                    rowVal += p
                    
                    errorArr[rowVal][x] = np.sum(lumpedArr[l][x][p]) - minVal
                    semiLumpedArr[rowVal][x] = np.sum(lumpedArr[l][x][p])
                    
                    
            for p in range(len(lumpedArr[l][x])): #read rows              
                
                quasiError[p] = np.sum(lumpedArr[l][x][p])


                prob[probIter] += np.sum(lumpedArr[l][x][p])
#                         print("prob is: " + str(prob))
                hitTimes[probIter] += len(lumpedArr[l][x][p])
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
            lumpedArrVal[l][x] = np.sum(lumpedArr[l][x][lowErrIndex])
#             print(minOfMaxErrs)
            quasiLumpedArr[l][x] = minOfMaxErrs
    
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

    return maxError, lumpedArrVal, sectorNumber



def combinationEnum(npArray):
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
    for i in range(1, len(npArray)): #we dont care about the first and the last raw and column because they are already in
        numOfSectorsIndex.append(i)
#             print("and here")
#             print(numOfSectorsIndex)
    listOfSectors = [] #we use it to prevent repitative sectors to operate
#             print(len(classifier))
    for i in range(2, 5):
#                 print("i is: " + str(i) )
#                 print("num of sector indices are: " + str(numOfSectorsIndex))
        extraArr = []
        for subset in permutations(numOfSectorsIndex, i): #drawing line for every possible index untill 5 classes
            if sorted(list(subset)) not in extraArr:
#                     print("subset list is: " + str(list(subset)))
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
#                 print("sector number is: " + str(sectorNumber))
#                 print("list of sectors are: " + str(listOfSectors))
                if sectorNumber not in listOfSectors:
#                             print("injam")
                    listOfSectors.append(sectorNumber)
        
    func = partial(sectoring, colEx)
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    result = p.map(func, listOfSectors)
    p.close()
    p.join()
    minError = np.inf
    bestLumped = []
    resultArr = []
    bestClustering = []
#     print(result)
    for i in range(len(result)):
        if result[i][0] <= minError:
            minError = result[i][0]
            resultArr = result[i]
#             print(result[i][0], result[i][1], result[i][2] )
#         if result[i][0] < minError and result[i][0] != 0:
#             minError = result[i][0]
#             bestLumped = result[i][1]
#             bestClustering = result[i][2]
#         print(minError)
#         print(bestLumped)
#         print(bestClustering)
#     return minError, bestLumped, bestClustering
    return resultArr




def lumping(numberOfStates, transitionMatrix):
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

    #*****for finding the classes that are remained untouched in the process of reducing the transition matrix *****
    for i in range(len(cuTrans_cpy[0])):
        remainedClasses.append(i)

    counter = 0

    for x in range(numberOfStates):
        print(x)
        rowArg = []
        colArg = []

        #*************removing zeros from columns and rows (matrix reduction) ****************
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
                        y += 1
                y = 0
                z += 1

        uncommonList = list(set(remainedClasses) - set(deleteList))
        print(uncommonList)

        classifierArr = []
        extraVar = math.inf
        prevIndex = 0

        result = combinationEnum(newArr)
        print(result)
        results.append(result)
    return results


def lumpingOneState(transitionMatrix, state):
    x = state
    cuTrans_cpy = transitionMatrix
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
#     uncommonList = list(set(remainedClasses) - set(deleteList))
#     listOfMerge = [6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42]
    
    classifierArr = []
    extraVar = math.inf
    prevIndex = 0
    
#     print(index, checkList[index], newArr.shape)
#     uncommonList = [0, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 44]
    result = combinationEnum(newArr)
    return result

def clustering(CU, bounds): #automatic clustering based on the 30 minutes time in hand
    x = CU
    for i in range(1, len(bounds)):
        if x < bounds[i] and x >= bounds[i-1]:
            return (i - 1)
#     if x < 256 and x >= bounds[i]:
    return (len(bounds) - 1)
