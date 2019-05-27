import numpy as np
import Plot as plot
import math
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
import time
import datetime
import pickle
import os
import pysal
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib as plt
from discreteMarkovChain import markovChain
import logging
from scipy import signal
import learningAlgs as classImportLA
import dataManipulation as dataMan
from itertools import permutations
import importlib
from datetime import timedelta








#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#*********************************** EXTRA FUNCTIONS *********************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************

#********We are getting the new transition matrix from "combinationEnum" function and starting to lump to get the original matrix
#********Since the lumping can be done in different ways by sectioning different parts of the matrix, we are trying all of them to get
#********the best possible lumped matrix

def sectoring(permutedArr):
#     print("aaaa")
    numOfSectorsIndex = []
    minLumpedError = math.inf
    bestLumped = [0]
    bestSectoring = []

    
    for i in range(1, len(permutedArr[0]) - 1): #we dont care about the first and the last raw and column because they are already in
        numOfSectorsIndex.append(i)
    
    for i in range(1, 5):
#         print(i)
        extraArr = []
        for subset in permutations(numOfSectorsIndex, i): #drawing line for every possible index untill 5 classes
            if sorted(subset) not in extraArr:
                extraArr.append(sorted(subset))
        
#         print(extraArr)
        for j in range(len(extraArr)):
            sectorNumber = [0]
            sectorNumber.extend(extraArr[j])
            sectorNumber.append(len(permutedArr[0]))
#             print(sectorNumber)
            sector = []
            arrangedSectors = [[] for i in range(len(sectorNumber))]
            lumpedArr = [[0 for i in range(len(extraArr[j]) + 1)] for i in range(len(extraArr[j]) + 1)]
            lumpedArrVal = [[0 for i in range(len(extraArr[j]) + 1)] for i in range(len(extraArr[j]) + 1)]
            quasiLumpedArr = [[0 for i in range(len(extraArr[j]) + 1)] for i in range(len(extraArr[j]) + 1)]
            
            for x in range(1, len(sectorNumber)):
                index = x
                arrangedSectors[x - 1].append(permutedArr[sectorNumber[x - 1] : sectorNumber[x], sectorNumber[x - 1] : sectorNumber[x]])
#                 print(permutedArr[sectorNumber[x - 1] : sectorNumber[x], sectorNumber[x - 1] : sectorNumber[x]])
                lumpedArr[x-1][x-1] = permutedArr[sectorNumber[x - 1] : sectorNumber[x], sectorNumber[x - 1] : sectorNumber[x]]
#                 print("lumped Array is: " + str(lumpedArr))
                sector.append(permutedArr[sectorNumber[x - 1] : sectorNumber[x], sectorNumber[x - 1] : sectorNumber[x]])
                index -= 1
                while index - 1 >= 0:
                    arrangedSectors[index - 1].insert(0, permutedArr[sectorNumber[index - 1] : sectorNumber[index], sectorNumber[x - 1] : sectorNumber[x]])
                    lumpedArr[index - 1][x - 1] = permutedArr[sectorNumber[index - 1] : sectorNumber[index], sectorNumber[x - 1] : sectorNumber[x]]
#                     print("lumped Array is: " + str(lumpedArr))
                    arrangedSectors[x - 1].insert(0, permutedArr[sectorNumber[x - 1] : sectorNumber[x], sectorNumber[index - 1] : sectorNumber[index]])
                    
                    lumpedArr[x - 1][index - 1] = permutedArr[sectorNumber[x - 1] : sectorNumber[x], sectorNumber[index - 1] : sectorNumber[index]]
#                     print("lumped Array is: " + str(lumpedArr))
                    sector.append(permutedArr[sectorNumber[index - 1] : sectorNumber[index], sectorNumber[x - 1] : sectorNumber[x]])
                    sector.append(permutedArr[sectorNumber[x - 1] : sectorNumber[x], sectorNumber[index - 1] : sectorNumber[index]])
                    index -= 1

#             print(sector)
#             print(arrangedSectors)
#             print("lumped Array is: " + str(lumpedArr))
            probIter  = 0
            prob = [0 for i in range(len(permutedArr))]
            hitTimes = [0 for i in range(len(permutedArr))]            
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
                    for p in range(len(lumpedArr[l][x])): #read rows
                        quasiError[p] = np.sum(lumpedArr[l][x][p])
                            
                        prob[probIter] += np.sum(lumpedArr[l][x][p])
#                         print("prob is: " + str(prob))
                        hitTimes[probIter] += len(lumpedArr[l][x][p])
                        probIter += 1
                    
#                     print(quasiError)
                    minOfMaxErrs = math.inf
                    lowErrIndex = math.inf
                    if len(quasiError) > 1:
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
                    quasiLumpedArr[l][x] = minOfMaxErrs
#                     print("error is: " + str(minOfMaxErrs))
                    
#             print("hit is " + str(quasiLumpedArr))
#             print("prob is: " + str(prob))
#             print("lumpedVal array is: " + str(lumpedArrVal))
            if np.sum(quasiLumpedArr) < minLumpedError:
                minLumpedError = np.sum(quasiLumpedArr)
                bestLumped = lumpedArrVal
                bestSectoring = sectorNumber
                
    # print("status:")
    # print(minLumpedError)
    # print(bestLumped)
    # print(bestSectoring)
    return minLumpedError, bestLumped, bestSectoring

#******************for create different sections for the file, we are enumerating the whole transition matrix*************
def combinationEnum(npArray):
    arr = []
    minErr = math.inf
    lumpedMat = []
    sector = []
    for i in range(len(npArray)):
        arr.append(i)
    for subset in permutations(arr, len(arr)):
        listIndex = list(subset)
        colEx = npArray[:,listIndex]
        colEx[listIndex]
        err, lump, sec = sectoring(colEx)
        if err < minErr:
            minErr = err
            lumpedMat = lump
            sector = sec
	  
    return minErr, lumpedMat, sector  






#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************
#********************************** MAIN OF THE PROGRAM ******************************************
#*************************************************************************************************
#*************************************************************************************************
#*************************************************************************************************

address = input("the address of the collected data files (not alligned files or CSV files): ")
#/home/Sepehr/Desktop/project/thesis/data/1node1-3-5/


#***************finding all the usefull files in the address *************
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
fileNameArr = ["500f80271400-1.txt"] #for test we are giving a ready file

LA = classImportLA.learningAlgs() #calling an object of the class

#*************checking if we have the processed version of file in our CSV or not, if not we are gonna process data otherwise just we gonna read it
pathFile = address + "/CSV/" + str(fileNameArr[0]) + ".csv"
importlib.reload(dataMan) #reload the class if it has cache (.pyc) to load the class from scratch
if os.path.isfile(pathFile) == False:
        print("we do not have processed data for file " + str(fileNameArr[0]) + " so we are making it")
        dataMan.normalDataSplitting(fileNameArr[0], 0, 0, timeInterval, address)

# print(fileNameArr[0])
# print(address)
stat, data = LA.csvChecker(fileNameArr[0], 0, address)        


print("now we have the processed data from pandas")
data["CU/255"] = data["CU"] / 255 #we add a column to our pandas table

secondsPerChunk = int(60 * int(timerInMinute)) + 1

data["timeIndex"] = -1
startIndex = 0
timeIndexVal = 1

#*******chunking data based on the time and put a value for its column
data["timeIndex"] = data["time"].apply(lambda x: math.floor(((x - data["time"][0]).seconds + ((x - data["time"][0]).days * 3600 * 24)) / secondsPerChunk))

numberOfChunks = data["timeIndex"][len(data) - 1] + 1

#*************now we want to calculate the transition matrix**************
cuTrans = np.zeros(shape=(numberOfChunks, 51, 51))
start = -1
next = -1
prevChunkVal = -1
newChunkVal = -1
firstIndexOfChunk = -1


cuTrans = np.zeros(shape=(numberOfChunks, 51, 51))
cuDifTrans = np.zeros(shape=(numberOfChunks, 51, 51))

start = -1
next = -1
prevChunkVal = -1
newChunkVal = -1
firstIndexOfChunk = -1
indexesOfChunks = data.index[data["timeIndex"] == i]
counter = 0
for index, row in data.iterrows():
    # if index % 10000 == 0:
        # print(index)
    newChunkVal = row["timeIndex"]
    if newChunkVal == prevChunkVal:
        start = next
        next = row["CU/255"] / 0.02
        if start != -1:
            cuTrans[newChunkVal, math.floor(start), math.floor(next)] += 1
        prevChunkVal = newChunkVal

    elif newChunkVal != prevChunkVal:
        if firstIndexOfChunk != -1:
            start = next
            next = data["CU/255"][firstIndexOfChunk] / 0.02
            cuTrans[prevChunkVal, math.floor(start), math.floor(next)] += 1
        firstIndexOfChunk = index
        next = row["CU/255"] / 0.02
        prevChunkVal = newChunkVal
indexesOfChunks = data.index[data["timeIndex"] == i]

#this is only for this dataSet, so it is for "test"
data["time"] = data["time"].apply(lambda x: x + timedelta(hours = -7)) #our data is in UTC so we should make it to Moutain time

#**********we wanna remove those 30 minutes with less than 280 CU values *******************
removableIndices = []
for i in range(cuTrans.shape[0]):
    if (np.sum(cuTrans[i]) < 280):
        removableIndices.append(i)
        

#**********we gonna normalize our data because the markovian transition matrix should sum up to 1 at each row *************

print("normalizing")
ans = np.zeros(shape=(cuTrans.shape[0], 51))
for x in range(cuTrans.shape[0]):
    for i in range(cuTrans.shape[1]):
        sum = 0
        for j in range(cuTrans.shape[2]):
            sum += cuTrans[x][i][j]
        if sum != 0:
            cuTrans[x][i] = cuTrans[x][i]/sum

	  
#***********creating irreducable transition matrix, so we are remove the rows and columns which are summing up to 0 **********
cuTrans_cpy = cuTrans
rowArg = []
colArg = []
y = 0
z = 0
x = 0
errorsArr = []
lumpedArr = []
sectorsArr = []
remainedClasses = []

#*********we define all the possible classes***********
for i in range(len(cuTrans_cpy[0])):
    remainedClasses.append(i)
    
    
for x in range(10):
    rowArg = []
    colArg = []
# if x == 0:
    for i in range(len(cuTrans_cpy[x])):
        if np.sum(cuTrans_cpy[x][i]) == 0:
            rowArg.append(i)
        if np.sum((cuTrans_cpy[x].T)[i] == 0):
            colArg.append(i)
            
    deleteList = list(set(rowArg) & set(colArg))
    sizeOfNewArr = len(deleteList)
#     print(deleteList)
    newArr = np.zeros(shape=(51-sizeOfNewArr, 51-sizeOfNewArr))
    
#     print(x)
#     print(cuTrans_cpy[x].shape)
    z = 0
    y = 0
    for i in range(len(cuTrans_cpy[x])):
        if i not in deleteList: 
            for j in range(51):
                if j not in deleteList:
                    newArr[z][y] = cuTrans_cpy[x][i][j]
#                     print(i, j)
                    y += 1
            y = 0
            z += 1
#     print(newArr.shape)
    
    
    
#     print(remainedClasses)
    uncommonList = list(set(remainedClasses) - set(deleteList))
    
    classifierArr = []
    extraVar = math.inf
    prevIndex = 0
    for i in range(len(uncommonList)):
        if i == 0:
            classifierArr.append([])
            classifierArr[i].append(uncommonList[i])
            extraVar = uncommonList[i]
            prevIndex = i
        elif extraVar + 1 == uncommonList[i]:
            classifierArr[prevIndex].append(uncommonList[i])
            extraVar += 1
        else:
            classifierArr.append([])
            classifierArr[prevIndex + 1].append(uncommonList[i])
            prevIndex += 1
            extraVar = uncommonList[i]
    
    print(x)
    print(classifierArr)
#     print(len(classifierArr))
#     print(uncommonList)
#     print(newArr)
#     arrNumpy = [[0], [7, 8, 9], [10, 11, 12, 13]]
#     assert (len(uncommonList) == newArr.shape[0]), "something is wrong when eliminating"
    print(newArr.shape)
#     combinationEnum(newArr, classifierArr)
    err, lump, sec = combinationEnum(newArr, classifierArr)
    errorsArr.append(err)
    lumpedArr.append(lump)
    sectorsArr.append(sec)
    with open("lumpedError", "wb") as file:
        pickle.dump(errorsArr, file)
        
    with open("lumpedArr", "wb") as file:
        pickle.dump(lumpedArr, file)
        
    with open("lumpedSec", "wb") as file:
        pickle.dump(sectorsArr, file)

  
    






#****************Extra Works and Testing************************
# #covariance of each two points for CU
# data["CU_dif"] = data.apply(lambda x: 0 if x["col1"] == 0 else (data["CU"][x["col1"]] - data["CU"][x["col1"] - 1]), axis = 1)
# #weights for CUs
# data["CU_Weighted_Dif"] = -1
# data["CU_Weighted_Dif"] = data["CU_dif"].apply(lambda x: math.fabs(x)*(1 / (1 - (math.fabs(x) / 225))))

# data_cpy = data #copying data
# arr = []
# chunkSize = np.zeros(shape=(numberOfChunks, 1))

# def sizeCounter(x):
    # chunkSize[x] += 1
    
# data_cpy["timeIndex"].apply(sizeCounter) #counting number of elements at each chunk

# #removing all the chunks with size of less than 280 for 30 minutes (because we probably losing some data during its collection)
# removableIndices = (np.argwhere(chunkSize < 280).T)[0]
# removableIndices
# a = []
# data_cpy.apply(lambda x: a.append(x["col1"]) if x["timeIndex"] in removableIndices else None, axis = 1)

# data_cpy.drop(a, axis = 0, inplace=True)
# new_cuTrans = np.delete(cuTrans, removableIndices, axis = 0)

