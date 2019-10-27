# %% codecell
from itertools import permutations
from itertools import combinations
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
# <codecell>

# %% codecell
# lumping()
def sectoring(permutedArr, sectorNumber):
    numbOfChunks = len(sectorNumber) - 1
    # print("here we look")
    # print(sectorNumber)

    for i in range(len(sectorNumber)):
        sectorNumber[i] -= 1

    disintegrated_list = [[0 for i in range(numbOfChunks)] for i in range(numbOfChunks)]


    splitted_matrix = []
    indices = []


    #************************SPLITTING PERMUTED ARRAY ************************
    #it starts from 1 since the permuted Array has an extra values of 0 and length of Array
    #first split
    for i in range(sectorNumber[-1]):
        vectorIndices = np.array([])
        for j in range(sectorNumber[-1]):
            vectorIndices= np.append(vectorIndices, (str(i) + str(j)))
        if i == 0:
            split_indices = vectorIndices
        else:
            split_indices = np.vstack((split_indices, vectorIndices))

    #OUTPUT EXAMPLE:
    # [['00' '01' '02' '03' '04' '05']
    #  ['10' '11' '12' '13' '14' '15']
    #  ['20' '21' '22' '23' '24' '25']
    #  ['30' '31' '32' '33' '34' '35']
    #  ['40' '41' '42' '43' '44' '45']
    #  ['50' '51' '52' '53' '54' '55']]


    horizental_indices_split = np.split(split_indices[:-1], sectorNumber[1:])
    horizental_array_split = np.split(permutedArr, sectorNumber[1:])


    for x in range(len(horizental_array_split) - 1):
        splitted_matrix.append(np.split(horizental_array_split[x], sectorNumber[1:], axis = 1)[:-1])

    for x in range(len(horizental_indices_split) - 1):
        indices.append(np.split(horizental_indices_split[x], sectorNumber[1:], axis = 1)[:-1])

    #OUTPUT EXAMPLE for 1,2,6 split:
    # [array([['00']], dtype='<U32'), array([['01']], dtype='<U32'), array([['02', '03', '04', '05']], dtype='<U32'), array([], shape=(1, 0), dtype='<U32')]
    # [array([['10']], dtype='<U32'), array([['11']], dtype='<U32'), array([['12', '13', '14', '15']], dtype='<U32'), array([], shape=(1, 0), dtype='<U32')]
    # [array([['20'],
    #        ['30'],
    #        ['40']], dtype='<U32'), array([['21'],
    #        ['31'],
    #        ['41']], dtype='<U32'), array([['22', '23', '24', '25'],
    #        ['32', '33', '34', '35'],
    #        ['42', '43', '44', '45']], dtype='<U32'), array([], shape=(3, 0), dtype='<U32')]


    #************************SPLITTING PERMUTED ARRAY DONE**********************


    #************************CALCULATING LUMPABILITY ERROR *********************
    quasi_error_array = np.zeros(shape=(permutedArr.shape[0], len(splitted_matrix)), dtype = np.float16)
    lumped_array = np.zeros(shape = (len(splitted_matrix), len(splitted_matrix)))
    for col in range(len(splitted_matrix)):
        quasi_row_counter = 0
        for row in range(len(splitted_matrix)):
            if splitted_matrix[row][col].shape[0] == 1:
                lumped_array[row][col] = np.sum(splitted_matrix[row][col])

            elif splitted_matrix[row][col].shape[0] != 1:
                # print(splitted_matrix[row][col])
                minValue = np.min(np.sum(splitted_matrix[row][col], axis = 1))
                lumped_array[row][col] = np.min(np.sum(splitted_matrix[row][col], axis = 1))
                for index in range(splitted_matrix[row][col].shape[0]):
                    quasi_error_array[quasi_row_counter + index, col] = np.sum(splitted_matrix[row][col][index]) - minValue
                quasi_row_counter += splitted_matrix[row][col].shape[0]

    decoupling_degree = 0
    for i in range(quasi_error_array.shape[0]):
        if np.sum(quasi_error_array[i]) > decoupling_degree:
            decoupling_degree = np.sum(quasi_error_array[i])

    error = 0
    for i in range(quasi_error_array.shape[0]):
        error += np.sum(quasi_error_array[i])
    #
    # print("quasi_error_array is: ", quasi_error_array)
    # print("decoupling_degree is: ", decoupling_degree)
    # print("error is: ", error)
    # print("lumped_array is: ", lumped_array)
    # print("sectorNumber is: ", sectorNumber)
    # print("\n\n")

    #**********************CALCULATING LUMPABILITY ERROR DONE ******************

    # print(decoupling_degree, error, lumped_array, sectorNumber)

    return decoupling_degree, error, lumped_array, sectorNumber



def combinationEnum(reduced_transition_matrix, percentageMatrix, one_by_one_clustering):
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

    matrixPlace = []
    newMatrixPlace = []

    range_iteration = []
    number_of_lumped_clusters = 0
    if one_by_one_clustering == True:
        number_of_lumped_clusters = reduced_transition_matrix.shape[0] - 2
        range_iteration = [number_of_lumped_clusters, number_of_lumped_clusters + 1]

    else:
        range_iteration = [5, 6]

    colEx = reduced_transition_matrix.copy()


    numOfSectorsIndex = []
    for i in range(1, len(reduced_transition_matrix)): #we dont care about the first and the last raw and column because they are already in
        numOfSectorsIndex.append(i)

    listOfSectors = [] #we use it to prevent repitative sectors to operate

    listOfSectors = combination(5, len(reduced_transition_matrix), percentageMatrix)

    # for i in range(range_iteration[0], range_iteration[1]):
    #     print(i)
    #     extraArr = []
    #     counter = 0
    #
    #     print("I am going into combination function")
    #
    #     for i in range()
    #
    #
    #     for subset in combinations(numOfSectorsIndex, i): #drawing line for every possible index untill 5 classes
    #         if (list(subset) not in extraArr):
    #             if (check_limitation(list(subset), len(colEx), percentageMatrix, one_by_one_clustering)
    #                 == True):
    #                 extraArr.append(list(subset))
    #                 sectorNumber = [0]
    #                 for j in range(len(list(subset))):
    #
    #                     sectorNumber.append(list(subset)[j])
    # #                         sectorNumber.extend(extraArr[j])
    #                 sectorNumber.append(len(colEx))
    #                 if sectorNumber not in listOfSectors:
    # #                             print("injam")
    #                     # result = sectoring(colEx, sectorNumber)
    #                     # print("resutl from sectoring is: ", result)
    #                     listOfSectors.append(sectorNumber)
    #
    #             elif (boundaries(subset, len(colEx), percentageMatrix) == True):
    #                 counter += 1
    #                 if counter % 100 == 0:
    #                     print(subset)
    #                 sectorNumber = [0]
    #
    #                 extraArr.append(list(subset))
    #                 for j in range(len(list(subset))):
    #
    #                     sectorNumber.append(list(subset)[j])
    # #                         sectorNumber.extend(extraArr[j])
    #                 sectorNumber.append(len(colEx))
    #                 if sectorNumber not in listOfSectors:
    # #                             print("injam")
    #                     # result = sectoring(colEx, sectorNumber)
    #                     # print("resutl from sectoring is: ", result)
    #                     listOfSectors.append(sectorNumber)

    print("we have these many sectors to check: ", len(listOfSectors))
    # print(listOfSectors)

    # for i in range(len(listOfSectors)):
    #     print(i)



    # print(listOfSectors[8])
    # result = sectoring(colEx, listOfSectors[8][0:-1])
    #
    # print("here")
    func = partial(sectoring, colEx)
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    lumping_results = p.map(func, listOfSectors)
    p.close()
    p.join()
    min_degree = np.inf
    min_error = np.inf
    best_lumped = []
    least_lumped_degree_array = []
    best_lumped_array = []
    best_sectors = []

    return lumping_results



    # result contains: decoupling_degree, error, lumped_array, sectorNumber

    #finding the least degree of coupling in our matrices
    # for i in range(len(lumping_results)):
    #     # print(lumping_results[i][0], lumping_results[i][1])
    #     if lumping_results[i][0] < min_degree:
    #         min_degree = lumping_results[i][0]
    #         least_lumped_degree_array = []
    #         least_lumped_degree_array.append(lumping_results[i])
    #
    #     elif lumping_results[i][0] == min_degree:
    #         least_lumped_degree_array.append(lumping_results[i])
    #
    # # print(least_lumped_degree_array)
    #
    # #finding the least error among least degrees of coupling in our matrices
    # if len(least_lumped_degree_array) >= 1:
    #     if len(least_lumped_degree_array) > 1:
    #         for i in range(len(least_lumped_degree_array)):
    #             if min_error > least_lumped_degree_array[i][1]:
    #                 min_error = least_lumped_degree_array[i][1]
    #                 best_lumped_array = []
    #                 best_lumped_array.append(least_lumped_degree_array[i])
    #
    #             elif min_error == least_lumped_degree_array[i][1]:
    #                 best_lumped_array.append(least_lumped_degree_array[i])
    #
    #         min_degree = best_lumped_array[0][0]
    #         min_error = best_lumped_array[0][1]
    #         best_lumped = best_lumped_array[0][2]
    #         best_sectors = best_lumped_array[0][3]
    #
    #     elif len(least_lumped_degree_array) == 1:
    #         min_degree = least_lumped_degree_array[0][0]
    #         min_error = least_lumped_degree_array[0][1]
    #         best_lumped = least_lumped_degree_array[0][2]
    #         best_sectors = least_lumped_degree_array[0][3]
    #     return min_degree, min_error, best_lumped, best_sectors
    #
    # else:
    #     return None


# def lumping():
#     newArr = np.array([[0.2, 0.28, 0.1, 0.21, 0.11, 0, 0.1],[0.29, 0.1, 0.2, 0.05, 0.31, 0, 0.05],[0.15, 0.2, 0.24, 0.12, 0.2, 0, 0.09],[0.27, 0.18, 0.22, 0.18, 0.01, 0, 0.14], [0.18, 0.2, 0.3, 0.31, 0.01, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0.25, 0.43, 0.07, 0.08, 0, 0.17]])
#     result = combinationEnum(newArr, 0)
#     print("result is: ", result)
#     results.append(result)
#     return results
# lumping()
# <codecell>


def lumping(reducedTransitionMatrix, percentageMatrix, boolean):
    result = combinationEnum(reducedTransitionMatrix, percentageMatrix, boolean)
    return result


def combination(numberOfSectors, numberOfStates, percentageMatrix):

    node_a = []
    node_a_stateSpace = True

    counter = 0 #index within sector values
    for j in range(len(percentageMatrix)):
        if percentageMatrix[j][2] == False:
            counter += 1
            if (percentageMatrix[j][1] >= 5) and (percentageMatrix[j][1] <= 25):
                node_a.append((counter, j))

            elif percentageMatrix[j][1] > 25:
                break

    root = Node(0)
    for i in range(len(node_a)):
        child = Node(node_a[i])
        root.add_child(child)

    extraArray = []

    count = 0
    for i in range(len(percentageMatrix)):
        if percentageMatrix[i][2] == False:
            count += 1
            extraArray.append((i, percentageMatrix[i][1]))
    # print("number of states are: ", numberOfStates, count)

    # print("for the first node we have:")
    # print(extraArray)
    # for child in root.children:
    #     print(child.data)
    # print("for the first node we had.")

    # print("I am going into recursive function of adding childre")

    root = add_child_recursion(root, 1, numberOfSectors, percentageMatrix, numberOfStates + 1)
    # print("I am going into recursive function of finding children")
    array = []
    sectorsArray = DFS_sectors(root, numberOfSectors + 1, array)
    # print(sectorsArray)
    sectors = []
    for i in range(len(sectorsArray)):
        extraArr = [0]
        for j in range(len(sectorsArray[i])):
            extraArr.append(sectorsArray[i][j][0])
        sectors.append(extraArr)
    # print(sectors)
    return sectors

def DFS_sectors(root, numberOfTotalSectors, array):
    sectors = []
    for child in root.children:
        array.append(child.data)
        if len(child.children) > 0:
            returnVal = DFS_sectors(child, numberOfTotalSectors, array)
            if len(returnVal) > 0:
                sectors.extend(returnVal)
        else:
            if len(array) == numberOfTotalSectors:
                sectors.append(list(array))
        del array[-1]
    return sectors


        # if len(array) == numberOfTotalSectors:
        #     sectors.append(array)
        #     print(sectors)
        # del array[-1]



def add_child_recursion(root, sectorCounter, numberOfSectors, percentageMatrix, numberOfStates):
    #counter is 1
    for child in root.children:
        # print(child.data)
        indices = child.data
        if sectorCounter < numberOfSectors:
            # print("here")
            returnedValues = checkBoundaries(indices[0], indices[1], percentageMatrix)

            # print(returnedValues)
            if returnedValues[0] == True:
                for i in range(len(returnedValues[1])):
                    grandChild = Node(returnedValues[1][i])
                    child.add_child(grandChild)
                # print(numberOfSectors, sectorCounter)
                if numberOfSectors >= sectorCounter + 1:
                    # print("here we go again")
                    add_child_recursion(child, sectorCounter + 1, numberOfSectors,
                                        percentageMatrix, numberOfStates)

        elif sectorCounter == numberOfSectors:
            # print("we want to add the last one")
            # print("now here")
            # print("we are adding the last one", (numberOfStates, len(percentageMatrix)))
            if ((percentageMatrix[-1][1] - percentageMatrix[indices[1]][1]) <= 25) and (
                (percentageMatrix[-1][1] - percentageMatrix[indices[1]][1]) >= 5):
                # print("we are adding the last one")
                grandChild = Node((numberOfStates, len(percentageMatrix)))
                child.add_child(grandChild)
    return root

def checkBoundaries(sector_index, percentageMatrix_index, percentageMatrix):
    # print("boundary checking")
    children = []
    for j in range(percentageMatrix_index + 1, len(percentageMatrix)):
        if percentageMatrix[j][2] == False:
            sector_index += 1
            if ((percentageMatrix[j][1] - percentageMatrix[percentageMatrix_index][1])
                >= 5) and (
                (percentageMatrix[j][1] - percentageMatrix[percentageMatrix_index][1])
                <= 25):
                children.append((sector_index, j))

            elif ((percentageMatrix[j][1] - percentageMatrix[percentageMatrix_index][1]) > 25):
                break
    if len(children) > 0:
        return True, children
    else:
        return False, children

def boundaries(candidate_subset, matrix_length, percentageMatrix):
    whole_subset = [0]
    whole_subset.extend(candidate_subset)
    whole_subset.append(matrix_length)

    list_of_percentages = [0]
    start = 0
    counter = 0
    for i in range(1, len(whole_subset)):
        for j in range(start, len(percentageMatrix)):
            start += 1
            if percentageMatrix[j][2] == False:
                counter += 1
                if counter == whole_subset[i]:
                    list_of_percentages.append((percentageMatrix[j][1]))
                    break

    for i in range(1, len(list_of_percentages)):
        if ((list_of_percentages[i] - list_of_percentages[i - 1]) > 25) or ((list_of_percentages[i] - list_of_percentages[i - 1]) < 5):
            return False

    return True


def check_limitation(candidate_subset, matrix_length, percentageMatrix, one_by_one_clustering):
    if one_by_one_clustering == False:
        return False
    else:
        whole_subset = [0]
        whole_subset.extend(candidate_subset)
        whole_subset.append(matrix_length)

        False_state_in_percentageMatrix = np.inf
        for i in range(1, len(whole_subset)):
            if whole_subset[i] - whole_subset[i - 1] == 2:
                False_state_in_percentageMatrix = whole_subset[i]
                break
                #it means states number (whole_subset[i] - 1) and (whole_subset[i] - 2) should merge
                #it is also the (whole_subset[i]) False and (whole_subset[i] - 1) False in percentageMatrix

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

        if index1_in_percentage_matrix != 0:
            if percentageMatrix[index2_in_percentage_matrix][1] - percentageMatrix[index1_in_percentage_matrix - 1][1] <= 25:
                return True
            else:
                return False

        else:
            if percentageMatrix[index2_in_percentage_matrix][1] <= 25:
                return True
            else:
                return False

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



class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)
