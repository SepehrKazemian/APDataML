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
#*******************************************************************************
# data.tail()
# plotDataFrame = data.loc[(data["timeIndex"] > 2) & (data["timeIndex"] < 12)]
# time = [j for j in range(data_CU_to_numpy.shape[0])]
# data_day_to_list = plotDataFrame["time"].to_list()
# a = []
# a.append(go.Scatter(x = time[:150000], y = data_CU_to_numpy[:150000]))
# fileName ="non-busy hour1-6pm.html"
# offline.plot(a, filename=fileName, image='svg')
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

    # avgPenalty *= 1-(arrayOfStatesPercentage[predictedStateIndex])
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


def clustering(normalizedTransitionMatrix, index):

    overal_merged_states = []
    overal_bandwidth_vector = []
    print(normalizedTransitionMatrix)
    print(np.sum(normalizedTransitionMatrix))
    print(normalizedTransitionMatrix.shape)

    # for i in range(normalizedTransitionMatrix.sha - 1pe[0]):
    for i in range(1):

        # print(normalizedTransitionMatrix.shape)
        percentageMatrix = bandwidthPercentage(normalizedTransitionMatrix)
        # a percentage Matrix for each time interval

        mergedStates = normalizedTransitionMatrix
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
                print("the value of i is: ", index)
                print(mergedStates)
                print(percentageMatrix)
                return mergedStates, percentageMatrix
                break

    return overal_merged_states, overal_bandwidth_vector




transExtra = cuTrans_cpy.copy()
overal_merged_states, overal_bandwidth_vector = clustering(transExtra)


def classifying(CU, boundaries):
    occupiedBandwidth = (CU / 255) * 100
    for i in range(boundaries.shape[0]):
        if occupiedBandwidth <= boundaries[i]:
            return i

saveResults = []




def simpleLogisticRegression(data, numberOfTimeIntervals, minuteSplit):

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

        x = 30
        iterPandas = data.loc[(data["timeIndex"] == x)].copy()

        trainingDataFrame = iterPandas.iloc[:int(np.floor(0.8 * len(iterPandas)))].copy()

        trainingTransitionMatrix = processData.markovianTransitionMatrixDegree1(trainingDataFrame, 255)
        normalizedTrainTrans = processData.normalizingTransMatrix(trainingTransitionMatrix)
        normalizedTrainTransition = normalizedTrainTrans.copy()
        transitionMatrix, boundaries = clustering(normalizedTrainTransition[-1], normalizedTrainTransition.shape[0])

        trainingDataFrame["cuClass"] = trainingDataFrame["CU"].apply(lambda x: classifying(x, boundaries))

        stackCounter = 0
        prevCU = 0
        for index, row in trainingDataFrame.iterrows():
            lastCU = np.zeros(boundaries.shape[0])
            lastCU[prevCU] = 1
            which6SecondsPerPeriod[int(row["periodParticle"])] = 1
            XArray = lastCU
            XArray = np.append(XArray, which6SecondsPerPeriod)

            if stackCounter == 0:
                XArraysForLearning = XArray
            else:
                XArraysForLearning = np.vstack((XArraysForLearning, XArray))

            recentCU = np.zeros(boundaries.shape[0])
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


        testingDataFrame = iterPandas.iloc[int(np.floor(0.2 * len(iterPandas))):].copy()

        testingDataFrame["cuClass"] = testingDataFrame["CU"].apply(lambda x: classifying(x, boundaries))


        stackCounter = 0
        prevCU = 0
        for index, row in testingDataFrame.iterrows():

            lastCU = np.zeros(boundaries.shape[0])
            lastCU[prevCU] = 1
            which6SecondsPerPeriod[int(row["periodParticle"])] = 1
            XArray = lastCU
            XArray = np.append(XArray, which6SecondsPerPeriod)

            if stackCounter == 0:
                XArraysForTesting = XArray

            else:
                XArraysForTesting = np.vstack((XArraysForTesting, XArray))
    #         print(XArraysForTesting)

            recentCU = np.zeros(boundaries.shape[0])
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
        # clf.fit(XArraysForLearning, YArraysForLearning.ravel())
        # Ypred = clf.predict(XArraysForTesting)
        # acc = accuracy_score(YArraysForTesting, Ypred)
        # print(acc)
        # saveResults.append((x, acc))

def tensorFlowLossFunction(XArraysForLearning, YArraysForLearning, XArraysForTesting, YArraysForTesting, boundaries):
    print(XArraysForLearning.shape)
    print(YArraysForLearning.shape)
    batch_size = 64
    learning_rate = 0.001
    num_steps = 10000
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape = (batch_size, XArraysForLearning.shape[1]))
        y_ = tf.placeholder(tf.float32, shape = (batch_size, YArraysForLearning.shape[1]))
        W = tf.Variable(tf.truncated_normal([XArraysForLearning.shape[1], YArraysForLearning.shape[1]]), name="weights", dtype=tf.float32)
        b = tf.Variable(tf.truncated_normal([YArraysForLearning.shape[1]]), dtype=tf.float32)

        tf_test_dataset64 = tf.constant(XArraysForTesting)
        tf_test_dataset = tf.cast(tf_test_dataset64, tf.float32)


        beta = 0.05
        logits = tf.matmul(x, W)
        train_prediction = tf.nn.softmax(logits)
        # train_prediction = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_, logits = logits)
        test_prediction = tf.nn.softmax(tf.add(tf.matmul(tf_test_dataset, W),b))

        # x = XArraysForLearning[0:(0 + batch_size), :]
        # y_ = tf.Variable(YArraysForLearning[0:(0 + batch_size), :])

        # loss = assymetricLossFunction(train_prediction, y_, boundaries)
        loss = assymetricLossFunction(train_prediction, y_, boundaries)
        # loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = y_)
        regularizer = tf.nn.l2_loss(W)
        # loss = tf.reduce_mean(loss + beta * regularizer)
        # loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = train_prediction, labels = y_)
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        prevAcc = 0
        earlyStoppingCounter = 0
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Initialized")

        for step in range(num_steps):
            i = 0
            batch_data = XArraysForLearning[i:(i + batch_size), :]
            batch_labels = YArraysForLearning[i:(i + batch_size), :]

            i += batch_size

            feed_dict = {x : batch_data, y_ : batch_labels}
            _, predictions, l = session.run([optimizer, train_prediction, loss], feed_dict=feed_dict)

            # if accuracy(test_prediction.eval(), YArraysForTesting) < prevAcc and earlyStoppingCounter == 400:
            #     print(prevAcc)
            #     break
            #
            # elif accuracy(test_prediction.eval(), YArraysForTesting) < prevAcc:
            #     earlyStoppingCounter += 1
            #
            # elif accuracy(test_prediction.eval(), YArraysForTesting) >= prevAcc:
            #     earlyStoppingCounter = 0

            if (step % 200 == 0):
                print("Minibatch step {0}".format(step))
                # print("training Acc is: {:.3f}".format(accuracy(predictions,batch_labels)))
                # prevAcc = accuracy(test_prediction.eval(), YArraysForTesting)
                print(np.mean(l))

        # print(session.run(W))
        print("\nPenalty Value: {:.3f}".format(assymetricPredictionScore(test_prediction.eval(), YArraysForTesting, boundaries)))
        print("\naccuracy Acc: {:.3f}".format(accuracy(test_prediction.eval(), YArraysForTesting)))


def assymetricLossFunction(prediction, correctLable, boundaries):
    sess = tf.Session()
    xAxisPoints = np.linspace(rayleigh.ppf(0.01), rayleigh.ppf(0.99), 338)
    #number of overal datapoints must stay the same all the time
    maxState = 338
    inverseDistrib = max(rayleigh.pdf(xAxisPoints)) - rayleigh.pdf(xAxisPoints)
    inverseDistrib = tf.constant(inverseDistrib)
    xAxisPoints -= xAxisPoints[np.argmin(inverseDistrib)]
    minState = np.argmin(inverseDistrib)
    numberOfOverUtilizedStates = maxState - minState
    numberOfUnderUtilizedStates = minState
    minState = tf.constant(minState, tf.float32)
    numberOfOverUtilizedStates = tf.constant(numberOfOverUtilizedStates, tf.float32)
    numberOfUnderUtilizedStates = tf.constant(numberOfUnderUtilizedStates, tf.float32)

    underUtilVal = numberOfUnderUtilizedStates / 100
    overUtilVal = numberOfOverUtilizedStates / 100


    boundaries = tf.constant(boundaries, tf.float32)
    correctLableIndex = tf.argmax(correctLable, 1)

    diffPercentage = []
    for index in range(correctLableIndex.shape[0]):
        diffPercentage.append(boundaries[correctLableIndex[index]] - boundaries[:])

    diffPercentage = tf.stack(diffPercentage)

    penalties = []

    counter = 0
    for i in range(diffPercentage.shape[0]):
        for j in range(diffPercentage.shape[1]):
            counter += 1

            penalties.append(tf.cond(
                    tf.greater(diffPercentage[i][j], 0),
                    lambda: inverseDistrib[tf.dtypes.cast(minState + tf.math.floor
                                                                   (tf.math.scalar_mul(diffPercentage[i][j],
                                                                                       overUtilVal)), tf.int32)],
                    lambda: inverseDistrib[tf.dtypes.cast(minState + tf.math.floor
                                                                   (tf.math.scalar_mul
                                                                    (diffPercentage[i][j], underUtilVal))
                                                                   , tf.int32)]
                    ))

    penalties = tf.stack(penalties)
    penalties = tf.dtypes.cast(penalties, tf.float32)
    penalties = tf.reshape(penalties, diffPercentage.shape)

    # weights = tf.reduce_sum(penalties * (1-prediction), axis=1)
    weights = (1 - penalties) * prediction
    # print(correctLable)
    # print(prediction)
    # print(penalties)
    loss = tf.losses.softmax_cross_entropy(onehot_labels = correctLable, logits = weights)
    # weighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels = penalties, logits = prediction)
    # loss = tf.reduce_sum(weighted_losses)
    # loss = tf.reduce_sum(penalties * prediction)
    return tf.reduce_mean(loss)

def assymetricPredictionScore(predictedLables, trueLables, boundaries):
    xAxisPoints = np.linspace(rayleigh.ppf(0.01), rayleigh.ppf(0.99), 338)
    #number of overal datapoints must stay the same all the time
    maxState = 338

    inverseDistrib = max(rayleigh.pdf(xAxisPoints)) - rayleigh.pdf(xAxisPoints)
    minState = np.argmin(inverseDistrib)

    underUtilizedSum = 0
    overUtilizedSum = 0
    numberOfUnderUtilizedStates = minState - 0
    numberOfOverUtilizedStates = maxState - minState

    xAxisPoints -= xAxisPoints[np.argmin(inverseDistrib)]

    underUtilVal = numberOfUnderUtilizedStates / 100
    overUtilVal = numberOfOverUtilizedStates / 100

    correctLableIndex = np.argmax(trueLables, 1)
    predictionIndex = np.argmax(predictedLables, 1)

    diffPercentage = np.zeros(shape = (predictedLables.shape))

    penalties = np.zeros(shape = (predictedLables.shape))

    for index in range(predictedLables.shape[0]):
        diffPercentage[index] = boundaries[correctLableIndex[index]] - boundaries[:]


    for i in range(diffPercentage.shape[0]):
        for j in range(diffPercentage.shape[1]):
            if diffPercentage[i][j] > 0:
                penalties[i][j] = inverseDistrib[minState + math.floor
                                                  (diffPercentage[i][j] * overUtilVal)] * 100
            else:
                penalties[i][j] = inverseDistrib[minState + math.floor
                                                  (diffPercentage[i][j] * underUtilVal)] * 100
    sumOfPenalty = 0
    for i in range(predictionIndex.shape[0]):
        sumOfPenalty += penalties[i][predictionIndex[i]]

    return sumOfPenalty

def accuracy(predictedLables, trueLables):
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    correctLableIndex = np.argmax(trueLables, 1)
    predictionIndex = np.argmax(predictedLables, 1)
    acc = np.float64(np.sum(correctLableIndex == predictionIndex)/predictedLables.shape[0])
    wrongPred = np.where(predictionIndex != correctLableIndex)
    # print(predictionIndex[wrongPred])
    # print(correctLableIndex[wrongPred])
    return acc


XArraysForLearning, YArraysForLearning, XArraysForTesting, YArraysForTesting, boundaries = simpleLogisticRegression(data, 48, 30)
tensorFlowLossFunction(XArraysForLearning, YArraysForLearning, XArraysForTesting, YArraysForTesting, boundaries)
