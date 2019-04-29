from dateutil.relativedelta import relativedelta
import datetime
import time
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
import pandas as pd
from dateutil import tz
import os
import subprocess
import numpy as np
import math
import os.path
import pickle
import threading, time
from threading import Thread
from multiprocessing import Process, Queue, dummy as multithreading
from multiprocessing.dummy import Pool as ThreadPool
import sklearn.linear_model as linear_model
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
import warnings
from sklearn.cluster import KMeans
import logging
import Plot as plot
import pysal
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from discreteMarkovChain import markovChain
import learningAlgs as LA
from scipy import signal





print("start loading")
address = input("give the channel seperation folder files address: ")
plotter = plot.plottData()
LA_Obj = LA.learningAlgs()
# fileNameArr = ["500f8027140-500f8022b88-H11.txt", "500f8022b88-500f8027140-H11.txt",
	# "500f8027112-500f8027140-H11.txt", "500f8027140-500f8027112-H11.txt",
	# "500f8027112-500f8022b88-H11.txt", "500f8022b88-500f8027112-H11.txt",
	# "500f801cf22-500f8022aca-H11.txt", "500f8022aca-500f801cf22-H11.txt",
	# "500f8027112-500f8022aca-H11.txt", "500f8022aca-500f8027112-H11.txt",
	# "500f8027140-500f8022aca-H11.txt", "500f8022aca-500f8027140-H11.txt",
	# "500f8022b88-500f8022aca-H11.txt", "500f8022aca-500f8022b88-H11.txt",
	# "500f8022ab4-500f8022aca-H11.txt", "500f8022aca-500f8022ab4-H11.txt",
	# "500f8022b88-500f8022ab4-H11.txt", "500f8022ab4-500f8022b88-H11.txt",
	# "500f8027112-500f8022ab4-H11.txt", "500f8022ab4-500f8027112-H11.txt",
	# "500f8022b88-500f801cf22-H11.txt", "500f801cf22-500f8022b88-H11.txt",
	# "500f8027140-500f8022ab4-H11.txt", "500f8022ab4-500f8027140-H11.txt",
	# "500f8027140-500f801cf22-H11.txt", "500f801cf22-500f8027140-H11.txt",
	# "500f8027140-500f801cf9c-H11.txt", "500f801cf9c-500f8027140-H11.txt",
	# "500f8022b88-500f801cf9c-H11.txt", "500f801cf9c-500f8022b88-H11.txt", 
	# "500f801cf9c-500f8027112-H11.txt", "500f8027112-500f801cf9c-H11.txt"]


# *****************************getting the fileNames and find its counterpart for processing*************************************
fileNameArr = os.listdir(address)
extraArr = []
for i in range(len(fileNameArr)):
	if ".txt" not in fileNameArr[i]:
		extraArr.append(fileNameArr[i])

for i in range(len(extraArr)):
	fileNameArr.remove(extraArr[i])

fileNameOrdered = []
extraArr = fileNameArr
for i in range(len(fileNameArr)):
	for j in range(i + 1, len(fileNameArr)):
		mac1 = fileNameArr[i][0:11]
		mac2 = fileNameArr[i][12:23]
		ch = fileNameArr[i][24:27]
		otherFileName = mac2 + "-" + mac1 + "-" + ch + ".txt"
		if fileNameArr[j] == otherFileName and otherFileName not in fileNameOrdered:
			fileNameOrdered.append(fileNameArr[i])
			fileNameOrdered.append(fileNameArr[j])
			break
			
print(fileNameOrdered)

	
fileNameArr = fileNameOrdered
	
	
# ************************loading the data from csv files ********************************	
dataDimensFreeHours = [[] for i in range(len(fileNameArr))]
dataDimensBusyHours = [[] for i in range(len(fileNameArr))]

for i in range(len(fileNameArr)):
	stat, chanUtil, timeArr = LA_Obj.seperatedCsvChecker(fileNameArr[i], 0, address)
	if stat == True:
		#print(timeArr)
		# print(timeArr[0])
		# print(timeArr[0].hour)
		# print(type(timeArr[0]))
		# print(timeArr[0].day)
		day = timeArr[0].day
		print("the day is: " + str(day))
		for j in range(len(chanUtil)):
			if (timeArr[j].hour > 19 and timeArr[j].day == day) or (timeArr[j].hour < 10 and timeArr[j].day == day + 1): #boundaries for time
				dataDimensFreeHours[i].append(chanUtil[j])

		for j in range(len(chanUtil)):
			if (timeArr[j].hour <= 19 and timeArr[j].hour >= 10 and timeArr[j].day == day): #boundaries for time
				dataDimensBusyHours[i].append(chanUtil[j])
				
		print(len(dataDimensFreeHours[i]))
		print(len(dataDimensBusyHours[i]))	
	
	
#/home/Sepehr/Desktop/project/thesis/data/11node1-5/alligned/	
	
name = []
lag0ValBusy = []
lag0ValFree = []
dataLenBusy = []
dataLenFree = []
busyTimeThreshold = []
freeTimeThreshold = []

#*******************************************Finding cross correlationg *************************	
for i in range(0, len(fileNameArr), +2):
	print("I am in")
	print(fileNameArr[i])
	print(i)
	
	if len(dataDimensFreeHours[i]) == 0 or len(dataDimensBusyHours[i]) == 0:
		print("I have to continue the loop")
		continue
	
	dataLenFree.append(len(dataDimensFreeHours[i]))
	dataLenBusy.append(len(dataDimensBusyHours[i]))
	print(dataLenFree)
	print(dataLenBusy)
	
	in1 = (dataDimensFreeHours[i] - np.mean(dataDimensFreeHours[i]))/(np.std(dataDimensFreeHours[i]) * len(dataDimensFreeHours[i]))
	in2 = (dataDimensFreeHours[i + 1] - np.mean(dataDimensFreeHours[i + 1]))/(np.std(dataDimensFreeHours[i + 1]))
	corrFree = signal.correlate(in1, in2, mode='full')
	
	in1 = (dataDimensBusyHours[i] - np.mean(dataDimensBusyHours[i]))/(np.std(dataDimensBusyHours[i]) * len(dataDimensBusyHours[i]))
	in2 = (dataDimensBusyHours[i + 1] - np.mean(dataDimensBusyHours[i + 1]))/(np.std(dataDimensBusyHours[i + 1]))	
	corrBusy = signal.correlate(in1, in2, mode='full')

	#print("len of corr is: " + str(len(corr)))
	np.set_printoptions(threshold=np.inf)
	# print(corr)
	counterFree = []
	for j in range( - int(len(corrFree) / 2), 0, +1):
		counterFree.append(j)
	for j in range(0, int(len(corrFree) / 2) + 1, +1):
		counterFree.append(j)
		
		
	counterBusy = []
	for j in range( - int(len(corrBusy) / 2), 0, +1):
		counterBusy.append(j)
	for j in range(0, int(len(corrBusy) / 2) + 1, +1):
		counterBusy.append(j)
		
	busyThreshold = 0
	for j in range(int(len(corrBusy) / 2), 0, -1):
		if corrBusy[j] > 0.3:
			busyThreshold += 1
		else:
			break
			
	for j in range(int(len(corrBusy) / 2), int(len(corrBusy)), +1):
		if corrBusy[j] > 0.3:
			busyThreshold += 1
		else:
			break
			
			
	freeThreshold = 0
	for j in range(int(len(corrFree) / 2), 0, -1):
		if corrFree[j] > 0.3:
			freeThreshold += 1
		else:
			break
			
	for j in range(int(len(corrFree) / 2), int(len(corrFree)), +1):
		if corrFree[j] > 0.3:
			freeThreshold += 1
		else:
			break

	name.append(fileNameArr[i])
			
	lag0ValBusy.append(np.max(corrBusy))
	lag0ValFree.append(np.max(corrFree))
	
	busyTimeThreshold.append(busyThreshold)
	freeTimeThreshold.append(freeThreshold)
	

	# trace1 = go.Scatter(
	    # x = counterFree,
	    # y = corrFree,
	    # mode = 'lines',
	    # name = 'lines'
	# )
	# data = [trace1]
	# name = str(fileNameArr[i]) + "FreeHours.html"
	# plotAdd = address + "/plot/" + name
	# offline.plot(data, filename = plotAdd)
	
	
	# trace2 = go.Scatter(
	    # x = counterBusy,
	    # y = corrBusy,
	    # mode = 'lines',
	    # name = 'lines'
	# )
	# data = [trace2]
	# name = str(fileNameArr[i]) + "BusyHours.html"
	# plotAdd = address + "/plot/" + name
	# offline.plot(data, filename = plotAdd)	
	

trace = go.Table(
    header=dict(values=['name', 'lag0ValBusy', 'lag0ValFree', 'busyThreshold', 'freeThreshold', 'dataDimensFreeHours', 'dataDimensBusyHours']),
    cells=dict(values=[name, lag0ValBusy, lag0ValFree, busyTimeThreshold, freeTimeThreshold, dataLenFree, dataLenBusy]
                       ))

data = [trace] 
plotAdd = address + "/plot/table.html"
offline.plot(data, filename = plotAdd)