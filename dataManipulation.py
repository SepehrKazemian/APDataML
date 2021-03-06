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
from scipy import stats
import pandas
import warnings
from sklearn.cluster import KMeans
import logging


def dataSplitting(fileName, channelBasedBool, classifier, timeInterval, address):
	print("new")
	logging.basicConfig(filename='example.log',level=logging.DEBUG)
	print("salam babaaaaa")
	#print(fileName + " in datasplitting func")
	now_timestamp = time.time()
	offset = datetime.datetime.fromtimestamp(now_timestamp) - datetime.datetime.utcfromtimestamp(now_timestamp)
	secondCounter = 0
	prevTime = ""
	curTime = ""
	currTimeStamp = ""
	cu = ""
	lineCounter = 0
	signalVal = 0
	counter = 0
	preMaxVal = 0
	channelCheck = 0
	maxVal = 0
	fileRead = address + str(fileName)
	countingLags = 1
	
	counterLag = 0
	strLags = "-" + str(counterLag) 
	fileWrite = str(fileName) + strLags
	print("the writing fileName is: " + fileWrite)
		
	#print("here")
	csvArrTime = []
	csvArrCU = []
	csvArrCUVal = []
	chanUtil = []
	chanUtilVal = []
	timeArr = []
	with open(fileRead) as fp:
		for line in fp:
			cu = ""
			lineCounter += 1
			if lineCounter % 10000 == 0:
				print(lineCounter)
			month = ""
			day = ""
			timer = ""
			year = ""
			channel = ""
			
			
			#Month recognization from the file
	#		print(line[13:21])
			timer = line[13:21]

			if line[0:3] == "Oct":
				month = "10"
			elif line[0:3] == "Nov":
				month = "11"
			elif line[0:3] == "Dec":
				month = "12"
			elif line[0:3] == "Sep":
				month = "09"
			elif line[0:3] == "Aug":
				month = "08"
			elif line[0:3] == "Jan":
				month = "01"
			elif line[0:3] == "Feb":
				month = "02"
			elif line[0:3] == "Mar":
				month = "03"
			elif line[0:3] == "Apr":
				month = "04"
			elif line[0:3] == "May":
				month = "05"
			elif line[0:3] == "Jun":
				month = "06"
			elif line[0:3] == "Jul":
				month = "07"
			else:
				print(line)
				print(line[0:3])
				print(line[4:6])
				print(line[8:12])
				print("problem in month")
				
			day = line[4:6]
			
			year = line[8:12]
			
			stringTime = month + "/" + day + "/" + year + " " + timer 
			
			try:
				currTimeStamp = time.mktime(datetime.datetime.strptime(stringTime, "%m/%d/%Y %H:%M:%S").timetuple())
			except ValueError:
				print("d" + str(stringTime))
			
			for i in range(36,39):
				if line[i] != " ":
					channel += line[i]
				else:
					break
			
			if int(channelBasedBool) == 1 and int(channelCheck) != int(channel):
				print("channel boolean is: " + str(channelBasedBool))
				if channelCheck != 0: #channelCheck != 0 (initial value) and it is != channel which means channel changed
					print("channel changed")
					pandasWriting(csvArrTime, csvArrCU, fileWrite)
					csvArrTime = []
					csvArrCU = []
					
				print(fileWrite)
				fileWrite = "channel" + str(channel)
				print(fileWrite)
				channelCheck = channel
			
			#channel changes the string positions
			if len(channel) == 1:
				for i in range(38,41):
					if line[i] != " ":
						cu += line[i]
					else:
						break
			elif len(channel) == 2:
				for i in range(39,42):
					if line[i] != " ":
						cu += line[i]
					else:
						break
			elif len(channel) == 3:
				for i in range(40,43):
					if line[i] != " ":
						cu += line[i]
					else:
						break				
			
			signalReverse = ""
			signal = ""
			
			for i in range(len(line) - 1, 0, -1):
				if line[i] == " ":
					break
				else:	
					signalReverse += line[i]
				
			signal = signalReverse[::-1]
			
			try:
				signalVal += int(signal)
			except ValueError:
				print("a" + str(signal))
			
			try:
				if counter == 0:
					# logging.info("first time")
					# logging.info(str(currTimeStamp))
					# logging.info(str(prevTime))
					# logging.info(str(int(cu)))
					# logging.info(str(maxVal))

					
					prevTime = currTimeStamp
					maxVal = int(cu)
					counter = 1
				
				elif currTimeStamp - prevTime >= timeInterval:
					# print("bigger than interval")
					# print("currTimeStamp is: " + str(currTimeStamp))
					# print("prevTime is: " + str(prevTime))
					
					# logging.info("second time")
					# logging.info(str(currTimeStamp))
					# logging.info(str(prevTime))
					# logging.info(str(int(cu)))
					# logging.info(str(maxVal))
					creatingLagsBoolean = 0
					
					
					while int((currTimeStamp - prevTime) / timeInterval) > 0:
						#print("wowowwww")
						if creatingLagsBoolean == 0 and int((currTimeStamp - prevTime) / timeInterval) < 100:
							currTimeStampUTC = datetime.datetime.fromtimestamp(prevTime).strftime('%Y-%m-%d %H:%M:%S')
							currTimeStampUTC = datetime.datetime.strptime(currTimeStampUTC, '%Y-%m-%d %H:%M:%S')
							#central = offset + currTimeStampUTC
							central = currTimeStampUTC
							
							#it seems sickit learn cannot work with datetime, so we have to reconvert it to
							#values
							centralTimeStamp = (datetime.datetime.strptime(str(central), "%Y-%m-%d %H:%M:%S") - datetime.datetime(1970,1,1)).total_seconds()
							
							csvArrTime.append(central)
							timeArr.append(central)
							csvArrCU.append(maxVal)
							if classifier == "man":
								chanUtil = np.append(chanUtil, normalClassification(maxVal/255))

						elif creatingLagsBoolean == 0:
							creatingLagsBoolean = 1
							pandasWriting(csvArrTime, csvArrCU, fileWrite, address)
							csvArrTime = []
							csvArrCU = []
							counterLag += 1
							strLags = "-" + str(counterLag)
							fileWrite =  str(fileName) + strLags
							
						prevTime = prevTime + timeInterval
					

					
					maxVal = 0
					
				
				elif currTimeStamp - prevTime < timeInterval:
					# logging.info("forth time")
					# logging.info(str(currTimeStamp))
					# logging.info(str(prevTime))
					# logging.info(str(int(cu)))
					# logging.info(str(maxVal))
					if maxVal < int(cu):
						maxVal = int(cu)
				

			except ValueError:
				print("channel utilization error")
	

	pandasWriting(csvArrTime, csvArrCU, fileWrite, address)
	
	if classifier == "k":
		chanUtil = clusteringKMeans(csvArrCU)

		
	return chanUtil, timeArr
	
	
	
def normalDataSplitting(fileName, channelBasedBool, classifier, timeInterval, address):
	logging.basicConfig(filename='example.log',level=logging.DEBUG)
	print("salam babaaaaaaa")
	#print(fileName + " in datasplitting func")
	now_timestamp = time.time()
	offset = datetime.datetime.fromtimestamp(now_timestamp) - datetime.datetime.utcfromtimestamp(now_timestamp)
	secondCounter = 0
	prevTime = ""
	curTime = ""
	currTimeStamp = ""
	cu = ""
	lineCounter = 0
	signalVal = 0
	counter = 0
	preMaxVal = 0
	channelCheck = 0
	maxVal = 0
	fileRead = address + str(fileName)
	countingLags = 1
	
	counterLag = 0
	strLags = "-" + str(counterLag) 
	fileWrite = str(fileName)
	print("the writing fileName is: " + fileWrite)
		
	#print("here")
	csvArrTime = []
	csvArrCU = []
	csvArrCUVal = []
	chanUtil = []
	chanUtilVal = []
	timeArr = []
	with open(fileRead) as fp:
		for line in fp:
			cu = ""
			lineCounter += 1
			if lineCounter % 10000 == 0:
				print(lineCounter)
			month = ""
			day = ""
			timer = ""
			year = ""
			channel = ""
			
			
			#Month recognization from the file
	#		print(line[13:21])
			timer = line[13:21]

			if line[0:3] == "Oct":
				month = "10"
			elif line[0:3] == "Nov":
				month = "11"
			elif line[0:3] == "Dec":
				month = "12"
			elif line[0:3] == "Sep":
				month = "09"
			elif line[0:3] == "Aug":
				month = "08"
			elif line[0:3] == "Jan":
				month = "01"
			elif line[0:3] == "Feb":
				month = "02"
			elif line[0:3] == "Mar":
				month = "03"
			elif line[0:3] == "Apr":
				month = "04"
			elif line[0:3] == "May":
				month = "05"
			elif line[0:3] == "Jun":
				month = "06"
			elif line[0:3] == "Jul":
				month = "07"				
			else:
				print(line)
				print(line[0:3])
				print(line[4:6])
				print("problem in month")
				
			day = line[4:6]
			
			year = line[8:12]
			
			stringTime = month + "/" + day + "/" + year + " " + timer 
			
			try:
				currTimeStamp = time.mktime(datetime.datetime.strptime(stringTime, "%m/%d/%Y %H:%M:%S").timetuple())
			except ValueError:
				print("d" + str(stringTime))
			
			for i in range(36,39):
				if line[i] != " ":
					channel += line[i]
				else:
					break
			
			if int(channelBasedBool) == 1 and int(channelCheck) != int(channel):
				print("channel boolean is: " + str(channelBasedBool))
				if channelCheck != 0: #channelCheck != 0 (initial value) and it is != channel which means channel changed
					print("channel changed")
					pandasWriting(csvArrTime, csvArrCU, fileWrite)
					csvArrTime = []
					csvArrCU = []
					
				print(fileWrite)
				fileWrite = "channel" + str(channel)
				print(fileWrite)
				channelCheck = channel
			
			#channel changes the string positions
			if len(channel) == 1:
				for i in range(38,41):
					if line[i] != " ":
						cu += line[i]
					else:
						break
			elif len(channel) == 2:
				for i in range(39,42):
					if line[i] != " ":
						cu += line[i]
					else:
						break
			elif len(channel) == 3:
				for i in range(40,43):
					if line[i] != " ":
						cu += line[i]
					else:
						break				
			
			signalReverse = ""
			signal = ""
			
			for i in range(len(line) - 1, 0, -1):
				if line[i] == " ":
					break
				else:	
					signalReverse += line[i]
				
			signal = signalReverse[::-1]
			
			try:
				signalVal += int(signal)
			except ValueError:
				print("a" + str(signal))
			
			try:
				if counter == 0:
					# logging.info("first time")
					# logging.info(str(currTimeStamp))
					# logging.info(str(prevTime))
					# logging.info(str(int(cu)))
					# logging.info(str(maxVal))

					
					prevTime = currTimeStamp
					maxVal = int(cu)
					counter = 1
				
				elif currTimeStamp - prevTime >= timeInterval:
					# print("bigger than interval")
					# print("currTimeStamp is: " + str(currTimeStamp))
					# print("prevTime is: " + str(prevTime))
					
					# logging.info("second time")
					# logging.info(str(currTimeStamp))
					# logging.info(str(prevTime))
					# logging.info(str(int(cu)))
					# logging.info(str(maxVal))
					creatingLagsBoolean = 0
					
					
					while int((currTimeStamp - prevTime) / timeInterval) > 0:
						#print("wowowwww")
						if creatingLagsBoolean == 0:
							currTimeStampUTC = datetime.datetime.fromtimestamp(prevTime).strftime('%Y-%m-%d %H:%M:%S')
							currTimeStampUTC = datetime.datetime.strptime(currTimeStampUTC, '%Y-%m-%d %H:%M:%S')
							#central = offset + currTimeStampUTC
							central = currTimeStampUTC
							
							#it seems sickit learn cannot work with datetime, so we have to reconvert it to
							#values
							centralTimeStamp = (datetime.datetime.strptime(str(central), "%Y-%m-%d %H:%M:%S") - datetime.datetime(1970,1,1)).total_seconds()
							
							csvArrTime.append(central)
							timeArr.append(central)
							csvArrCU.append(maxVal)
							creatingLagsBoolean = 1
							if classifier == "man":
								chanUtil = np.append(chanUtil, normalClassification(maxVal/255))

							
						prevTime = prevTime + timeInterval
					

					
					maxVal = 0
					
				
				elif currTimeStamp - prevTime < timeInterval:
					# logging.info("forth time")
					# logging.info(str(currTimeStamp))
					# logging.info(str(prevTime))
					# logging.info(str(int(cu)))
					# logging.info(str(maxVal))
					if maxVal < int(cu):
						maxVal = int(cu)
				

			except ValueError:
				print("channel utilization error")
	

	pandasWriting(csvArrTime, csvArrCU, fileWrite, address)


			
def pandasWriting(csvArrTime, csvArrCU, fileWrite, address):
	dataFile = pandas.DataFrame(data = {"time": csvArrTime, "CU": csvArrCU})
	with open((str(address) + "/CSV/" + str(fileWrite) + ".csv"), 'a') as file:
		dataFile.to_csv(file, sep = ",", header = False)
	print("hereee")

def normalClassification(value):
	if (value) < 0.25:
		return "0"
	elif (value) < 0.5:
		return "1"
	elif (value) < 0.75:
		return "2"		
	else:
		return "3"	

		
def clusteringKMeans(values):
	kmeans = KMeans(n_clusters = 4, random_state = 0).fit(values)
	classifiedValues = kmeans.labels_
	print("centers are: " + str(kmeans.cluster_centers_))
	return classifiedValues
