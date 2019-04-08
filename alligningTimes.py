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
from scipy import stats
import pandas
import warnings
from sklearn.cluster import KMeans
import logging

class alligningTime:
	def __init__(self):
		self.set1 = set()
		self.set2 = set()

	def alignTime(self, fileName, setName):
		stringTime = ""
		now_timestamp = time.time()
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
		fileWrite = str(fileName)
			
		#print("here")
		csvArrTime = []
		csvArrCU = []
		csvArrCUVal = []
		chanUtil = []
		chanUtilVal = []
		timeArr = []
		#print(fileName + " in datasplitting func")
		with open(fileName) as fp:
			for line in fp:

				cu = ""
				lineCounter += 1
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
				else:
					print("problem in month")
					
				day = line[4:6]
				if day[0] == " ":
					listDay = list(day)
					listDay[0] = "0"
					day = ''.join(listDay)
				
				year = line[8:12]
				stringTime = month + "/" + str(day) + "/" + str(year) + " " + str(timer)
				try:
					currTimeStamp = time.mktime(datetime.datetime.strptime(stringTime, "%m/%d/%Y %H:%M:%S").timetuple())
					setName.add(stringTime)
				except ValueError:
					print("d" + str(stringTime))			
				#currTimeStamp = time.mktime(datetime.datetime.strptime(stringTime, "%m/%d/%Y %H:%M:%S").timetuple())
				
		
		return setName
	
	def writeTheAllignedFile(self, fileName, unitedSet, fileWrite):
		stringTime = ""
		now_timestamp = time.time()
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
			
		#print("here")
		csvArrTime = []
		csvArrCU = []
		csvArrCUVal = []
		chanUtil = []
		chanUtilVal = []
		timeArr = []
		#print(fileName + " in datasplitting func")
		with open(fileName) as fp:
			for line in fp:
				cu = ""
				lineCounter += 1
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
				else:
					print("problem in month")
					
				day = line[4:6]
				if day[0] == " ":
					listDay = list(day)
					listDay[0] = "0"
					day = ''.join(listDay)
				
				year = line[8:12]
				stringTime = month + "/" + str(day) + "/" + str(year) + " " + str(timer)
				if stringTime in unitedSet:
					with open(fileWrite, 'a') as file:
						file.write(line)

				
							
				
if __name__ == '__main__':
	fileName1 = "500f801cf9c0-2.txt"
	fileName2 = "500f801cf9c0-5.txt"
	obj = alligningTime()
	print(obj.set1)
	print(obj.set2)
	thread1 = threading.Thread(target = obj.alignTime, args = (fileName1, obj.set1))
	thread2 = threading.Thread(target = obj.alignTime, args = (fileName2, obj.set2))	
	thread1.start()
	thread2.start()
	thread1.join()
	thread2.join()
	unitedSet = (obj.set1 & obj.set2)
	fileWrite1 = "Node2.txt"
	fileWrite2 = "Node5.txt"
	print("starting to write")
	thread1 = threading.Thread(target = obj.writeTheAllignedFile, args = (fileName1, unitedSet, fileWrite1))
	thread2 = threading.Thread(target = obj.writeTheAllignedFile, args = (fileName2, unitedSet, fileWrite2))	
	thread1.start()
	thread2.start()
	thread1.join()
	thread2.join()	
