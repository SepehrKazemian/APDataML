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

class alligningTime:
	def __init__(self):
		self.set1 = set()
		self.set2 = set()

	def alignTime(self, fileName, setName):
		#print(fileName)
		#print(setName)
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
				timer = line[13:23]

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
					#currTimeStamp = time.mktime(datetime.datetime.strptime(stringTime, "%m/%d/%Y %H:%M:%S").timetuple())
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
				timer = line[13:23]

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

	address = input("give the channel seperation folder files address: ")
	
	files = os.listdir(address)
	for i in range(len(files)):
		if ".txt" not in files[i]:
			del files[i]
	
	
	chan1Alligning = []
	chan6Alligning = []
	chan11Alligning = []
	for i in range(len(files)):
		if "CH11" in files[i]:
			chan11Alligning.append(files[i])
		
		elif "CH1" in files[i]:
			chan1Alligning.append(files[i])
		
		elif "CH6" in files[i]:
			chan6Alligning.append(files[i])
			


	obj = alligningTime()
	
	allignedDirectory = address + "/alligned/"
	if os.path.isdir(allignedDirectory) == False:
		os.system("mkdir " + str(allignedDirectory))
	
	print(files)
	print(chan1Alligning)
	print(chan6Alligning)
	print(chan11Alligning)
	
	#number of files for chan1, 6, 11 wont be the same as some of the AP might not worked on that specific channel when we were monitoring them
	for i in range(len(chan1Alligning)):
		for j in range(i + 1, len(chan1Alligning)):
			obj.set1 = set()
			obj.set2 = set()
			fileName1 = address + "/" + chan1Alligning[i]
			fileName2 = address + "/" + chan1Alligning[j]
			thread1 = threading.Thread(target = obj.alignTime, args = (fileName1, obj.set1))
			thread2 = threading.Thread(target = obj.alignTime, args = (fileName2, obj.set2))
			thread1.start()
			thread2.start()
			thread1.join()
			thread2.join()
			unitedSet = (obj.set1 & obj.set2)
			fileWrite1 = allignedDirectory + chan1Alligning[i][0:11] + "-" + chan1Alligning[j][0:11] + "-" + chan1Alligning[i][-3:] + ".txt"
			fileWrite2 = allignedDirectory + chan1Alligning[j][0:11] + "-" + chan1Alligning[i][0:11] + "-" + chan1Alligning[i][-3:] + ".txt"
			thread1 = threading.Thread(target = obj.writeTheAllignedFile, args = (fileName1, unitedSet, fileWrite1))
			thread2 = threading.Thread(target = obj.writeTheAllignedFile, args = (fileName2, unitedSet, fileWrite2))	
			thread1.start()
			thread2.start()
			thread1.join()
			thread2.join()


	obj.set1 = set()
	obj.set2 = set()
	for i in range(len(chan6Alligning)):
		for j in range(i + 1, len(chan6Alligning)):
			obj.set1 = set()
			obj.set2 = set()
			fileName1 = address + "/" + chan6Alligning[i]
			fileName2 = address + "/" + chan6Alligning[j]		
			thread1 = threading.Thread(target = obj.alignTime, args = (fileName1, obj.set1))
			thread2 = threading.Thread(target = obj.alignTime, args = (fileName2, obj.set2))
			thread1.start()
			thread2.start()
			thread1.join()
			thread2.join()
			unitedSet = (obj.set1 & obj.set2)
			fileWrite1 = allignedDirectory + chan6Alligning[i][0:11] + "-" + chan6Alligning[j][0:11] + "-" + chan6Alligning[i][-3:] + ".txt"
			fileWrite2 = allignedDirectory + chan6Alligning[j][0:11] + "-" + chan6Alligning[i][0:11] + "-" + chan6Alligning[i][-3:] + ".txt"
			thread1 = threading.Thread(target = obj.writeTheAllignedFile, args = (fileName1, unitedSet, fileWrite1))
			thread2 = threading.Thread(target = obj.writeTheAllignedFile, args = (fileName2, unitedSet, fileWrite2))	
			thread1.start()
			thread2.start()
			thread1.join()
			thread2.join()

	obj.set1 = set()
	obj.set2 = set()
	for i in range(len(chan11Alligning)):
		for j in range(i + 1, len(chan11Alligning)):
			obj.set1 = set()
			obj.set2 = set()
			fileName1 = address + "/" + chan11Alligning[i]
			fileName2 = address + "/" + chan11Alligning[j]			
			thread1 = threading.Thread(target = obj.alignTime, args = (fileName1, obj.set1))
			thread2 = threading.Thread(target = obj.alignTime, args = (fileName2, obj.set2))
			thread1.start()
			thread2.start()
			thread1.join()
			thread2.join()
			unitedSet = (obj.set1 & obj.set2)
			fileWrite1 = allignedDirectory + chan11Alligning[i][0:11] + "-" + chan11Alligning[j][0:11] + "-" + chan11Alligning[i][-3:] + ".txt"
			fileWrite2 = allignedDirectory + chan11Alligning[j][0:11] + "-" + chan11Alligning[i][0:11] + "-" + chan11Alligning[i][-3:] + ".txt"
			thread1 = threading.Thread(target = obj.writeTheAllignedFile, args = (fileName1, unitedSet, fileWrite1))
			thread2 = threading.Thread(target = obj.writeTheAllignedFile, args = (fileName2, unitedSet, fileWrite2))	
			thread1.start()
			thread2.start()
			thread1.join()
			thread2.join()			

			
	
	# print(obj.set2)
	# obj.crossCor()
