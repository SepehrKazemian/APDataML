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

class plottData:


	def __init__(self):
		self.Xlearn = []
		self.Ylearn = []
		self.Xtest = []
		self.Ytest = []
		self.Ytrain = []
		self.timeArrStamp = []
		self.Xtrain =[]
		self.from_zone = tz.gettz('UTC')
		self.to_zone = tz.gettz('America/Edmonton')
		now_timestamp = time.time()
		self.offset = datetime.datetime.fromtimestamp(now_timestamp) - datetime.datetime.utcfromtimestamp(now_timestamp)
		self.arr = []
		for i in range(30):
			for j in range(60):
				self.arr.append("2018-10-23 13:" + str(i) + ":" + str(j))
			
	def threadCall(self):


		#kList = [60, 30, 20, 15, 10, 5]
		methodAlgs = ["NeuralNetwork", "NaiveBayes", "LogisticRegression"]
		kList = [0]
		
		bestError = []
		hiddenLayers = [1, 5, 10]
		regularizer = ['l1', 'l2']
		minErrorNN = 0
		minErrorLR = 0
		
		bestHL_k = 0
		bestLR_k = 0
		bestHL_kIndex = float('inf')
		bestLR_kIndex = float('inf')
		NN_k = float('inf')
		LR_k = float('inf')
		
		for i in range(len(kList)):
			# print("k is: " + str(kList[i]))
			# self.makeDataSet(kList[i])
			
			#NeuralNetwork testing with cross validation
			# bestHLIndex = float('inf')
			# bestRegIndex = float('inf')
			# for x in range(len(hiddenLayers)):
				# errorVal = self.CrossValidation("NeuralNetwork", hiddenLayers[x])
				# print(hiddenLayers[x])
				# print(errorVal)
				# if minErrorNN < errorVal:
					# minErrorNN = errorVal
					# bestHLIndex = x
			
			# if bestHL_k < minErrorNN:
				# bestHL_k = minErrorNN
				# bestHL_kIndex = bestHLIndex
				# NN_k = kList[i]
			# print("min NN_K is: " + str(NN_k) + " with value of: " + str(hiddenLayers[bestHL_kIndex]))
				
			# #LogisticRegression testing with cross validation			
			# for x in range(len(regularizer)):
				# errorVal = self.CrossValidation("LogisticRegression", regularizer[x])
				# print("our regularizer: " + str(regularizer[x]))
				# print("our error of reg is: " + str(errorVal))
				# if minErrorLR < errorVal:
					# minErrorLR = errorVal
					# bestRegIndex = x
			
			# if bestLR_k < minErrorLR:
				# bestLR_k = minErrorLR
				# bestLR_kIndex = bestRegIndex
				# LR_k = kList[i]

			# #NaiveBayes testing with cross validation			
			# bestNBIndex = float('inf')
			# bestRegIndex = float('inf')
			# errorVal = self.CrossValidation("NaiveBayes", None)
			# print("final F1-score is: " + str(errorVal))
			# if minErrorNN < errorVal:
				# minErrorNN = errorVal
				# bestNBIndex = kList[i]

			print("\n\n\n\nk is: " + str(kList[i]))
			NNError = 0
			LRError = 0
			NBError = 0
			
			self.makeDataSet(kList[i])
			if kList[i] != 0:
				for x in range(len(hiddenLayers)):	
					NNError = self.NeuralNetwork(self.Xtrain, self.Ytrain, self.Xtest, self.Ytest, hiddenLayers[x])
					print("NeuralNetwork accuracy is: " + str(NNError))
				
				for x in range(len(regularizer)):
					LRError = self.LogisticRegression(self.Xtrain, self.Ytrain, self.Xtest, self.Ytest, regularizer[x])
					print("LinearRegression accuracy is: " + str(LRError))
				
				NBError = self.NaiveBayes(self.Xtrain, self.Ytrain, self.Xtest, self.Ytest)
				print("NaiveBayes accuracy is: " + str(NBError))			
			
		
		
		# self.makeDataSet(10)
		# x = np.array_split(self.Xtest, 10)
		# y = np.array_split(self.Ytest, 10)
		# lr = []
		# for i in range(10):
			# lr.append(self.LogisticRegression(self.Xtrain, self.Ytrain, x[i], y[i], "l1"))
		# print(lr)
			
		# self.makeDataSet(60)
		# x = np.array_split(self.Xtest, 10)
		# y = np.array_split(self.Ytest, 10)
		# nn = []
		# for i in range(10):
			# nn.append(self.NeuralNetwork(self.Xtrain, self.Ytrain, x[i], y[i], 5))
		# print(nn)

		# nn = [0.8662543486664089, 0.9018167761886355, 0.947816003092385, 0.89563200618477, 0.8260533436412834, 0.8790104367993815, 0.9783533049864708, 0.9218870843000774, 0.8604021655065739, 0.848414539829853]
		# lr = [0.9344135802469136, 0.9344135802469136, 0.9633487654320988, 0.9189814814814815, 0.8460648148148148, 0.8819444444444444, 0.9756944444444444, 0.9868776534156696, 1.0, 1.0]
		# nb = [0.8670274449168921, 0.7959025898724391, 0.9404715887127948, 0.8218013142636258, 0.7108620023192888, 0.8790104367993815, 0.9783533049864708, 0.9218870843000774, 0.8604021655065739, 0.848414539829853]
		
		
		# self.makeDataSet(60)
		# x = np.array_split(self.Xtest, 10)
		# y = np.array_split(self.Ytest, 10)
		# nb = []
		# for i in range(10):
			# nb.append(self.NaiveBayes(self.Xtrain, self.Ytrain, x[i], y[i]))


			
		# print(nb)
			# print("min K is: " + str(kList[i]) + " with value of: " + str(minErrorNN))
		# ln = stats.ttest_ind(nn,lr)
		# lb = stats.ttest_ind(lr,nb)
		# bn = stats.ttest_ind(nn,nb)
		
		# print(ln, lb, bn)
		
		# ln = stats.ttest_ind(lr,nn)
		# lb = stats.ttest_ind(lr,nb)
		# bn = stats.ttest_ind(nb,nn)
		
		# print(ln, lb, bn)
				

		
			
#			self.NaiveBayes()
			#self.NeuralNetwork(self.Xtrain, self.Ytrain, self.Xtest, self.Ytest)

	def makeDataSet(self, k):
		fileNameTest = "40017ad6b2e0test"
		fileNameTrain = "40017ad6b2e0train"	
		thread1 = Thread( target = self.plotting, args = (fileNameTest, k,))
		thread2 = Thread( target = self.plotting, args = (fileNameTrain, k,) )
		thread1.start()
		thread2.start()
		thread1.join()
		thread2.join()
		
	def plotting(self, fileName, k):
		#print("plotting")
#		fileName = fileNameArr[i]
		model = "classification"
		timerInMinute = int(k)
		numberOfSamples = timerInMinute * 12 #60 seconds / 5 seconds per sample
		
		# check pickle existed or not, if we have it, for sure we had splitted the data
		#otherwise we have to do it, and save it
		fileNameAns = self.functionCall(fileName, timerInMinute, numberOfSamples)
		
		#if we don't have the pickles search for csv file of data
		print(fileName + " in plotting func")
		if fileNameAns == False:
		

			CSVStat, CU, Time = self.csvChecker(fileName, model)
			
			#if we have csv file:
			if CSVStat == True:
				if k != 0:
					self.createTimeSplitter(timerInMinute)
					self.dataPoint(numberOfSamples, timerInMinute, fileName, CU, Time)
				
				elif k == 0:
					self.naiveCalculate(CU, Time)
				
			else:
				self.dataSplitting(fileName, timerInMinute, numberOfSamples)
				
			
			
			

	def csvChecker(self, fileName, model):
		pathFile = "node1/extractedData/" + fileName + ".csv"
		print(str(fileName) + " is in csvChecker")
		if os.path.isfile(pathFile) == True:
			#read the data from csv file
			data = pandas.read_csv(pathFile, parse_dates=["time"])
			
			#pulling the time data out
			#data["time"] = self.convertTime(data["time"])
			# print(data["time"])
			# print(type(data["time"].values))
			timeArr = data["time"]
			# print("ggggggggg")
			# print(timeArr.shape)
			# #timeArr = timeArr.astype(object)
			# print(timeArr[0])
			# print(timeArr[0].hour)
			# timeArr = timeArr.reshape((len(timeArr), 1))
			# print(type(timeArr[0]))
			# print(timeArr.shape)
			
			#pulling CU data out and convert it to a classifier or non classifier
			if model == "classification":
				chanUtil = data[["CU"]].values
				for i in range(len(chanUtil)):
					chanUtil[i] = self.normalClassification(float(chanUtil[i]))
			else:
				chanUtil = data[["CU"]].values
			chanUtil = chanUtil.reshape((len(chanUtil)))
			#print(chanUtil.shape)
			
			return True, chanUtil, timeArr
		
		else:
			return False, False, False
			
	
	def convertTime(self, value):
		dates = {date:pd.to_datetime(date) for date in value.unique()}
		return value.map(dates)
	
	
	def functionCall(self, fileName, timerInMinute, numberOfSamples):
		#print(str(fileName) + " is in functionCall")
		fileNameSearchX = "X" + str(fileName) + str(timerInMinute) + ".txt"
		fileNameSearchY = "Y" + str(fileName) + str(timerInMinute) + ".txt"
		
		if (os.path.isfile(fileNameSearchX) == True) and (os.path.isfile(fileNameSearchY) == True):
			with open(fileNameSearchX, "rb") as f:
				if "train" not in fileName:
					self.Xtest = pickle.load(f)
				else:
					self.Xtrain = pickle.load(f)

			with open(fileNameSearchY, "rb") as f:
				if "train" not in fileName:
					self.Ytest = pickle.load(f)
				else:
					self.Ytrain = pickle.load(f)
			
			return True
		
		else:
			return False
			
		# else:
			# self.dataSplitting(fileName)
			# self.createTimeSplitter(timerInMinute)
			# self.dataPoint(numberOfSamples, timerInMinute, fileName)
		
	def dataSplitting(self, fileName, timeSplitter, numberOfSamples):
		#print(fileName + " in datasplitting func")

		secondCounter = 0
		prevTime = ""
		curTime = ""
		cu = ""
		lineCounter = 0
		signalVal = 0
		counter = 0
		preMaxVal = 0
		fileWrite = "node1/extractedData/" + str(fileName)
		#print("here")
		csvArrTime = []
		csvArrCU = []
		chanUtil = []
		timeArr = []
		with open(fileWrite) as fp:
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
				else:
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
				
				#channel changes the string positions
				if channel == "1" or channel == "6":
					for i in range(38,41):
						if line[i] != " ":
							cu += line[i]
						else:
							break
				elif channel == "11":
					for i in range(39,42):
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
						prevTime = currTimeStamp
						maxVal = int(cu)
						counter = 1
					
					elif currTimeStamp - prevTime >= 5:
						
						
						while int((currTimeStamp - prevTime) / 5) > 0:
							currTimeStampUTC = datetime.datetime.fromtimestamp(prevTime).strftime('%Y-%m-%d %H:%M:%S')
							currTimeStampUTC = datetime.datetime.strptime(currTimeStampUTC, '%Y-%m-%d %H:%M:%S')
							central = self.offset + currTimeStampUTC
							
							#it seems LR in sickit learn cannot work with datetime, so we have to reconvert it to
							#values
							centralTimeStamp = time.mktime(datetime.datetime.strptime(str(central), "%Y-%m-%d %H:%M:%S").timetuple())
							
							csvArrTime.append(central)
#							self.timeArrStamp.append(centralTimeStamp)
							timeArr.append(central)
								
							csvArrCU.append(maxVal/255)
							chanUtil = np.append(chanUtil, self.normalClassification(maxVal/255))
					#		self.chanUtil = np.append(self.chanUtil, (maxVal/255))
							if maxVal > 255:
								print("bilakh " + str(maxVal))
							prevTime = prevTime + 5
							
						maxVal = 0
						
					
					elif currTimeStamp - prevTime < 5:
						if maxVal < int(cu):
							maxVal = int(cu)
					

				except ValueError:
					print("channel utilization error")
		#print(len(chanUtil))
		#print(len(timeArr))
		
		dataFile = pandas.DataFrame(data = {"time": csvArrTime, "CU": csvArrCU})
		dataFile.to_csv("node1/extractedData/" + str(fileName) + ".csv", sep = ",", index = False)
		
		#print(fileName)
		#print("csv file is added")
		#print("now we have finished reading data")
		
		self.createTimeSplitter(timeSplitter)
		self.dataPoint(numberOfSamples, timeSplitter, fileName, chanUtil, timeArr)

		
		
	def normalClassification(self, value):
		if (value) < 0.4:
			return "0"
		elif (value) < 0.6:
			return "0.25"
		elif (value) < 0.8:
			return "0.5"		
		else:
			return "0.75"		
	
		
	def createTimeSplitter(self, timeSplitter):
		#print(timeSplitter)
		splits = int((24 * 60) / (int(timeSplitter)))
		self.timeSplit = np.zeros(splits)
		
		self.days = np.zeros(7)
	
	
	def dataPoint(self, numberOfSamples, timerInMinute, fileName, chanUtil, timeArr):
		#print(fileName + " in dataPoint func")
		Ypred = 0
		time.sleep(10)
		counter = 0
		zeros = np.zeros(numberOfSamples + int(60 / timerInMinute * 24) + 7)
		numberOfFeatures = numberOfSamples + int(60 / timerInMinute * 24) + 7
		print(numberOfFeatures, numberOfSamples)
		
		sampleCounter = 0
		XZeros = []
		for i in range(0, len(timeArr) - numberOfFeatures, 1):
			#to increase the efficiency, we do not want to append, so we make the zeros of the array
			#then in the next step we will replace elements of the array by their indexes
			sampleCounter += 1
		
		#print("aaaa")
		print(sampleCounter)
		print(chanUtil.shape)
		print(len(timeArr) - numberOfSamples)
			
		XZeros = np.zeros([sampleCounter, numberOfFeatures])
			
			
		if "train" in fileName:	
			self.Xtrain = np.zeros([sampleCounter, numberOfFeatures])
			self.Ytrain = np.zeros([sampleCounter,1])
		
			
		if "train" not in fileName:
			self.Xtest = np.zeros([sampleCounter, numberOfFeatures])
			self.Ytest = np.zeros([sampleCounter,1])
		

		for i in range(0, len(timeArr) - numberOfFeatures, 1): #start to iterate from zero till the end and slide every minute forward
			
			#we want to keep some lists untouched
			Xarr = np.copy(zeros)
			timePartition = np.copy(self.timeSplit)
			dayPartition = np.copy(self.days)
			
	#		print(Xarr.shape)
	#		print(chanUtil.shape)
			Xarr[0 : numberOfSamples] = chanUtil[i : i + numberOfSamples] #replacing elements instead of appending them

			#which of the 20 minutes, the first element existed on
			firstElement = timeArr[i].hour * int(60 / timerInMinute) + math.ceil(timeArr[i].minute / int(timerInMinute))
			
			#which of the 20 minutes, the last element existed on
			lastElement = timeArr[i + numberOfSamples].hour * int(60 / timerInMinute) + math.ceil(timeArr[i + numberOfSamples].minute / int(timerInMinute))
			
			#our array starts from 1 so we have to use one-hot on element - 1
			#we replace the value insede the zeros array copy that we have
	#		print(i , numberOfSamples, firstElement)
			Xarr[numberOfSamples + firstElement - 1] = 1
			Xarr[numberOfSamples + lastElement - 1] = 1

			#finding the first and last element weekday
			firstDay = timeArr[i].weekday()
			lastDay = timeArr[i + numberOfSamples].weekday()
			
			#adding values. e.g. if 20 min: i + 240 ==> values
			# + we have 72 of 20 minutes + 7 values for days
			#if days are starting from 0 to 6, i + 240 + 72 + 0 is the value of the day
			Xarr[numberOfSamples + int(60 / timerInMinute * 24) + firstDay] = 1
			Xarr[numberOfSamples + int(60 / timerInMinute * 24) + lastDay] = 1			
			
			if "train" in fileName:	
				self.Xtrain[counter] = Xarr
		#		print(self.Xtrain.shape)
		#		print(Xarr.shape)
			
				
			if "train" not in fileName:
				self.Xtest[counter] = Xarr
		#		print(self.Xtest.shape)
		#		print(Xarr.shape)
			
			
			#Ypred = self.chanUtil[i + 241]
			if chanUtil[i + numberOfSamples + 1] == 0:
				Ypred = 0
			elif chanUtil[i + numberOfSamples + 1] == 0.25:
				Ypred = 1
			elif chanUtil[i + numberOfSamples + 1] == 0.5:
				Ypred = 2
			elif chanUtil[i + numberOfSamples + 1] == 0.75:
				Ypred = 3


			if "train" in fileName:
				self.Ytrain[counter] = Ypred

			
			if "train" not in fileName:
				self.Ytest[counter] = Ypred

			
			counter += 1

#			print(self.Ylearn)
#			print(self.Ylearn.shape)
#			print(self.Xlearn)
#			print(self.Xlearn.shape)
			
#		print(numberOfSamples)
		if "train" not in fileName:
			XmatrixName = "X" + str(fileName) + str(timerInMinute) + ".txt"
			with open(XmatrixName, 'wb') as f:
				pickle.dump(self.Xtrain, f)
					
			YmatrixName = "Y" + str(fileName) + str(timerInMinute) + ".txt"
			with open(YmatrixName, 'wb') as f:
				pickle.dump(self.Ytrain, f)	
				
		if "train" in fileName:
			XmatrixName = "X" + str(fileName) + str(timerInMinute) + ".txt"
			with open(XmatrixName, 'wb') as f:
				pickle.dump(self.Xtest, f)
					
			YmatrixName = "Y" + str(fileName) + str(timerInMinute) + ".txt"
			with open(YmatrixName, 'wb') as f:
				pickle.dump(self.Ytest, f)	
		

		
	def naiveCalculate(self, chanUtil, timeArr):
		print("here i am")
		chanUtilPred = np.zeros(len(chanUtil))
		equalCount = 0
		
		for i in range(len(chanUtil) - 1):
			chanUtilPred[i + 1] = chanUtil[i]
			
		for i in range(len(chanUtil)):
			if chanUtilPred[i] == chanUtil[i]:
				equalCount += 1
		
		print(equalCount / len(chanUtil))
		
		
	def CrossValidation(self, name, extra):
		print(name)
		error = float('inf')
		kf = KFold(n_splits = 5)
		
		arrAvgError = []

		for train_index, test_index in kf.split(self.Xtrain):
			print("TRAIN:", train_index, "TEST:", test_index)
			X_train, X_test = self.Xtrain[train_index], self.Xtrain[test_index]
			y_train, y_test = self.Ytrain[train_index], self.Ytrain[test_index]
			if name == "NeuralNetwork":
				error = self.NeuralNetwork(X_train, y_train, X_test, y_test, extra)
			elif name == "NaiveBayes":
				error = self.NaiveBayes(X_train, y_train, X_test, y_test)
			elif name == "LogisticRegression":
				error = self.LogisticRegression(X_train, y_train, X_test, y_test, extra)
				
			arrAvgError.append(error)
		
		return np.mean(arrAvgError)
		
	
	def LogisticRegression(self, Xtrain, Ytrain, Xtest, Ytest, reg):
		#Xtrain, Ytrain = load_iris(return_X_y=True)
		solvers = ""
		if reg == "l2":
			solvers = "lbfgs"
		else:
			solvers = "saga"

		clf = LogisticRegression(penalty = reg, max_iter = 100000, random_state = 0, solver = solvers , multi_class = 'multinomial')
#		print(Xtrain.shape)
		Ytrain = Ytrain.reshape(len(Ytrain),)
#		print(Ytrain.shape)
		clf.fit(Xtrain[:, :], Ytrain[:])
		FinalXtest = Xtest[:,:]
		Ypred = clf.predict(FinalXtest)
		Ypred = Ypred.reshape((len(Ypred),1))
		
		errorW = f1_score(Ytest, Ypred, average='weighted', labels=np.unique(Ypred))
		errorMa = f1_score(Ytest, Ypred, average='macro', labels=np.unique(Ypred))
		errorMi = f1_score(Ytest, Ypred, average='micro', labels=np.unique(Ypred))
		# print("errorW is: " + str(errorW))		
		# print("errorMa is: " + str(errorMa))		
		# print("errorMi is: " + str(errorMi))		
		

		recallW = recall_score(Ytest, Ypred, average='weighted')
		recallMa = recall_score(Ytest, Ypred, average='macro')
		recallMi = recall_score(Ytest, Ypred, average='micro')
		# print("recallW is: " + str(recallW))		
		# print("recallMa is: " + str(recallMa))
		# print("recallMi is: " + str(recallMi))		
		
		precisionW = precision_score(Ytest, Ypred, average='weighted')
		precisionMa = precision_score(Ytest, Ypred, average='macro')
		precisionMi = precision_score(Ytest, Ypred, average='micro')
		# print("precisionW is: " + str(precisionW))		
		# print("precisionMa is: " + str(precisionMa))
		# print("precisionMi is: " + str(precisionMi))			
		
		acc = accuracy_score(Ytest, Ypred)
		print("acc is: " + str(acc))		
		return acc
		
		
	def NeuralNetwork(self, Xtrain, Ytrain, Xtest, Ytest, HLNumber):	
#		clf = GaussianNB()
		if HLNumber == 1:
			clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1)
			
		if HLNumber == 5:
			clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 100, 100, 100, 100), random_state=1)

		if HLNumber == 10:
			clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100, 100), random_state=1)

		#print(a.T)
	#	print(type(Xtrain))
		data_X_train = Xtrain[:, :] #a.reshape(-1,1)

		#diabetes_X_test = b.T#.reshape(-1,1)

		# Split the targets into training/testing sets
		data_y_train = Ytrain[:]

		#diabetes_y_test = ([42, 44, 46.1, 48.2, 50.3])

	#	print(data_y_train.shape)
	#	print(data_X_train.shape)
		data_y_train = data_y_train.reshape(len(data_y_train),)
		clf.fit(data_X_train, data_y_train)
		
		# Create linear regression object
		#regr = linear_model.LinearRegression()

		# Train the model using the training sets
		#regr.fit(data_X_train, data_y_train)
		
	#	print(Xtest)
		FinalXtest = Xtest[:,:]
#		print(self.Xtest[5300,:])
#		print(self.Xtest[5466,:])
#		print(self.Xtest[5469,:])
#		print(self.Xtest)
        # Make predictions using the testing set
		#diabetes_y_pred = regr.predict(diabetes_X_test)
		#print(diabetes_y_pred)

		Ypred = clf.predict(FinalXtest)
		
        # The coefficients
	#	print('Coefficients: \n', regr.coef_)
        # The mean squared error
		# y = np.where(diabetes_y_pred < 0)
		# x = np.where(diabetes_y_pred > 1)
		# print(x)
		Ypred = Ypred.reshape((len(Ypred),1))
	#	print(Ytest.shape)
	#	print(Ypred.shape)
	#	diabetes_y_pred[x] = 1
	#	diabetes_y_pred[y] = 0
		error = f1_score(Ytest, Ypred, average='weighted', labels=np.unique(Ypred))	
		acc = accuracy_score(Ytest, Ypred)
		
		errorW = f1_score(Ytest, Ypred, average='weighted', labels=np.unique(Ypred))
		errorMa = f1_score(Ytest, Ypred, average='macro', labels=np.unique(Ypred))
		errorMi = f1_score(Ytest, Ypred, average='micro', labels=np.unique(Ypred))
		
		recallW = recall_score(Ytest, Ypred, average='weighted')
		recallMa = recall_score(Ytest, Ypred, average='macro')
		recallMi = recall_score(Ytest, Ypred, average='micro')
		
		precisionW = precision_score(Ytest, Ypred, average='weighted')
		precisionMa = precision_score(Ytest, Ypred, average='macro')
		precisionMi = precision_score(Ytest, Ypred, average='micro')

		# print("acc is: " + str(acc))
		
		# print("recallW is: " + str(recallW))		
		# print("recallMi is: " + str(recallMi))	
		
		# print("precisionW is: " + str(precisionW))		
		# print("precisionMi is: " + str(precisionMi))		

		# print("fscoreW is: " + str(errorW))		
		# print("fscoreW is: " + str(errorMi))



		return acc	
		# print("Mean squared error: %.2f" % (self.Ytest - diabetes_y_pred))
		# print(np.mean(np.square(self.Ytest-diabetes_y_pred)))
		# print(np.abs(regr.coef_).argsort()[0,-40:])
		# #print(np.min(diabetes_y_pred))
		# print(np.max(diabetes_y_pred))
		# print(x)
		
        # Explained variance score: 1 is perfect prediction


	def NaiveBayes(self, Xtrain, Ytrain, Xtest, Ytest):
		warnings.filterwarnings('ignore')
		
		clf = GaussianNB()

		data_X_train = Xtrain[:, :] #a.reshape(-1,1)


		# Split the targets into training/testing sets
		data_y_train = Ytrain[:]

		data_y_train = data_y_train.reshape(len(data_y_train),)
		clf.fit(data_X_train, data_y_train)
		
		#print(self.Xtest)
		FinalXtest = Xtest[:,:]

		Ypred = clf.predict(FinalXtest)
		Ypred = Ypred.reshape((len(Ypred),1))

		errorW = f1_score(Ytest, Ypred, average='weighted', labels=np.unique(Ypred))
#		print("errorW is: " + str(errorW))		
		
		

		recallW = recall_score(Ytest, Ypred, average='weighted', labels=np.unique(Ypred))
#		print("recallW is: " + str(recallW))		
	
		
		precisionW = precision_score(Ytest, Ypred, average='weighted', labels=np.unique(Ypred))
#		print("precisionW is: " + str(precisionW))		
		
		
		acc = accuracy_score(Ytest, Ypred)
#		print("acc is: " + str(acc))		
		return acc


if __name__ == '__main__':
	obj = plottData()
	obj.threadCall()