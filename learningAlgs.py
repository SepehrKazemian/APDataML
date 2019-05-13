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
import dataManipulation as dataMan
from sklearn.cluster import KMeans


class learningAlgs:


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
		print("here")
			
	def threadCall(self):
		#kList = [60, 30, 20, 15, 10, 5]
		methodAlgs = ["NeuralNetwork", "NaiveBayes", "LogisticRegression"]
		kList = [20]
		
		bestError = []
#		hiddenLayers = [1, 5, 10]
		hiddenLayers = [1]
	#	regularizer = ['l1', 'l2']
		regularizer = ['l1']
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
			for x in range(len(hiddenLayers)):	
				NNError = self.NeuralNetwork(self.Xtrain, self.Ytrain, self.Xtest, self.Ytest, hiddenLayers[x])
				print("NeuralNetwork accuracy is: " + str(NNError))
			
			# for x in range(len(regularizer)):
				# print(self.Ytest)
				# print(self.Ytrain)
				# print(np.unique(self.Ytest))
				# print(np.unique(self.Ytrain))
				# LRError = self.LogisticRegression(self.Xtrain, self.Ytrain, self.Xtest, self.Ytest, regularizer[x])
				# print("LinearRegression accuracy is: " + str(LRError))
			
			# NBError = self.NaiveBayes(self.Xtrain, self.Ytrain, self.Xtest, self.Ytest)
			# print("NaiveBayes accuracy is: " + str(NBError))			
			
		
		
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

	#create two threads to find the testing and training files
	#first it should check whether we have the analyzed dataset or not
	def makeDataSet(self, k):
		fileNameTest = "40017ad6b2e0test"
		fileNameTrain = "40017ad6b2e0train"
		print("creating threads\n")
		thread1 = Thread( target = self.pickleChecker, args = (fileNameTest, k,))
		thread2 = Thread( target = self.pickleChecker, args = (fileNameTrain, k,) )
		thread1.start()
		thread2.start()
		thread1.join()
		thread2.join()
		
	#check whether we have the analyzed dataset with feature engineering or not.
	#if so, run the algorithm, otherwise check whether we have the the processed
	#dataset or not
	def pickleChecker(self, fileName, k):
	
		print("checking pickles\n")
		model = "KMeansClassification"
		timerInMinute = int(k)
		numberOfSamples = timerInMinute * 12 #60 seconds / 5 seconds per sample
		
		#check pickle existed or not, if we have it, for sure we had splitted the data
		#otherwise we have to do it, and save it. Returns True or False
		fileNameAns = self.functionCall(fileName, timerInMinute, numberOfSamples)
		
		#if we don't have the pickles search for csv file of data
		print(fileName + " in plotting func\n")
		if fileNameAns == False:

			print("we do not have the pickles, lets check processed data\n")
			CSVStat, CU, Time = self.csvChecker(fileName, model)
			
			#if we have csv file we have to making it ready for feature engineering
			#othewise we have to process the raw data
			if CSVStat == False:
				print("we do not have the processed data, going for processing\n")
				if model == "KMeansClassification":
					CU, Time = dataMan.dataSplitting(fileName, 0, "k")
				elif model == "manualClassification":
					CU, Time = dataMan.dataSplitting(fileName, 0, "man")
				print("processing finished\n")
				
			print("doing feature engineering\n")
			self.createTimeSplitter(timerInMinute)
			print(np.unique(CU))
			self.dataPoint(numberOfSamples, timerInMinute, fileName, CU, Time)
			

	#checking the whether we processed the raw data previously or not
	def csvChecker(self, fileName, model, address):
		pathFile = address + "/CSV/" + fileName + ".csv"
		print(str(fileName) + " is in csvChecker\n")
		if os.path.isfile(pathFile) == True:
			print("we have the csv file: pulling out data\n")
			#read the data from csv file
			headers = ['col1', 'CU', 'time']
			parse_date = ['time']
			data = pandas.read_csv(pathFile, header = None, names = headers, parse_dates = parse_date)
			
			#pulling the time data out
			timeArr = data["time"]
			
			#pulling CU data out and convert it to a classification or non classification problem
			chanUtil = data[["CU"]].values
			if model == "manualClassification":
				print("doing manual clustering\n")
				for i in range(len(chanUtil)):
					chanUtil[i] = dataMan.normalClassification(float(chanUtil[i]))
			
			elif model == "KMeansClassification":
				print("doing K-means clustering\n")
				chanUtil = dataMan.clusteringKMeans(chanUtil)
					
			
			chanUtil = chanUtil.reshape((len(chanUtil)))
			print(chanUtil)
			print(np.unique(chanUtil))
			
			print("returning the processed raw data back\n")
			time.sleep(1)
			return True, chanUtil, timeArr
		
		else:
			print("processed raw data is not available\n")
			return False, False, False
			
			
	def seperatedCsvChecker(self, fileName, model, address):
		
		counter = 0
		pathFile = address + "/CSV/" + fileName + "-" + str(counter) + ".csv"
		arr = []
		size = []
		while os.path.isfile(pathFile) == True:
			counter += 1
			print(str(pathFile) + " is in csvChecker\n")
			statinfo = os.stat(pathFile)
			arr.append(pathFile)
			size.append(statinfo.st_size)
			pathFile = address + "/CSV/" + fileName + "-" + str(counter) + ".csv"
		
		pathFile = arr[np.argmax(size)]
		print("selected file is: " + pathFile)
		
		if os.path.isfile(pathFile) == True:
			print("we have the csv file: pulling out data\n")
			#read the data from csv file
			headers = ['col1', 'CU', 'time']
			parse_date = ['time']
			data = pandas.read_csv(pathFile, header = None, names = headers, parse_dates = parse_date)
			
			return True, data
		
		else:
			print("processed raw data is not available\n")
			return False, False, False			
			
	
	def convertTime(self, value):
		dates = {date:pd.to_datetime(date) for date in value.unique()}
		return value.map(dates)
	
	
	#looking for pickles (engineered feature matrix) to load
	def functionCall(self, fileName, timerInMinute, numberOfSamples):
		print("looking for pickles (engineered feature matrix) to load\n")
		fileNameSearchX = "X" + str(fileName) + str(timerInMinute) + ".txt"
		fileNameSearchY = "Y" + str(fileName) + str(timerInMinute) + ".txt"
		
		if (os.path.isfile(fileNameSearchX) == True) and (os.path.isfile(fileNameSearchY) == True):
			print("pickles have been found\n")
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
			print("pickles are not found\n")
			return False
			
	
		
	#we are optimizing the process by creating the numpy arrays to just
	#replace the new data instead of adding them to array
	def createTimeSplitter(self, timeSplitter):
		print("making the processed data ready for feature engineering")
		splits = int((24 * 60) / (int(timeSplitter)))
		self.timeSplit = np.zeros(splits)
		
		self.days = np.zeros(7)
	
	#here we are making the features and targets for training and testing
	def dataPoint(self, numberOfSamples, timerInMinute, fileName, chanUtil, timeArr):
		print("doing the feature engineering")
		Ypred = 0
		time.sleep(10)
		
		# print(timeArr[0])
		# print(timeArr[240])
		# print(type(a[0]))
		# print(a[0].day())
		# print(timeArr[0].hour)
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
			
			
			Ypred = chanUtil[i + numberOfSamples + 1]
			# if chanUtil[i + numberOfSamples + 1] == 0:
				# Ypred = 0
			# elif chanUtil[i + numberOfSamples + 1] == 0.25:
				# Ypred = 1
			# elif chanUtil[i + numberOfSamples + 1] == 0.5:
				# Ypred = 2
			# elif chanUtil[i + numberOfSamples + 1] == 0.75:
				# Ypred = 3


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
	obj = learningAlgs()
	obj.threadCall()
		
