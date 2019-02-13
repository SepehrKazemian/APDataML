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


class MarkovChain():
	def __init__(self):
		self.timeArr = []
		
		
	def timeChunks(self, time, cu, timerInMinute):
		print("chunking the data\n")
		numberOfTimeElements = int(60 / 5 * 20)
		cuValuesArr = np.zeros(shape=(int(len(time) / numberOfTimeElements), int(numberOfTimeElements)))
		timeValuesArr = np.zeros(shape=(int(len(time) / numberOfTimeElements), int(numberOfTimeElements)))
		arrCounter = 0
		timeSplitterCounter = 0
		while ((len(cu) - timeSplitterCounter) > numberOfTimeElements):
			cuValuesArr[arrCounter] = cu[timeSplitterCounter: timeSplitterCounter + numberOfTimeElements]
			timeValuesArr[arrCounter] = time[timeSplitterCounter: timeSplitterCounter + numberOfTimeElements]			
			timeSplitterCounter += numberOfTimeElements
			arrCounter += 1
			
		return cuValuesArr, timeValuesArr
		
		
	def pickleChecker(self, timerInMinute):
		
		print("checking pickles\n")
		
		fileName = "./node1/markovianTimePickles/cuTrans" + str(timerInMinute)
		if os.path.isfile(fileName) == True:
			with open(fileName, "rb") as f:
				cuTrans = pickle.load(f)
			
			print("checking transition probabilities\n")
			return cuTrans

		else:
			return None
		
		
	def transitionMatrix(self):
		
		timerInMinute = 20
		cuTrans = self.pickleChecker(timerInMinute)
		if cuTrans == None:
			print("gathering processed data\n")
			a = plot.plottData()
			CU, Time = a.prepration()
			cuValuesArr, timeValuesArr = self.timeChunks(Time, CU, timerInMinute)
			self.transProbCalc(cuValuesArr, timeValuesArr, timerInMinute)


		
		
	def transProbCalc(self, cuValuesArr, timeValuesArr, timerInMinute):
		print("calculating transition probabilities\n")
		cuTrans = np.zeros(shape=(cuValuesArr.shape[0], 51, 51))
		for i in range(cuValuesArr.shape[0]):
			for j in range(cuValuesArr.shape[1] - 1):
				start = cuValuesArr[i, j] / 0.02
				next = cuValuesArr[i, j + 1] / 0.02
				cuTrans[i, math.floor(start), math.floor(next)] += 1
				
		fileName = "./node1/markovianTimePickles/cuTrans" + str(timerInMinute)
		with open(fileName, 'wb') as f:
			pickle.dump(cuTrans, f)
		
				
		# cuTrans = np.zeros((51,51))
		
		# for i in range(len(CU) - 1):
			# start = CU[i] / 0.02
			# next = CU[i + 1] / 0.02
			# cuTrans[math.floor(start), math.floor(next)] += 1
		# probability = np.true_divide(cuTrans, len(CU) - 1)
		
		# fileName = "transitionMatrix"
		# titleAP = "file name is : " + str(fileName)
		# print("yoooohooooooooooooooooooooooooooooooo")
		# trace = go.Heatmap(z = probability)
		# data = [trace]
		# offline.plot(data, fileName)
		
		
if __name__ == '__main__':
	obj = MarkovChain()
	obj.transitionMatrix()