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


class MarkovChain():
	def __init__(self):
		self.timeArr = []
		
	#chunking the time with the its CUs 
	def timeChunks(self, time, cu, timerInMinute):
		print("chunking the data\n")
		numberOfTimeElements = int(60 / 5 * 20) #time is for 20 minutes with each sample for each 5 seconds 
		cuValuesArr = np.zeros(shape=(int(len(time) / numberOfTimeElements), int(numberOfTimeElements)))
		timeValuesArr = np.zeros(shape=(int(len(time) / numberOfTimeElements), int(numberOfTimeElements)), dtype='U28')
		time = time.astype(str)
		print(type(time))
		print(time[0: 200])
		arrCounter = 0
		timeSplitterCounter = 0
		while ((len(cu) - timeSplitterCounter) > numberOfTimeElements):
			cuValuesArr[arrCounter] = cu[timeSplitterCounter: timeSplitterCounter + numberOfTimeElements]
			timeValuesArr[arrCounter] = time[timeSplitterCounter: timeSplitterCounter + numberOfTimeElements]
			timeSplitterCounter += numberOfTimeElements
			arrCounter += 1
			
		return cuValuesArr, timeValuesArr
		
		
	def pickleChecker(self, timerInMinute, fileName):
		
		print("checking pickles\n")
		
		pickleName = "./node1/markovianTimePickles/cuTrans" + str(fileName) + str(timerInMinute)
		if os.path.isfile(pickleName) == True:
			with open(pickleName, "rb") as f:
				cuTrans = pickle.load(f)
			
			pickleName = "./node1/markovianTimePickles/cuValuesArr" + str(fileName) + str(timerInMinute)
			with open(pickleName, 'rb') as f:
				cuValuesArr = pickle.load(f)
				
			pickleName = "./node1/markovianTimePickles/timeValuesArr" + str(fileName) + str(timerInMinute)
			with open(pickleName, 'rb') as f:
				timeValuesArr = pickle.load(f)
			
			print("checking transition probabilities\n")
			return cuTrans, cuValuesArr, timeValuesArr

		else:
			return None, None, None
		
		
	def transitionMatrix(self):
		
		plotterClass = plot.plottData()
		timerInMinute = 20
		fileName = ['500f8022b880', '500f80271400']
		
		CU_FileChunks = None
		
		for i in range(len(fileName)):
			cuTrans, cuValuesArr, timeValuesArr = self.pickleChecker(timerInMinute, fileName[i])
			if type(cuTrans) != np.ndarray:
				print("gathering processed data\n")
				CU, Time = plotterClass.prepration(fileName[i])
				print(CU[0])
				CU = CU/255
				print(CU[0])
				cuValuesArr, timeValuesArr = self.timeChunks(Time, CU, timerInMinute)
				cuTrans, cuValuesArr, timeValuesArr = self.transProbCalc(cuValuesArr, timeValuesArr, timerInMinute, fileName[i])

			if CU_FileChunks == None:
				CU_FileChunks = np.zeros(shape=(len(fileName), cuValuesArr.shape[0], cuValuesArr.shape[1]))
				CU_FileChunks[i] = cuValuesArr
			else:
				CU_FileChunks[i] = cuValuesArr
				
		#self.markovianSteadyState(cuTrans, cuValuesArr, timeValuesArr)
		self.crossCorrelation(CU_FileChunks)
				
		
	def crossCorrelation(self, CU_FileChunks):
		autoCor = np.zeros(shape=(CU_FileChunks.shape[1], CU_FileChunks.shape[2]))
		crossCor = np.zeros(shape=(CU_FileChunks.shape[1], CU_FileChunks.shape[2]))
		crossCorComparison = np.zeros(shape=(CU_FileChunks.shape[1], CU_FileChunks.shape[2]))
		
		print(CU_FileChunks.shape[1])
		time.sleep(1)
		for i in range(CU_FileChunks.shape[1]):
			autoCor[i] = signal.correlate(CU_FileChunks[0][i], CU_FileChunks[0][i], mode='same')
			crossCor[i] = signal.correlate(CU_FileChunks[0][i], CU_FileChunks[1][i], mode='same')
			crossCorComparison[i] = crossCor[i] - autoCor[i]
			if np.amax(np.absolute(crossCorComparison[i])) != 0:
				crossCorComparison[i] = crossCorComparison[i] / np.amax(np.absolute(crossCorComparison[i]))
				
		counter = []
		for i in range((CU_FileChunks.shape[1] * CU_FileChunks.shape[2]) + (100 * CU_FileChunks.shape[1])):
			counter.append(i)
		
		print("coutning")
		
		data = []
		for i in range(CU_FileChunks.shape[1]):
			for j in range(CU_FileChunks.shape[2]):
				data.append(crossCorComparison[i][j])
			#print(crossCor[i])
			time.sleep(3)
			for j in range(100):
				data.append(-10)
		print("coutning")
				
		
		trace = go.Scatter(
		    x = counter,
		    y = data,
		    mode = 'lines',
		    name = 'data'
		)
		data = [trace]
		offline.plot(data, filename = "crossCorrelationComparison.html")
		
	
	
		
	def markovianSteadyState(self, cuTrans, cuValuesArr, timeValuesArr):
		
		
		np.set_printoptions(threshold=np.inf)

		
		#normalizing data
		print(cuTrans.dtype)
		# cuTrans = cuTrans.astype(complex)
		# print(cuTrans.dtype)
		print("normalizing")
		ans = np.zeros(shape=(cuTrans.shape[0], 51))
		for x in range(cuTrans.shape[0]):
			for i in range(cuTrans.shape[1]):
				sum = 0
				for j in range(cuTrans.shape[2]):
					sum += cuTrans[x][i][j]
				if sum != 0:
					cuTrans[x][i] = cuTrans[x][i]/sum
					
		# zeroBoolX = 0
		# zeroBoolY = 0
		# arrZeroIndex = np.full((cuTrans.shape[0],cuTrans.shape[1]), np.inf)
		# indexCounter = 0
		# for x in range(cuTrans.shape[0]):
			# for i in range(cuTrans.shape[1]):
				# for j in range(cuTrans.shape[2]):
					# if cuTrans[x][i][j] != 0:
						# zeroBoolX = 1
				# if zeroBoolX == 0:
					# for z in range(cuTrans.shape[2]):
						# if cuTrans[x][z][i] != 0:
							# zeroBoolY = 1
					# if zeroBoolY == 0:
						# arrZeroIndex[x][indexCounter] = i
						# indexCounter += 1
				
				# zeroBoolX = 0
				# zeroBoolY = 0
			# indexCounter = 0
		
		# print(cuTrans[0])
		# print(arrZeroIndex[0])
		# print("*****************************************")
						
		# counter  = 0
		# for x in range(cuTrans.shape[0]):
			# for i in range(arrZeroIndex.shape[1]):
				# if arrZeroIndex[x][i] == np.inf:
					# counter += 1

			# ssm = np.zeros(shape=(counter, counter))
			# axisDelete = np.zeros(shape=(counter, cuTrans.shape[2])) 
			# axisDelete = np.delete(cuTrans[x], arrZeroIndex[x], 0)
			# print(axisDelete.shape)
			# ssm = np.delete(axisDelete, arrZeroIndex[x], 1)
			# print(ssm.shape)
			# print(ssm)
			# print(pysal.spatial_dynamics.ergodic.steady_state(ssm))
			# #print(ans[x])
			# mc = markovChain(ssm)
			# mc.computePi('linear')
			# print(mc.pi)
			# print(pysal.spatial_dynamics.ergodic.steady_state(cuTrans[x]))
			# time.sleep(10)

					
		# print(cuTrans[0].shape)
		# print(cuTrans[0])
		print("*****************************************")
		
		time.sleep(10)
		#ans = ans.astype(complex)
		#print(cuTrans[0])
		for i in range(cuTrans.shape[0]):
			#print(i)
			result = pysal.spatial_dynamics.ergodic.steady_state(cuTrans[i])
			#mc = markovChain(cuTrans[i])
			#print(cuTrans[i])
			#mc.computePi('linear')
			#print(mc.pi)
			#print(result)
			#time.sleep(5)
			if np.any(result.imag != 0):
				print(i)
			ans[i] = result
		
		print("imaginary printing")
		time.sleep(10)
		print("*************************************************")
		print(ans[0])
		
		#print(ans[60])
		#print(ans[272])
		#print(ans[0])
		
		plotterIndexes = np.full((ans.shape[0],ans.shape[0], 1), np.inf)
		sqrt_dis = np.full((ans.shape[0], ans.shape[0]), np.inf)
		maxValArr = np.zeros((ans.shape[0], ans.shape[0]))
		#sqrt_dis = sqrt_dis.astype(complex) 
		#b = np.array([0.0])
		arr1 = []
		
		print("square root difference")
		
		for i in range(ans.shape[0]):
			for j in range(i + 1, ans.shape[0]):
				maxValArr[i][j] = (np.sum((ans[i] - ans[j])**2))
		maxVal = np.amax(maxValArr) + 0.1
		print("maxVal is: " + str(maxVal))
		
		totW = 0
		for i in range(ans.shape[0]):
			for j in range(i + 1, ans.shape[0]):
				totW += 1 / (1 - ((np.sum((ans[i] - ans[j])**2))/maxVal))
				#sqrt_dis[i][j] = (np.sum((ans[i] - ans[j])**2))**(1/2)
				#plotterIndexes[i][j] = euc_dis[i][j]
			#print(sqrt_dis[i], i)
		print("total weight is: " + str(totW))
		for i in range(ans.shape[0]):
			for j in range(i + 1, ans.shape[0]):
				w = 1 / (1 - ((np.sum((ans[i] - ans[j])**2))/maxVal))
				sqrt_dis[i][j] = ((w/totW) * (np.sum((ans[i] - ans[j])**2)))**(1/2)		
		
		
		print(ans[0])
		print(ans[60])
		print((np.sum((ans[0] - ans[60])**2))**(1/2))
		print(sqrt_dis[0][60])
		#time.sleep(10)
		indexes = np.argwhere(sqrt_dis[0] > 0.7)
		print("indexes: " + str(indexes))
		print(indexes[0])
		
		timePlotter = []
		#cuValuesArr, timeValuesArr
		print(timeValuesArr[0][0])
		print(timeValuesArr[0][1])
		print(timeValuesArr[0][2])
		print(indexes.shape)
		print(indexes)
		# print(indexes[1, 0])
		# print(indexes[0, 1])
		# print(indexes[0, 3])

		print(timeValuesArr.shape)
		for i in range(len(indexes)):
			for j in range(len(timeValuesArr[0])):
				# print(indexes[i, 0])
				# print(j)
				timePlotter.append(timeValuesArr[indexes[i, 0]][j])
		
		cuPlotter = []
		for i in range(len(indexes)):
			for j in range(len(cuValuesArr[0])):
				cuPlotter.append(cuValuesArr[indexes[i, 0]][j])
		
		for j in range(ans.shape[0]):
			if sqrt_dis[0][j] != np.inf:
				arr1.append(sqrt_dis[0][j])

		#a.plotting("HighValuesCU", cuPlotter, timePlotter , 1, 0)
			
		counter = []
		for i in range(ans.shape[0]):
			counter.append(i)
		
		
		fileName = "transitionMatrix"
		titleAP = "file name is : steadyState"
		print("yoooohooooooooooooooooooooooooooooooo")
		trace = go.Heatmap(z = sqrt_dis)
		data = [trace]
		offline.plot(data, filename='weighted-mrse-heatmap.html')
		
		# print(euc_dis.shape)
		# euc_dis = euc_dis.reshape((1, euc_dis.shape[0]))
		# print(euc_dis.shape)
		a.plotting("steadyStateValues", counter, arr1 , 1, 2)
					
		#print(np.where(plotterIndexes != np.inf))
				
		print(np.amax(sqrt_dis))
		print(np.mean(sqrt_dis))
		print(np.amin(sqrt_dis))
		
		# print(np.argwhere(euc_dis < 0.5))
		
		
		# transProbCost = np.zeros(shape=(cuTrans.shape[0]))
		# for i in range(cuTrans.shape[0]):
			# for j in range(cuTrans.shape[1]):
				# jvalue = 1 / (1 - (j/52))
				# for z in range(cuTrans.shape[2]):
					# zvalue = 1 / (1 - (z/52))
					# transProbCost[i] += (z+1) * (j+1) * cuTrans[i][j][z]
					
		# for i in range(cuTrans.shape[0] - 1):
			# for j in range(cuTrans.shape[0] - 1):
				# if i != j:
					# f = np.array([cuTrans[i], cuTrans[j+1]])
					# res = pysal.spatial_dynamics.markov.kullback(f)
					# if res['Conditional homogeneity'] < 30:
						# print(res['Conditional homogeneity'])
						# print(np.transpose(np.nonzero(cuTrans[-1])))
						# print(np.transpose(np.nonzero(cuTrans[-2])))
		
		# arr = []
		
		# one_step_transition = np.array([[ 0.25      ,  0.5       ,  0.25      ],
        # [ 0.33333333,  0.        ,  0.66666667],
        # [ 0.33333333,  0.33333333,  0.33333333]])
		# steady_state_matrix = self.steadyState(one_step_transition.transpose())
		# print(steady_state_matrix)
		# b = pysal.spatial_dynamics.ergodic.steady_state(one_step_transition)
		# print(b)
		
			#val = np.reshape(val, (1, val.shape[0]))
			#print(val.shape)
			# if val.shape[0] < 20:
				# for j in range(20 - val.shape[0]):
					# #print(val)
					# a = np.array([0])
					# val = np.vstack((val, 0))
			# #print(val)
			# print(val1)
			# print(val2)
			#ans[i] = val
			#print(ans[i])
			#print(i)
			#print(ans)
			#time.sleep(1)
			
		#print(cuTrans.shape[0])
		# a, indices = np.unique(ans, return_inverse=True)
		
		# print(a)
		# print(len(a))
		#print(indices)
			
			
		# for i in range(ans.shape[0] - 1):
			# for j in range(i + 1, ans.shape[0] - 1):
				# if np.array_equal(ans[i], ans[j]) == True:
					# print("same")
					# print(i, j)
		
		# print(transProbCost)
		# print(np.where(cuTrans[-1] != 0))
		# print(np.where(cuTrans[-2] != 0))
		# f = np.array([cuTrans[-1], cuTrans[-2]])
		# print(pysal.spatial_dynamics.markov.kullback(f))
		# print(np.transpose(np.nonzero(cuTrans[-1])))
		# print(np.transpose(np.nonzero(cuTrans[-2])))
		# print(cuTrans[-1])
		# print(cuTrans[-2])
					
		
		# print(cuTrans[0])
		# print(cuValuesArr[0])
		# print(timeValuesArr[0])
		# print(type(timeValuesArr[0]))
		# print(type(timeValuesArr[0][0]))
		
		# print(cuTrans[600])


	def steadyState(self, p):
		dim = p.shape[0]
		q = (p-np.eye(dim))
		ones = np.ones(dim)
		q = np.c_[q,ones]
		QTQ = np.dot(q, q.T)
		bQT = np.ones(dim)
		return np.linalg.solve(QTQ,bQT)
		
	#finding transition states
	def transProbCalc(self, cuValuesArr, timeValuesArr, timerInMinute, fileName):
		print("calculating transition probabilities\n")
		cuTrans = np.zeros(shape=(cuValuesArr.shape[0], 51, 51))
		np.set_printoptions(threshold=np.inf)

		#each 5 values from 0 to 255 is one class now --> from 100 it will get 2%
		for i in range(cuValuesArr.shape[0]):
			for j in range(cuValuesArr.shape[1] - 1):
				start = cuValuesArr[i, j] / 0.02
				next = cuValuesArr[i, j + 1] / 0.02
				cuTrans[i, math.floor(start), math.floor(next)] += 1
			start = cuValuesArr[i, -1] / 0.02
			next = cuValuesArr[i, 0] / 0.02
			cuTrans[i, math.floor(start), math.floor(next)] += 1
				
		pickleName = "./node1/markovianTimePickles/cuTrans"+ str(fileName) + str(timerInMinute)
		with open(pickleName, 'wb') as f:
			pickle.dump(cuTrans, f)
			
		pickleName = "./node1/markovianTimePickles/cuValuesArr"+ str(fileName) + str(timerInMinute)
		with open(pickleName, 'wb') as f:
			pickle.dump(cuValuesArr, f)			

		
		pickleName = "./node1/markovianTimePickles/timeValuesArr"+ str(fileName) + str(timerInMinute)
		with open(pickleName, 'wb') as f:
			pickle.dump(timeValuesArr, f)
		
		return cuTrans, cuValuesArr, timeValuesArr
		
		
				
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