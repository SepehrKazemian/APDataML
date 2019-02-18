import numpy as np
import Plot as plot
import math
# import matplotlib.pyplot as plt
# import plotly.plotly as py
# import plotly.graph_objs as go
# import plotly.offline as offline
import time
import datetime
import pickle
import os
import pysal
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib as plt

class MarkovChain():
	def __init__(self):
		self.timeArr = []
		
		
	def timeChunks(self, time, cu, timerInMinute):
		print("chunking the data\n")
		numberOfTimeElements = int(60 / 5 * 20)
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
		
		
	def pickleChecker(self, timerInMinute):
		
		print("checking pickles\n")
		
		fileName = "./node1/markovianTimePickles/cuTrans" + str(timerInMinute)
		if os.path.isfile(fileName) == True:
			with open(fileName, "rb") as f:
				cuTrans = pickle.load(f)
			
			fileName = "./node1/markovianTimePickles/cuValuesArr" + str(timerInMinute)
			with open(fileName, 'rb') as f:
				cuValuesArr = pickle.load(f)
				
			fileName = "./node1/markovianTimePickles/timeValuesArr" + str(timerInMinute)
			with open(fileName, 'rb') as f:
				timeValuesArr = pickle.load(f)
			
			print("checking transition probabilities\n")
			return cuTrans, cuValuesArr, timeValuesArr

		else:
			return None, None, None
		
		
	def transitionMatrix(self):
		
		a = plot.plottData()
		timerInMinute = 20
		cuTrans, cuValuesArr, timeValuesArr = self.pickleChecker(timerInMinute)
		if type(cuTrans) != np.ndarray:
			print("gathering processed data\n")
			CU, Time = a.prepration()
			cuValuesArr, timeValuesArr = self.timeChunks(Time, CU, timerInMinute)
			cuTrans = self.transProbCalc(cuValuesArr, timeValuesArr, timerInMinute)
		
		np.set_printoptions(threshold=np.inf)

		
		#normalizing data
		print(cuTrans.dtype)
		cuTrans = cuTrans.astype(complex)
		print(cuTrans.dtype)
		print("normalizing")
		ans = np.zeros(shape=(cuTrans.shape[0], 51), dtype=complex)
		for x in range(cuTrans.shape[0]):
			for i in range(cuTrans.shape[1]):
				sum  = 0
				for j in range(cuTrans.shape[2]):
					sum += cuTrans[x][i][j]
				if sum != 0:
					cuTrans[x][i] = cuTrans[x][i]/sum
		time.sleep(1)
			
		print(cuTrans[0])
		for i in range(cuTrans.shape[0]):
			result = pysal.spatial_dynamics.ergodic.steady_state(cuTrans[i])
			ans[i] = result
		
		print(ans[60])
		print(ans[272])
		print(ans[0])
		
		plotterIndexes = np.full((ans.shape[0],ans.shape[0], 1), np.inf)
		sqrt_dis = np.full((ans.shape[0], ans.shape[0], 1), np.inf)
		sqrt_dis = sqrt_dis.astype(complex) 
		b = np.array([0.0])
		arr1 = []
		
		print("square root difference")
		for i in range(ans.shape[0]):
			for j in range(i + 1, ans.shape[0]):
				sqrt_dis[i][j] = np.sum((ans[i] - ans[j])**2)**(1/2)
				#plotterIndexes[i][j] = euc_dis[i][j]
			#print(sqrt_dis[i], i)
		
		print(ans[0])
		print(ans[60])
		print(np.sum((ans[0] - ans[60])**2)**(1/2))
		
		print(np.argwhere(sqrt_dis[0] > 1))
		
		for j in range(ans.shape[0]):
			if sqrt_dis[0][j] != np.inf:
				arr1.append((sqrt_dis[0][j]))
			
		counter = []
		for i in range(ans.shape[0]):
			counter.append(i)
		
		# print(euc_dis.shape)
		# euc_dis = euc_dis.reshape((1, euc_dis.shape[0]))
		# print(euc_dis.shape)
		#a.plotting("steadyStateValues", counter, arr1 * 100, 1, 2)
					
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
		
		
	def transProbCalc(self, cuValuesArr, timeValuesArr, timerInMinute):
		print("calculating transition probabilities\n")
		cuTrans = np.zeros(shape=(cuValuesArr.shape[0], 51, 51))
		np.set_printoptions(threshold=np.inf)

		for i in range(cuValuesArr.shape[0]):
			for j in range(cuValuesArr.shape[1] - 1):
				start = cuValuesArr[i, j] / 0.02
				next = cuValuesArr[i, j + 1] / 0.02
				cuTrans[i, math.floor(start), math.floor(next)] += 1
				
		fileName = "./node1/markovianTimePickles/cuTrans" + str(timerInMinute)
		with open(fileName, 'wb') as f:
			pickle.dump(cuTrans, f)

		
		return cuTrans
		
		
				
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