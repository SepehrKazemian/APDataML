import numpy as np
import Plot as plot
import math
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline

class MarkovChain():
	def __init__(self):
		a = 1
		
		
	def transitionMatrix(self):
		a = plot.plottData()
		CU, Time = a.prepration()
		print(CU, Time)
		
		cuTrans = np.zeros((51,51))
		
		for i in range(len(CU) - 1):
			start = CU[i] / 0.02
			next = CU[i + 1] / 0.02
			cuTrans[math.floor(start), math.floor(next)] += 1
		
		
		print(np.count_nonzero(cuTrans))
		probability = np.true_divide(cuTrans, len(CU) - 1)
		print(np.count_nonzero(cuTrans))
		
		fileName = "transitionMatrix"
		titleAP = "file name is : " + str(fileName)
		print("yoooohooooooooooooooooooooooooooooooo")
		trace = go.Heatmap(z = probability)
		data = [trace]
		offline.plot(data, fileName)
		
		
if __name__ == '__main__':
	obj = MarkovChain()
	obj.transitionMatrix()