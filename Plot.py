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
import learningAlgs as classImportLA
import dataManipulation as dataMan

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
import warnings



class plottData:


	def __init__(self):
		#initialize the class, creating lists, taking care of time differences
		self.timeArr = []
		self.maxArr = []
		self.minArr = []
		self.from_zone = tz.gettz('UTC')
		self.to_zone = tz.gettz('America/Edmonton')
		now_timestamp = time.time()
		self.offset = datetime.datetime.fromtimestamp(now_timestamp) - datetime.datetime.utcfromtimestamp(now_timestamp)
		self.signalVal = 0
		self.arr = []
		for i in range(30):
			for j in range(60):
				self.arr.append("2018-10-23 13:" + str(i) + ":" + str(j))
		self.chanUtil = []
			
		
	def prepration(self):
	
		channelPlotterBool = 0
		channelPlotterBool = input("Plotting frequency based [2], channel based [1], or AP based [0]? (default is 0) ")
		fileName = ""
		
		print(channelPlotterBool)
		
		if int(channelPlotterBool) != 0 and int(channelPlotterBool) != 1 and int(channelPlotterBool) != 2:
			return 0
			
		print("1")	
		#finding all the files in the folder
		nodeNumber = input("what is the node number? ")
		files = subprocess.Popen("ls " + "node" + str(nodeNumber) + "/extractedData/", shell=True, stdout=subprocess.PIPE)
		fileNames = files.stdout.read().decode("ascii")
		fileNameArr = []
		strName = ""
		
		
		print("2")
		for i in range(len(str(fileNames))):
			if str(fileNames[i]) != "\n":
				strName += str(fileNames[i])
			else:
				fileNameArr.append(strName)
				strName = ""
				
				 
		preHour = ''
		counter = -1
		min = float('inf')
		max = -float('inf') 
		prevTimeStamp = ""
		result = 0
		
		#check did we process raw data for channel-based or not
		if int(channelPlotterBool) == 1:
			result = self.CSVCheck(fileName, nodeNumber, channelPlotterBool)
		
		#if we did not proccess raw data for channel-based or it is not for channel-based at all:
		if int(channelPlotterBool) != 1 or (int(channelPlotterBool) == 1 and result == False):
			print("3")
			#choosing each file in the folder for the plotting
			print("existing files are: " + str(fileNameArr))
			for i in range(len(fileNameArr)):
				time.sleep(0.5)
				fileName = fileNameArr[i]
				if int(channelPlotterBool) != 1:
					result = self.CSVCheck(fileName, nodeNumber, channelPlotterBool)
				
				if (int(channelPlotterBool) != 1 and result == False) or int(channelPlotterBool) == 1:
					print("reading file with the name: " + str(fileName))
					cu = ""
					lineCounter = 0
					fileNameStr = "node" + str(nodeNumber) + "/extractedData/" + str(fileName)
					#sending file for data extraction
					dataMan.dataSplitting(fileName, channelPlotterBool, 0)
					#now we have files for that
		
		
	def CSVCheck(self, fileName, nodeNumber, channelPlotterBool):
		obj = classImportLA.learningAlgs()
		if int(channelPlotterBool) == 1:
			for i in range(1, 4):
				if i == 1:
					num = 1
				elif i == 2:
					num = 6
				elif i == 3:
					num = 11
				fileName = "channel" + str(num)
				CSVStat, CU, Time = obj.csvChecker(fileName, 0)

				if CSVStat == True:
					print("here it is\n")
					self.plotting(fileName, CU, Time, nodeNumber, channelPlotterBool)
				
				else:
					return False
						
					
		else:
			CSVStat, CU, Time = obj.csvChecker(fileName, 0)
			if CSVStat == True:
				print("here it is")
				if int(channelPlotterBool) == 0:
					self.plotting(fileName, CU, Time, nodeNumber, channelPlotterBool)
					
				else: #channelPlotterBool == 2
					CU = np.round(CU, decimals = 5)
					CUVal, count = np.unique(CU, return_counts = True)
					self.plotting(fileName, CUVal, count, nodeNumber, channelPlotterBool)
				return True
				
			else:
				return False
			

	def plotting(self, fileName, CU, Time, nodeNumber, channelPlotterBool):
		print(Time)
		print(CU)
		if int(channelPlotterBool) == 2: #Time is Count here
			count = Time
			data = [go.Scatter( x = CU, y = count )]
		else:
			data = [go.Scatter( x = Time, y = CU )]
		
		fileName = str(fileName) + " mode" + channelPlotterBool
		titleAP = "file name is : " + str(fileName)
		print("yoooohooooooooooooooooooooooooooooooo")
		layout = go.Layout(title= titleAP, showlegend = False)
		fig = go.Figure(data=data, layout=layout)
		plotName = "node" + str(nodeNumber) + "/plot/" + str(fileName) + ".html"
		offline.plot(fig, filename = plotName)
			

if __name__ == '__main__':
	obj = plottData()
	obj.prepration()