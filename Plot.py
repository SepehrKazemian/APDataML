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
		channelBasedFileExist = 0
		channelPlotterBool = input("Plotting frequency based [2], channel based [1], or AP based [0]? (default is 0) ")
		plotterBool = int(input("do you want to plot the data? Yes[1], No[0]"))
		#fileName = ""
		
		print(channelPlotterBool)
		
		if int(channelPlotterBool) != 0 and int(channelPlotterBool) != 1 and int(channelPlotterBool) != 2:
			return 0
			
		#finding all the files in the folder
		nodeNumber = input("what is the node number? ")
		files = subprocess.Popen("ls " + "node" + str(nodeNumber) + "/extractedData/", shell=True, stdout=subprocess.PIPE)
		fileNames = files.stdout.read().decode("ascii")
		fileNameArr = []
		strName = ""
		
		
		# for i in range(len(str(fileNames))):
			# if str(fileNames[i]) != "\n":
				# strName += str(fileNames[i])
			# else:
				# fileNameArr.append(strName)
				# strName = ""
				
		fileNameArr = ['40017ad6b2e0-6', '500f801e28a0-6']
				 
		preHour = ''
		counter = -1
		min = float('inf')
		max = -float('inf') 
		prevTimeStamp = ""
		result = 0
		
		#check did we process raw data for channel-based or not by checking the existence of CSV file
		if int(channelPlotterBool) == 1:
			for i in range(1, 11, 5): #checking channels 1, 6, 11  
				fileName = "channel" + str(i)
				ans = os.path.isfile(fileName)
				if ans == False:
					channelBasedFileExist = 0
					break
				else:
					channelBasedFileExist = 1
			
			if channelBasedFileExist == 0:
				for i in range(1, 11, 5): 
					fileName = "channel" + str(i)
					ans = os.path.isfile(fileName)
					if ans == True:
						os.system("rm -f " + str(fileName))
				for i in range(len(fileNameArr)):
					self.processRawDataCaller(fileNameArr[i], nodeNumber, channelPlotterBool)
				
				LA = classImportLA.learningAlgs()
				for i in range(1, 11, 5): 
					fileName = "channel" + str(i)
					stat, chanUtil, timeArr = LA.csvChecker(fileName, 0)
					if int(channelPlotterBool) == 1:
						self.plotting(fileName, chanUtil, timeArr, nodeNumber, channelPlotterBool)
					else:
						return chanUtil, timeArr
						
						
		
		#check did we process raw data for AP-based or not by checking the existence of CSV file
		if int(channelPlotterBool) == 0 or int(channelPlotterBool) == 2:
			LA = classImportLA.learningAlgs()
			for i in range(len(fileNameArr)):
				pathFile = "node1/CSV/" + fileNameArr[i] + ".csv"
				if os.path.isfile(pathFile) == False:
					self.processRawDataCaller(fileNameArr[i], nodeNumber, channelPlotterBool)
				stat, chanUtil, timeArr = LA.csvChecker(fileNameArr[i], 0)
				if int(channelPlotterBool) == 1:
					self.plotting(fileNameArr[i], chanUtil, timeArr, nodeNumber, channelPlotterBool)
				# else:
					# return chanUtil, timeArr			
		
			

	def processRawDataCaller(self, fileName, nodeNumber, channelPlotterBool):
		fileNameStr = "node" + str(nodeNumber) + "/extractedData/" + str(fileName)
		#sending file for data extraction
		dataMan.dataSplitting(fileName, channelPlotterBool, 0)
		#now we have files for that
		
			

	def plotting(self, fileName, CU, Time, nodeNumber, channelPlotterBool):
		#print(Time)
		#print(CU)
		if int(channelPlotterBool) == 2: #Time is Count here
			count = Time
			data = [go.Scatter( x = CU, y = count )]
		else:
			data = [go.Scatter( x = Time, y = CU )]
		
		fileName = str(fileName) + " mode" + str(channelPlotterBool)
		titleAP = "file name is : " + str(fileName)
		print("yoooohooooooooooooooooooooooooooooooo")
		layout = go.Layout(title= titleAP, showlegend = False)
		fig = go.Figure(data=data, layout=layout)
		plotName = "node" + str(nodeNumber) + "/plot/" + str(fileName) + ".html"
		offline.plot(fig, filename = plotName)
			

if __name__ == '__main__':
	obj = plottData()
	obj.prepration()