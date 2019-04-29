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
			
		
	def prepration(self, fileName):
	
		channelPlotterBool = 0
		channelBasedFileExist = 0
		channelPlotterBool = input("Plotting frequency based [2], channel based [1], or AP based [0]? (default is 0) ")
		plotterBool = int(input("do you want to plot the data? Yes[1], No[0]"))
		timeInterval = int(input("what is the interval of data in the dataset? "))
		#fileName = ""
		
		print(channelPlotterBool)
		
		if int(channelPlotterBool) != 0 and int(channelPlotterBool) != 1 and int(channelPlotterBool) != 2:
			return 0
			
		#finding all the files in the folder
		#nodeNumber = input("what is the node number? ")
		nodeNumber = 2
		address = input("what is the absolute address? ")
		
		extraArr = []
		if fileName == "":
			fileNameArr = os.listdir(address)
			for i in range(len(fileNameArr)):
				if ".txt" not in fileNameArr[i]:
					extraArr.append(fileNameArr[i])
		
			for i in range(len(extraArr)):
				fileNameArr.remove(extraArr[i])
		
			print(fileNameArr)
			# for i in range(len(str(fileNames))):
				# if str(fileNames[i]) != "\n":
					# strName += str(fileNames[i])
				# else:
					# fileNameArr.append(strName)
					# strName = ""
		
		else:
			fileNameArr = []
			fileNameArr.append(fileName)

		#fileNameArr = ['40017ad6b2e0-6', '500f801e28a0-6']
				 
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
				# absoluteAdd = "/home/Sepehr/Desktop/project/thesis/data/11node1-5/alligned"
				# fileAbsoluteAdd = absoluteAdd + str(fileNameArr[i])
				pathFile = address + "/CSV/"+fileNameArr[i] + ".csv"
				if os.path.isfile(pathFile) == False:
					self.processRawDataCaller(fileNameArr[i], address, channelPlotterBool, timeInterval)
				stat, chanUtil, timeArr = LA.csvChecker(fileNameArr[i], 0, address)
				if int(plotterBool) == 1:
					self.plotting(fileAbsoluteAdd, chanUtil, timeArr, address, channelPlotterBool)
				# else:
					# return chanUtil, timeArr			
		
			

	def processRawDataCaller(self, fileName, address, channelPlotterBool, timeInterval):
		fileNameStr = str(fileName)
		#sending file for data extraction
		dataMan.dataSplitting(fileName, channelPlotterBool, 0, timeInterval, address)
		#now we have files for that
		
			

	def plotting(self, fileName, CU, Time, address, channelPlotterBool):
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

		absoluteAdd = "/home/Sepehr/Desktop/project/thesis/data/11node1-5/alligned"		
		plotName = address + "/plot/" + str(fileName) + ".html"
		offline.plot(fig, filename = plotName)
			

if __name__ == '__main__':
	obj = plottData()
	obj.prepration("")