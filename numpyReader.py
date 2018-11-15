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


class plottData:


	def __init__(self):
		self.timeArr = []
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
			
		
	def plotting(self):
	


#		fileName = fileNameArr[i]
		fileName = "500f8022abe0"
		timerInMinute = 20
		numberOfSamples = timerInMinute * 12 #60 seconds / 5 seconds per sample

		
		self.dataSplitting(fileName)
		self.createTimeSplitter(timerInMinute)
		self.dataPoint(numberOfSamples, timerInMinute)
		
	def dataSplitting(self, fileName):
		secondCounter = 0
		prevTime = ""
		curTime = ""
		cu = ""
		lineCounter = 0
		counter = 0
		preMaxVal = 0
		with open("aa/" + str(fileName)) as fp:
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
					self.signalVal += int(signal)
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
							self.timeArr.append(central)
							self.chanUtil = np.append(self.chanUtil, maxVal)
							prevTime = prevTime + 5
							
						maxVal = 0
						
					
					elif currTimeStamp - prevTime < 5:
						if maxVal < int(cu):
							maxVal = int(cu)
					

				except ValueError:
					print("channel utilization error")
		print(len(self.chanUtil))
		print(len(self.timeArr))

		
	def createTimeSplitter(self, timeSplitter):
		print(timeSplitter)
		splits = int((24 * 60) / (int(timeSplitter)))
		self.timeSplit = np.zeros(splits)
		
		self.days = np.zeros(7)
	
	
	def dataPoint(self, numberOfSamples, timerInMinute):
		print(self.timeArr[0])
		print(self.timeArr[240])
		print(self.timeArr[0].day)
		print(self.timeArr[0].hour)
		Xlearn = []
		Ylearn = []
		for i in range(0, len(self.timeArr) - numberOfSamples, 12): #start to iterate from zero till the end and slide every minute forward
			Xarr = []
			Yarr = []
			Xarr = np.append(Xarr, self.timeArr[i : i + 240])

			firstElement = self.timeArr[i].hour * 3 + math.ceil(self.timeArr[i].minute / int(timerInMinute))
			lastElement = self.timeArr[i + 240].hour * 3 + math.ceil(self.timeArr[i + 240].minute / int(timerInMinute))
			self.timeSplit[firstElement - 1] = 1
			self.timeSplit[lastElement - 1] = 1
			Xarr = np.append(Xarr[i], self.timeSplit)
			
			firstDay = self.timeArr[i].weekday()
			lastDay = self.timeArr[i + 240].weekday()
			self.days[firstDay] = 1
			self.days[lastDay] = 1
			Xarr = np.append(Xarr, self.days)
			
			Xlearn = np.vstack([Xlearn, Xarr])
			
			self.days[firstDay] = 0
			self.days[lastDay] = 0
			self.timeSplit[firstElement - 1] = 0
			self.timeSplit[lastElement - 1] = 0
			
			Ypred = self.chanUtil[i + 241]
			Ylearn = np.vstack([Ylearn, Xarr])
			
			Yarr = np.append(Yarr, self.chanUtil[i : i + 240])
			
			print(Ylearn)
			print(Ylearn.shape)
			print(Xlearn)
			print(Xlearn.shape)
			
		print(numberOfSamples)
				

	#		print(self.timeArr)
	#		print(self.minArr)
	#		print(self.maxArr)
#			self.signalVal = int(self.signalVal / lineCounter)
#			data = [go.Scatter( x = self.timeArr, y=self.chanUtil )]
#			titleAP = "AP MAC Address is: " + str(fileName) + " with Mean Signal Value of: " + str(self.signalVal)
#			layout = go.Layout(title= titleAP, showlegend = False)
#			fig = go.Figure(data=data, layout=layout)
#			offline.plot(fig, filename = str(fileName) + ".html")
#			self.signalVal = 0
#			self.timeArr = []
#			self.chanUtil = []


if __name__ == '__main__':
	obj = plottData()
	obj.plotting()