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


class plottData:


	def __init__(self):
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
			
		
	def plotting(self):
	
		files = subprocess.Popen("ls aa/", shell=True, stdout=subprocess.PIPE)
		fileNames = files.stdout.read().decode("ascii")
		fileNameArr = []
		strName = ""
		
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
		
		for i in range(len(fileNameArr)):
			print(fileNameArr)
			time.sleep(1)
			fileName = fileNameArr[i]
#			fileName = "500f8022abe0"
			print(fileName)
			cu = ""
			lineCounter = 0
			with open("aa/" + str(fileName)) as fp:
				for line in fp:
					cu = ""
					lineCounter += 1
					month = ""
					day = ""
					timer = ""
					year = ""
					channel = ""
					
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
					
					currTimeStampUTC = datetime.datetime.fromtimestamp(currTimeStamp).strftime('%Y-%m-%d %H:%M:%S')
					currTimeStampUTC = datetime.datetime.strptime(currTimeStampUTC, '%Y-%m-%d %H:%M:%S')
					central = self.offset + currTimeStampUTC
					try:
						self.timeArr.append(central)
						self.chanUtil.append(int(cu))
					except ValueError:
						print("channel utilization error")

	#		print(self.timeArr)
	#		print(self.minArr)
	#		print(self.maxArr)
			self.signalVal = int(self.signalVal / lineCounter)
			data = [go.Scatter( x = self.timeArr, y=self.chanUtil )]
			titleAP = "AP MAC Address is: " + str(fileName) + " with Mean Signal Value of: " + str(self.signalVal)
			layout = go.Layout(title= titleAP, showlegend = False)
			fig = go.Figure(data=data, layout=layout)
			offline.plot(fig, filename = str(fileName) + ".html")
			self.signalVal = 0
			self.timeArr = []
			self.chanUtil = []


if __name__ == '__main__':
	obj = plottData()
	obj.plotting()