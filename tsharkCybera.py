#!/usr/bin/python

import threading, time
import subprocess
from subprocess import Popen, PIPE
import os
import gzip

class tshark():
	def __init__(self):
		self.totalLength = 0
		with open("node1.txt", "r") as txt:
			node1Content = txt.readline()
		with open("node2.txt", "r") as txt:
			node2Content = txt.readline()
		if node1Content == "":
			node1Content = "1"
		if node2Content == "":
			node2Content = "1"

		self.nodeCounter = [node1Content[0:len(node1Content)-1], node2Content[0:len(node2Content)-1]]
		print(self.nodeCounter)
		self.secondChar = None

	def continuousReading(self):
		nodeName1 = 'node1'
		nodeName2 = 'node2'
		thread1 = threading.Thread(target = self.runningCommand, args = (nodeName1, 0))
		thread2 = threading.Thread(target = self.runningCommand, args = (nodeName2, 1))
		thread1.start()
		thread2.start()


	def runningCommand(self, nodeName, arrIndex):
		fileName = nodeName + "." + str(self.nodeCounter[arrIndex])
		print(fileName)	
		nodeText = str(nodeName) + ".txt"
		while True:
			fileName = nodeName + "." + str(self.nodeCounter[arrIndex])
			with open(nodeText, "w") as textStat:
				textStat.write(str(self.nodeCounter[arrIndex]))
			if os.path.isfile(fileName) == True:
				print(fileName + " file is here")
#			if os.path.isfile(fileName) == False:
#				if startingName == 'node1.' + str(secondCounter):
#					secondCounter += 1 
#				startingName = 'node1.' + str(secondCounter)
#				num = 0
#				fileName = startingName + str(num)
				tsharkCommand = 'tshark -r ' + fileName + ' -Tfields -e wlan.ta -e wlan_radio.channel -e wlan.qbss.cu -e wlan_radio.signal_dbm -e frame.time -E separator=/s'
			#should call pipe for the subprocess to read and write simultaneously
				proc = Popen(tsharkCommand, stdout = PIPE, shell = True)
	
				while proc.poll() is None:
					time.sleep(0.00001)
					#for every incoming line we should block and then read it
					line = proc.stdout.readline()
#					print(line.decode('ascii'))
#					print(line)
					self.analyze(line.decode('ascii'), arrIndex)
			
				with open(fileName, 'rb') as inFile:
					content = inFile.read()
				gzName = fileName + ".gz"
				with gzip.GzipFile(filename= gzName, mode="wb", compresslevel=9) as file:
					file.write(content)
				print(gzName + " is completed")
				if arrIndex == 0:
					os.system("/home/ubuntu/gdrive-linux-x64 upload --parent 1KyFUjKDHIR5hapu02re-b7z5cf2mBEDf " + gzName )
					print(gzName + " is uploaded")

				elif arrIndex == 1:
					os.system("/home/ubuntu/gdrive-linux-x64 upload --parent 101ltxcgr5dyChf5JEG913pds-uzSZ1sK " + gzName )
					print(gzName + " is uploaded")

				os.system("rm " + fileName)
				os.system("rm " + gzName)
#				if int(self.nodeCounter[arrIndex]) =< 100000:
				self.nodeCounter[arrIndex] = str(int(self.nodeCounter[arrIndex]) + 1)
#				nodeText = str(nodeName) + "txt"	
#				with open(nodeText, "w") as textStat:
#					textStat.write(self.nodeCounter[arrIndex])
	#			elif int(self.nodeCounter[arrIndex]) > 100000:
	#				if self.secondChar == None:
	#					self.secondChar = 0
	#					nodeName += str(self.secondChar)
	#				else:
	#					nodeName = nodeName[0: len(nodeName) - len(str(self.secondChar))]
	#					self.secondChar += 1
	#					nodeName += str(self.secondChar)
	#				self.nodeCounter[arrIndex] = '0'
					
					
		
			time.sleep(100)


			
	def analyze(self, line, arrIndex):
		counter = 0
		macAdd = ''
		channel = ''
		cu = ''
		sig = ''
		time = ''
		for i in range(len(line)):
			if line[i] != " " and counter == 0:
				if line[i] != ":":
					macAdd += line[i]
			
			elif line[i] != " " and counter == 1:
				channel += line[i]

			elif line[i] != " " and counter == 2:
				cu += line[i]
			
			
			elif line[i] != " " and counter == 3:
				sig += line[i]
			
			elif counter == 4:
				if line[i] != "\n":
					time += line[i]
			
			else:
				counter += 1
		
#		print(macAdd)
#		print(channel)
#		print(cu)
		if len(macAdd) == 12:
			directory = "node" + str(arrIndex+1)  + "/" + str(macAdd[0:12])
			with open(directory, "a") as file:
				file.write(time + " " + channel + " " + cu + " " + sig + "\n")
		
		else:
			print("macAdd is: " + str(macAdd))
			
			

if __name__ == '__main__':
#	print("aaa")
	tshObj = tshark()
	tshObj.continuousReading()
