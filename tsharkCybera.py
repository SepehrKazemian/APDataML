#!/usr/bin/python

import threading, time
import subprocess
from subprocess import Popen, PIPE
import os
import gzip

class tshark():
	def __init__(self):
		self.totalLength = 0
		self.numbNodes = sys(arv[1])
		self.nodeCounter = [0 for i in range(self.numbNodes)]
		
		#based on number of nodes that we have, we create or check the previous values of read data
		for i in range(1, self.numbNodes + 1):
			nodeCounterName = "node" + i + ".txt"
			if os.path.isfile(nodeCounterName) == True:
				if (os.stat(nodeCounterName).st_size == 0) == False:
					with open(nodeCounterName, "r") as txt:
						self.nodeCounter[i - 1] = str(int(txt.readline()))
				else:
					self.nodeCounter[i - 1] = str(0)
			else:
				os.system("touch node1.txt")
				self.nodeCounter[i - 1] = str(0)	
			
			
		print(self.nodeCounter)
		self.secondChar = None

	def continuousReading(self):
		#we are creating pool of threads for each node
		threadPool = []
		for i in range(self.numbNodes):
			nodeName = "node" + str(i + 1)
			thread = threading.Thread(target = self.runningCommand, args = (nodeName, i))
			threadPool.append(thread)
		
		for i in range(self.numbNodes):
			threadPool[i].start()
		

	def runningCommand(self, nodeName, arrIndex):
		fileName = nodeName + "." + str(self.nodeCounter[arrIndex])
		print(fileName)	
		nodeText = str(nodeName) + ".txt"
		while True:
			fileName = nodeName + "." + str(self.nodeCounter[arrIndex])
			
			#we should update the text to state for which number we are waiting
			with open(nodeText, "w") as textStat:
				textStat.write(str(self.nodeCounter[arrIndex]))
			if os.path.isfile(fileName) == True:
				print(fileName + " file is here")
				#when we got the number we should pass it to tshark to analyze it
				tsharkCommand = 'tshark -r ' + fileName + ' -Tfields -e wlan.ta -e wlan_radio.channel -e wlan.qbss.cu -e wlan_radio.signal_dbm -e frame.time -E separator=/s'
				#should call pipe for the subprocess to read and write simultaneously
				proc = Popen(tsharkCommand, stdout = PIPE, shell = True)
	
				while proc.poll() is None:
					time.sleep(0.00001)
					#for every incoming line we should block and then read it
					line = proc.stdout.readline()
					#sending line for analyzing
					self.analyze(line.decode('ascii'), arrIndex)
			
				with open(fileName, 'rb') as inFile:
					content = inFile.read()
				gzName = fileName + ".gz"
				with gzip.GzipFile(filename= gzName, mode="wb", compresslevel=9) as file:
					file.write(content)
				print(gzName + " is completed")
				
				#uploading files into gdrive as backup/ up to 4 device is supported in this code
				if arrIndex == 0:
					os.system("/home/ubuntu/gdrive-linux-x64 upload --parent 1KyFUjKDHIR5hapu02re-b7z5cf2mBEDf " + gzName )
					print(gzName + " is uploaded")

				elif arrIndex == 1:
					os.system("/home/ubuntu/gdrive-linux-x64 upload --parent 101ltxcgr5dyChf5JEG913pds-uzSZ1sK " + gzName )
					print(gzName + " is uploaded")
					
				elif arrIndex == 2:
					os.system("/home/ubuntu/gdrive-linux-x64 upload --parent 101ltxcgr5dyChf5JEG913pds-uzSZ1sK " + gzName )
					print(gzName + " is uploaded")

				elif arrIndex == 3:
					os.system("/home/ubuntu/gdrive-linux-x64 upload --parent 101ltxcgr5dyChf5JEG913pds-uzSZ1sK " + gzName )
					print(gzName + " is uploaded")

				os.system("rm " + fileName)
				os.system("rm " + gzName)
				self.nodeCounter[arrIndex] = str(int(self.nodeCounter[arrIndex]) + 1)
			

					
					
		
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
			directory = "node" + str(arrIndex+1)  + "/" + str(macAdd[0:12]) + ".txt"
			with open(directory, "a") as file:
				file.write(time + " " + channel + " " + cu + " " + sig + "\n")
			
			chanDirectory = "channel/" + "channel" + str(channel) + ".txt"
			with open(chanDirectory, "a") as file:
				file.write(time + " " + channel + " " + cu + " " + sig + "\n")			
		
		else:
			print("macAdd is: " + str(macAdd))
			
			

if __name__ == '__main__':
#	print("aaa")
	if len(sys.argv) == 2:
		tshObj = tshark()
		tshObj.continuousReading()
	else:
		print("Enter Number of Nodes")