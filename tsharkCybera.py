#!/usr/bin/python

import threading, time
import subprocess
from subprocess import Popen, PIPE
import os
import gzip

class tshark():
	def __init__(self):
		self.totalLength = 0
		self.nodeCounter = ["439", "439"]
		self.secondChar = None

	def continuousReading(self):
		nodeName2 = 'node2.'
		thread2 = threading.Thread(target = self.runningCommand, args = (nodeName2, 1))
		thread2.start()


	def runningCommand(self, nodeName, arrIndex):
		fileName = ""
		counter = 0
		while True:
			fileName = nodeName + str(self.nodeCounter[arrIndex])
			os.system("cp " + fileName + ".gz ./" + fileName + "cp.gz") 
			os.system("gunzip " + fileName + ".gz")
			if os.path.isfile(fileName) == True:
				print(fileName + " file is here")
#			if os.path.isfile(fileName) == False:
#				if startingName == 'node1.' + str(secondCounter):
#					secondCounter += 1 
#				startingName = 'node1.' + str(secondCounter)
#				num = 0
#				fileName = startingName + str(num)
#				f = gzip.open(fileName, 'r')
#				content = f.read()
#				f.close()
#				print(content)
				tsharkCommand = 'tshark -r ' + fileName + ' -Tfields -e wlan.ta -e wlan_radio.channel -e wlan.qbss.cu -e wlan_radio.signal_dbm -e frame.time -E separator=/s'
			#should call pipe for the subprocess to read and write simultaneously
				proc = Popen(tsharkCommand, stdout = PIPE, shell = True)
	
				while proc.poll() is None:
					time.sleep(0.000001)
					#for every incoming line we should block and then read it
					line = proc.stdout.readline()
#					print(line.decode('ascii'))
#					print(line)
					self.analyze(line.decode('ascii'), arrIndex)

				self.nodeCounter[arrIndex] = str(int(self.nodeCounter[arrIndex]) + 1)	
				counter += 1
				os.system("rm " + fileName)
			else:
				print(fileName)
				break


			'''			
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
				if int(self.nodeCounter[arrIndex]) < 100000:
					self.nodeCounter[arrIndex] = str(int(self.nodeCounter[arrIndex]) + 1)
				elif int(self.nodeCounter[arrIndex]) >= 100000:
					if self.secondChar == None:
						self.secondChar = 0
						self.nodeCounter[arrIndex] += str(self.secondChar)
					else:
						self.nodeCounter[arrIndex] = self.nodeCounter[arrIndex][0: len(self.nodeCounter[arrIndex]) - len(str(self.secondChar))]
						self.secondChar += 1
						self.nodeCounter[arrIndex] += str(self.secondChar)
					self.nodeCounter[arrIndex] = '0'
					
		
			time.sleep(100)
			'''

			
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
#		print(line)	
#		print(macAdd)
#		print(channel)
#		print(cu)
		if len(macAdd) == 12:
			directory = "node" + str(arrIndex+1)  + "/" + str(macAdd[0:12])
			with open(directory, "a") as file:
				file.write(time + " " + channel + " " + cu + " " + sig + "\n")
		
		else:
			print(line)
			print("macAdd is: " + str(macAdd))
			
			

if __name__ == '__main__':
#	print("aaa")
	tshObj = tshark()
	tshObj.continuousReading()
