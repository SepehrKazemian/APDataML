import os, sys
import threading, time
import subprocess
from subprocess import Popen, PIPE
import os
import datetime
import logging
#import urllib2.request

class captureAndSend():
	
	def __init__(self):
		#self.proc = ""
		#***************ENTER THE NODE NUMBER*************
		self.nodeNo = str(sys.argv[1])
		self.nodeNumber = self.nodeNo

		#************************CYBERA PRIVATE KEY ACCESS*************************
		self.sshName = str(sys.argv[2])

		#************************ADDRESS TO SEND THE FILES IN CYBERA*******************
		self.cybDirect = "/mnt/250/new"
		self.cyberaDirectory = "ubuntu@199.116.235.145:" + str(self.cybDirect)
		self.secondChar = None
		self.notUp = []
		
		#************************NUMBER OF WIFI DRIVERS********************
		self.numChipsets = int(sys.argv[3])

	def threads(self):

		#***********************STARTING THREADS***********************************
		chipset1 = "wlan0"
		name1 = "24node"
		print(chipset1)
		now = datetime.datetime.now()
		# with open("log.txt", "a") as log:
			# log.write(str(now.strftime("%Y-%m-%d %H:%M"))+ chipset + " \n ")
#		file.write(chipset)
		if self.numChipsets == 2: #in this case wlan0 is for 5ghz
			chipset2 = "wlan1"
			name1 = "/data/5node"
			name2 = "/data/24node"
			
			thread1Chip1 = threading.Thread(target = self.hopping5GHz, args = (chipset1,))
			
			thread2Chip1 = threading.Thread(target = self.capturing, args = (chipset1, name1))
			
			thread1Chip2 = threading.Thread(target = self.hopping24GHz, args = (chipset2,))
			thread2Chip2 = threading.Thread(target = self.capturing, args = (chipset2, name2))			
			
		
			thread1Chip1.start()
			thread2Chip1.start()
			thread1Chip2.start()
			thread2Chip2.start()

		elif self.numChipsets == 1:			
			thread1Chip1 = threading.Thread(target = self.hopping24GHz, args = (chipset1,))
			thread2Chip1 = threading.Thread(target = self.capturing, args = (chipset1, name1))		
		
			thread1Chip1.start()
			thread2Chip1.start()		


	def capturing(self, chipset, name):
		proc = ""
		loggerName = "/data/Logger.log"
		logging.basicConfig(filename = loggerName, format='%(threadName)s:%(message)s', level=logging.DEBUG)
		print(loggerName)
		progStart = 0
		iterations = 0		
		captureNo = 0
		start = 0
#wlxd8e743040e42
#		tcpdumpCommand = "sudo tshark -Pq -n -i " + chipset + " -Tfields -e wlan_radio.channel -e wlan.fc -e wlan.ta -e frame.time_epoch -E separator=/s"

		while True:
			iterations += 1
			#************************LOGGING******************************
			if iterations%30 == 0:
				logging.info(str(loggerName))
				now = datetime.datetime.now()
				logging.info(str(threading.currentThread().getName()))
				logging.info(str(now.strftime("%Y-%m-%d %H:%M")) + " number of iterations are: " + str(iterations) + "\n")

			#************************RUN THE CAPTURING FOR THE FIRST TIME***********************************
			if start == 0:
				#tcpdumpCommand = "tcpdump -i " + chipset + " -en -vvs 0 link[0]==0x80 -w node" + self.nodeNo + "-" + str(captureNo)
				
				#*****************READ FROM THE FILE TO CONTINUE THE PREVIOUS ONE***********************
				if progStart == 0:
					
					progStart = 1
					counterName = str(name) + ".txt"
					if os.path.isfile(counterName) == True:
						logging.info("we have the " + str(counterName) + "\n")
						if (os.stat(counterName).st_size == 0) == False:
							logging.info(str(counterName) + " is not empty\n")
							with open(counterName, "r") as txt:
								cont = txt.readline()
								logging.info(str(counterName) + " value is: " + str(int(cont)) + "\n")
						else:
							cont = 0
							logging.info(str(counterName) + " is empty so value is: " + str(int(cont)) + "\n")
					else:
						logging.info("we dont have the file so make it \n")
						os.system("touch " + str(counterName))
						cont = 0
						
					captureNo = int(cont)


				#name = "node.txt"


				fileName = str(name) + self.nodeNo + "." + str(captureNo)
				
				logging.info("capture number is: "+ str(captureNo) +" and file Name is: " + str(fileName) + "\n")
		
				proc = Popen(["tcpdump", "-i", chipset, "-en", "-vvs", "0", "link[0]!=0x80", "-w", str(fileName)])
				now = datetime.datetime.now()
				logging.info(str(now.strftime("%Y-%m-%d %H:%M"))+ " the process id is: " + str(proc.pid) + "\n")
				start += 1

			#**********************CHECK THE SIZE OF THE FILE*******************************
			elif start != 0:
				try:
					size = os.stat(fileName).st_size
					logging.info(str(now.strftime("%Y-%m-%d %H:%M")) + " size of the file is: " + str(size) + " for the file " + fileName + "\n")
				
					if size > 50000000:
						now = datetime.datetime.now()
						logging.info(str(now.strftime("%Y-%m-%d %H:%M"))+ "the size is big enough for saving \n")
#						self.file.write("the size is getting big enough to transfer")
						start = 0
						captureNo += 1
						with open(counterName, "w") as log:
							log.write(str(captureNo))						
						
						os.system("kill -9 " + str(proc.pid))
						

						thread3 = threading.Thread(target = self.sendingFunc, args = (fileName,))
						logging.info("uploading function is called")
						thread3.start()
						
							
				except FileNotFoundError:
					now = datetime.datetime.now()
					logging.info(str(now.strftime("%Y-%m-%d %H:%M"))+ "No file \n")
	#				self.file.write("File is not still created")
					print("continueing")
				time.sleep(10)

		
		#should call pipe for the subprocess to read and write simultaneously
	def sendingFunc(self, fileName):
		#*************CONNECTION TEST*********************
		testConn = os.system("ssh -q -o BatchMode=yes -o ConnectTimeout=2 -i " + self.sshName +  " ubuntu@199.116.235.145  echo ok 2>&1")
		logging.info("test connection is: " + str(testConn)+ "\n")

			
		if testConn == 0:
		#***************CONNECTION IS OK WITH CYBERA**********************

			if len(self.notUp) == 0:
			#*****************NO FILE IS IN THE QUEUE TO UPLOAD*****************
				#command = ("scp " + fileName + " -i " + self.sshName + " ubuntu@199.116.235.145:/home/ubuntu/data")
				proc1 = Popen(["scp", "-i", self.sshName , fileName, self.cyberaDirectory])
				sts = os.waitpid(proc1.pid, 0)
				logging.info("I am waiting\n")
				now = datetime.datetime.now()
				logging.info(str(now.strftime("%Y-%m-%d %H:%M"))+ "One file is uploaded " + str(fileName) + " sts is: " +str(sts) + "\n")
				logging.info("one file uploaded\n")
				
				proc2 = Popen(["scp", "-i", self.sshName , "/data/err", self.cyberaDirectory])
				proc3 = Popen(["scp", "-i", self.sshName , "/data/Logger.log", self.cyberaDirectory])
				sts = os.waitpid(proc2.pid, 0)
				sts = os.waitpid(proc3.pid, 0)
				logging.info("sending logs\n")
				now = datetime.datetime.now()
				logging.info("log files are uploaded")
								
				
				os.remove(fileName)
			else:

			#****************THERE ARE SOME FILES IN THE QUEUE THAT SHOULD BE UPLOADED****************
				logging.info("uploading chunks\n")
				self.notUp.append(fileName)
#				self.file.write("all files are: " + str(self.notUp))
				for i in range(len(self.notUp)):
					proc1 = Popen(["scp", "-i", self.sshName, self.notUp[i], self.cyberaDirectory])
					sts = os.waitpid(proc1.pid, 0)
					now = datetime.datetime.now()
					logging.info(str(now.strftime("%Y-%m-%d %H:%M"))+str(self.notUp[i]) + 'is done with sts of ' + str(sts) + '\n')
					print(str(self.notUp[i]) + " is done")					
					os.remove(self.notUp[i])
				self.notUp.clear()
				now = datetime.datetime.now()
				logging.info(str(now.strftime("%Y-%m-%d %H:%M"))+ "all files are uploaded\n")
#				self.file.write("all files are uploaded")
		else:
		#****************NO CONNECTION WITH CYBERA*****************************
			self.notUp.append(fileName)
			now = datetime.datetime.now()
			logging.info(str(now.strftime("%Y-%m-%d %H:%M"))+ "no connection, files are: " + str(self.notUp) + "\n")

		logging.info("uploading thread ended")	
		

	def hopping24GHz(self, chipset):
		i = 1
		while True:
			proc = Popen("iwconfig " + chipset + " channel " + str(i), stdout = PIPE, shell = True)
			time.sleep(1)
			if i == 1:
				i = 6
			elif i == 6:
				i = 11
			elif i == 11:
				i = 1
				
	def hopping5GHz(self, chipset):
		arrOfChannels = [36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 149, 153, 157, 161, 165]
		i = 0
		while True:
			proc = Popen("iwconfig " + chipset + " channel " + str(arrOfChannels[i]), stdout = PIPE, shell = True)
			time.sleep(1)
			i += 1
			if i == len(arrOfChannels):
				i = 0


if __name__ == "__main__":
	if len(sys.argv) == 4:
		obj = captureAndSend()
		obj.threads()
	else:
		print("enter nodeNo, ssh private key directory, and update the node.txt file, number of wifi drivers")
	
