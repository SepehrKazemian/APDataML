import os, sys
import threading, time
import subprocess
from subprocess import Popen, PIPE
import os
import datetime
#import urllib2.request

class captureAndSend():
	
	def __init__(self):
		self.proc = ""
		#***************ENTER THE NODE NUMBER*************
		self.nodeNo = str(sys.argv[1])
		self.nodeNumber = self.nodeNo

		#************************CYBERA PRIVATE KEY ACCESS*************************
		self.sshName = str(sys.argv[2])

		#************************ADDRESS TO SEND THE FILES IN CYBERA*******************
		self.cybDirect = "/mnt/250"
		self.cyberaDirectory = "ubuntu@199.116.235.145:" + str(self.cybDirect)
		self.secondChar = None
		self.notUp = []
#		self.file = open("log.txt", "w")

	def threads(self):

		#***********************STARTING THREADS***********************************
		chipset = "wlan0"
		print(chipset)
		now = datetime.datetime.now()
		with open("log.txt", "a") as log:
			log.write(str(now.strftime("%Y-%m-%d %H:%M"))+ chipset + " \n ")
#		file.write(chipset)
		thread = threading.Thread( target = self.hopping, args = (chipset,))
		thread2 = threading.Thread(target = self.capturing, args = (chipset,))
		print("thread name is: " + str(thread))
		thread.start()
		thread2.start()



	def capturing(self, chipset):
		progStart = 0
		iterations = 0		
		captureNo = 0
		start = 0
#wlxd8e743040e42
#		tcpdumpCommand = "sudo tshark -Pq -n -i " + chipset + " -Tfields -e wlan_radio.channel -e wlan.fc -e wlan.ta -e frame.time_epoch -E separator=/s"

		while True:
			iterations += 1
			#************************LOGGING******************************
			if iterations%10 == 0:
				now = datetime.datetime.now()
				with open("log.txt", "a") as log:
					log.write(str(now.strftime("%Y-%m-%d %H:%M")) + " number of iterations are: " + str(iterations) + "\n")


			#************************RUN THE CAPTURING FOR THE FIRST TIME***********************************
			if start == 0:
				tcpdumpCommand = "tcpdump -i " + chipset + " -en -vvs 0 link[0]==0x80 -w node" + self.nodeNo + "-" + str(captureNo)
				
				#*****************READ FROM THE FILE TO CONTINUE THE PREVIOUS ONE***********************
				if progStart == 0:

					with open("node.txt", "r") as txt:
						cont = txt.readline()

				#	for i in range(3, len(cont)):
#						if cont[i] == "e":
#							continue
#						elif cont[i] == "-":
#							change = 1
#						elif change == 0:
#							self.nodeNo += cont[i]
#						elif change == 1:
#							captureNo += cont[i]

					progStart = 1
					if cont != "":
						captureNo = int(cont[0:len(cont)-1])
						print(captureNo)

				name = "node.txt"
				with open(name, "w") as txt:
					txt.write(str(captureNo))
#					proc1 = Popen(["scp", "-i", self.sshName , fileName, self.cyberaDirectory])
#					print("file name is uploaded to be checked")

				fileName = "node" + self.nodeNo + "." + str(captureNo)
				
		
				self.proc = Popen(["tcpdump", "-i", chipset, "-en", "-vvs", "0", "link[0]==0x80", "-w", "node"+self.nodeNo+"."+str(captureNo)])
				print(self.proc.pid)
				now = datetime.datetime.now()
				with open("log.txt", "a") as log:                        
					log.write(str(now.strftime("%Y-%m-%d %H:%M"))+ " the process id is: " + str(self.proc.pid) + "\n") 
				start += 1

			#**********************CHECK THE SIZE OF THE FILE*******************************
			elif start != 0:
				try:
					statinfo = os.stat(fileName)
					size = statinfo.st_size
					with open("log.txt", "a") as log:
						log.write(str(now.strftime("%Y-%m-%d %H:%M")) + " size of the file is: " + str(size) + " for the file " + fileName + "\n")
				
					if size > 100000000:
						now = datetime.datetime.now()
						with open("log.txt", "a") as log:                                                              
							log.write(str(now.strftime("%Y-%m-%d %H:%M"))+ "the size is big so transfer \n")
#						self.file.write("the size is getting big enough to transfer")
						os.system("kill -9 " + str(self.proc.pid))
						start = 0

						#*************************SOLVE NAMING PROBLEM IN A LONG RUN********************
#						if captureNo =< 100000:
						captureNo += 1
#						elif captureNo > 100000:
#							if self.secondChar == None:
#								self.secondChar = 0
#								self.nodeNo += str(secondChar)
#							else:
#								self.nodeNo = self.nodeNo[0: len(self.nodeNo) - len(str(self.secondChar))]
#								self.secondChar += 1
#								self.nodeNo += str(secondChar)
#							captureNo = 0
						

						thread3 = threading.Thread(target = self.sendingFunc, args = (fileName,))
						with open("log.txt", "a") as log:
							log.write("uploading function is called")
						thread3.start()
						thread3.join()
						with open("log.txt", "a") as log:
							log.write("uploading thread comes back")
#						thread3.join()
						#upload via ssh and exit program
				except FileNotFoundError:
					now = datetime.datetime.now()
					with open("log.txt", "a") as log:                                                              
						log.write(str(now.strftime("%Y-%m-%d %H:%M"))+ "No file \n")
	#				self.file.write("File is not still created")
					print("continueing")
				time.sleep(10)

		
		#should call pipe for the subprocess to read and write simultaneously
	def sendingFunc(self, fileName):
		#*************CONNECTION TEST*********************
		testConn = os.system("ssh -q -o BatchMode=yes -o ConnectTimeout=2 -i " + self.sshName +  " ubuntu@199.116.235.145  echo ok 2>&1")
		with open("log.txt", "a") as log:
			log.write("test connection is: " + str(testConn)+ "\n")
		print(testConn)
		if testConn == 0:
		#***************CONNECTION IS OK WITH CYBERA**********************

			if len(self.notUp) == 0:
			#*****************NO FILE IS IN THE QUEUE TO UPLOAD*****************
				command = ("scp " + fileName + " -i " + self.sshName + " ubuntu@199.116.235.145:/home/ubuntu/data")
				proc1 = Popen(["scp", "-i", self.sshName , fileName, self.cyberaDirectory])
				sts = os.waitpid(proc1.pid, 0)
				with open("log.txt", "a") as log:
					log.write("I am waiting")
				now = datetime.datetime.now()
				with open("log.txt", "a") as log:                                                              
					log.write(str(now.strftime("%Y-%m-%d %H:%M"))+ "One file is uploaded " + str(fileName) + " sts is: " +str(sts) + "\n")
#				self.file.write("one file is uploaded with sts of " + str(sts) + " and name of " + str(fileName) + "\n")
				print("one file uploaded")
				os.system("rm "+ fileName)
			else:

			#****************THERE ARE SOME FILES IN THE QUEUE THAT SHOULD BE UPLOADED****************
				self.notUp.append(fileName)
#				self.file.write("all files are: " + str(self.notUp))
				for i in range(len(self.notUp)):
					proc1 = Popen(["scp", "-i", self.sshName, self.notUp[i], self.cyberaDirectory])
					sts = os.waitpid(proc1.pid, 0)
					now = datetime.datetime.now()
					with open("log.txt", "a") as log:                                                              
						log.write(str(now.strftime("%Y-%m-%d %H:%M"))+str(self.notUp[i]) + 'is done with sts of ' + str(sts) + '\n')
					print(str(self.notUp[i]) + " is done")					
					os.system("rm " + self.notUp[i])
				self.notUp.clear()
				now = datetime.datetime.now()
				with open("log.txt", "a") as log:                                                              
					log.write(str(now.strftime("%Y-%m-%d %H:%M"))+ "all files are uploaded\n")
#				self.file.write("all files are uploaded")
		else:
		#****************NO CONNECTION WITH CYBERA*****************************
			self.notUp.append(fileName)
			now = datetime.datetime.now()
			with open("log.txt", "a") as log:                                                              
				log.write(str(now.strftime("%Y-%m-%d %H:%M"))+ "no connection, files are: " + str(self.notUp) + "\n")
#			self.file.write("no connection to upload the file")
	#		self.file.write("list is " + str(self.notUp))
			
		

	def hopping(self, chipset):
		i = 1
		while True:
			proc = Popen("iwconfig " + chipset + " channel " + str(i), stdout = PIPE, shell = True)
			time.sleep(1.5)
			if i == 1:
				i = 6
			elif i == 6:
				i = 11
			elif i == 11:
				i = 1

	def func2():
		while True:
			print(2)


if __name__ == "__main__":
	if len(sys.argv) == 3:
		obj = captureAndSend()
		obj.threads()
	else:
		print("enter nodeNo, ssh private key directory, and update the node.txt file")
	
