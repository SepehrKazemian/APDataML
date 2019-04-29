import subprocess
import os

#ssh_address = '/mnt/250/new/node1/'
ssh_address = input("what is the cybera directory you want to scp files from? ")

client_address = input("what is the client directory you want to copy files into? ")
#client_address = "."

size = input("what the minimume size of the file should be? ")

command = '(ssh -i /home/Sepehr/.ssh/cybera ubuntu@199.116.235.145 find ' + ssh_address + ' -size +' + size + 'M)'

proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
(out, err) = proc.communicate()
list_of_files = out.decode("utf-8")
list_of_files = list_of_files.replace(ssh_address, '')
list_of_files = list_of_files.replace("\n", '')
list_of_files_array = []

for i in range(0, len(list_of_files), +16):
	list_of_files_array.append(list_of_files[i: i+16])

	
extra_array = [] #to remove the same AP files with the different last bit
for i in range(len(list_of_files_array)):
	if list_of_files_array[i] not in extra_array:
		ap_mac_char = list_of_files_array[i][0: 11]
		for j in range(i + 1, len(list_of_files_array)):
			if list_of_files_array[j][0: 11] == ap_mac_char:
				extra_array.append(list_of_files_array[j])
			
for i in range(len(extra_array)):
	list_of_files_array.remove(extra_array[i])
	
print(list_of_files_array)

for i in range(len(list_of_files_array)):
	os.system("scp -i /home/Sepehr/.ssh/cybera ubuntu@199.116.235.145:" + ssh_address + list_of_files_array[i] + " " + client_address)
			
#print(proc)