import re
import string
import ast

import matplotlib.pyplot as plt
from statistics import mean, stdev

with open("flunresdata","r") as file1:
	strfile=file1.read()
	strlist=list()
	x=re.search("Server",strfile)
	t1=x.span()
	print(t1)
	training_times = []
	#start=0
	#end=start
	for i in range(1,11):
		srchstr="Device "+str(i)
#print("searching string is   ",srchstr)
		if(srchstr=="Device 10"):
			x = re.search(srchstr, strfile)
			start = x.span()[1]
			end=len(strfile)
		else:
			x=re.search(srchstr,strfile)
			start = x.span()[1]
			end = re.search("Device " + str(i + 1), strfile).span()[0]
		temp_str = strfile[start:end]
		train_ind_start = re.search("Network Speed per round", temp_str).span()[1]
		train_ind_end = re.search("The Network Utilisations are", temp_str).span()[0]
		training_times.append(eval(temp_str[train_ind_start:train_ind_end]))

	mean_nets = []
	for x in training_times:
		mean_nets.append(mean(x))
	
	print(mean_nets)
	#print(training_times)
	#max_train_times = []
	#for x in range(len(training_times[0])):
	#	max_train_times.append(max([training_times[y][x] for y in range(10)]))
	#print(max_train_times)
	#train_times = []
	#for x in training_times:
	#	print(mean(x), stdev(x))
	#	train_times.append(mean(x))
	#print(mean(net_speed))
		#temp_str = temp_str[train_ind_star:train_in
		#training_times.append([temp_str[train_ind_start:train_ind_end].split(",")])
	#for x in range(len(training_times)):
	#	floated = []
	#	for y in training_times[x]:
	#		if y[0] == "[":
	#			p = y[1:]
	#			floated.append(float(p))
	#		elif y[-1] == "]":
	#			p = y[:-1]
	#			floated.append(float(p))
	#		else:
	#			floated.append(float(y))
	#	training_times[x] = floated


	#print(training_times[3])

	#		t2=x.span()
	#		end=t2[0]
	#tempstr=strfile[start:end]
	#strlist.append(tempstr)
	#start=end

#	rounds = [x for x in range(0, 100)]

#	plt.ylim(125, 140)
#	for x in training_times:
#		print(len(x))
#		plt.plot(rounds, x)
#	plt.savefig("traintimefedadapt.png")
#	tikzplotlib.save("fedadapt.tex")
print()
print()
#print(strlist[0])
print()
print()
