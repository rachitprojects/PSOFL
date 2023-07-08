import re

def get_nearest(time, list_of_time, y):

    #print(time, y)
    if time[0:2][-1] == ":":
        time = "0" + time
    for x in range(len(list_of_time)):
        if list_of_time[x][0:2][-1] == ":":
            list_of_time[x] = "0" + list_of_time[x]
    min_time = list_of_time[min(range(len(list_of_time)), key = lambda i: abs(int(list_of_time[i][0:2]) * 3600 + int(list_of_time[i][3:5]) * 60 + int(list_of_time[i][6:8])
                                                                                - ( int(time[0:2]) * 3600 + int(time[3:5]) * 60 + int(time[6:8]) ) ))]
    return min_time


with open("flunresdata") as cpufile:
	filedata = cpufile.readlines()
	counter = 0
	timestamp_dev = []
	for x in filedata:
		if x[0:2] == "[[":
			timestamp_dev.append(eval(x))
			print()
			counter += 1
	device_cpu_util = [[] for x in range(1, 11)]
	for y in range(1, 11):
		#device_cpu_util = [[] for x in range(1, 11)]
		for timestamps in timestamp_dev[y]:
			with open("cpuoutdev" + str(y) + ".txt", 'r') as cpudet:
			#for timestamps in timestamp_dev[y]:
				#print(timestamps)
				data = cpudet.readlines()
				util_dict = {}
				for i in range(3,len(data)):
					item = re.sub(" +", " ", data[i]).lstrip().split(" ")
					util_dict[item[0][0:8]] = item[4]
				cpu_start = timestamps[0]
				cpu_end = timestamps[-1]
				origin_cpu_start, origin_cpu_end = cpu_start, cpu_end
				#print("Util dict is ", list(util_dict))
				if cpu_start not in util_dict.keys():
					cpu_start = get_nearest(cpu_start, list(util_dict), y)
				if cpu_end not in util_dict.keys():
					cpu_end = get_nearest(cpu_end, list(util_dict), y)

        			#utils = 0
				cpu_utils = list(util_dict.values())[list(util_dict).index(cpu_start):list(util_dict).index(cpu_end) + 1]
				cpu_utils = [float(x) for x in cpu_utils]
				device_cpu_util[y - 1].append(max(cpu_utils) - min(cpu_utils))
				#if len(cpu_utils) > 14:
				#	print(len(cpu_utils), origin_cpu_start, origin_cpu_end, y)
				#print(cpu_utils)

	# for utl in device_cpu_util:
	# 	print(utl)
	global_average = []
	for x in range(len(device_cpu_util[0])):
		global_average.append(sum([device_cpu_util[y][x] for y in range(10)]) / 10)
		print()
		print()
	print(global_average)

