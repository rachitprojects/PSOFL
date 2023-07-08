'''Some helper functions for FedAdapt, including:
	- get_local_dataloader: split dataset and get respective dataloader.
	- get_model: build the model according to location and split layer.
	- send_msg: send msg with type checking.
	- recv_msg: receive msg with type checking.
	- split_weights_client: split client's weights from holistic weights.
	- split_weights_server: split server's weights from holistic weights
	- concat_weights: concatenate server's weights and client's weights.
	- zero_init: zero initialization.
	- fed_avg: FedAvg aggregation.
	- norm_list: normlize each item in a list with sum.
	- str2bool.
'''
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


import pickle, struct, socket
from vgg import *
from config import *
import collections
import numpy as np
import re

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)

def get_local_dataloader(CLIENT_IDEX, cpu_count):
	indices = list(range(N))
	part_tr = indices[int((N/K) * CLIENT_IDEX) : int((N/K) * (CLIENT_IDEX+1))]

	transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
	trainset = torchvision.datasets.CIFAR10(
		root=dataset_path, train=True, download=True, transform=transform_train)
	subset = Subset(trainset, part_tr)
	trainloader = DataLoader(
		subset, batch_size=B, shuffle=True, num_workers=cpu_count)

	classes = ('plane', 'car', 'bird', 'cat', 'deer',
		   'dog', 'frog', 'horse', 'ship', 'truck')
	return trainloader,classes

def get_model(location, model_name, layer, device, cfg):
	cfg = cfg.copy()
	net = VGG(location, model_name, layer, cfg)
	net = net.to(device)
	logger.debug(str(net))
	return net

def send_msg(sock, msg):
	msg_pickle = pickle.dumps(msg)
	sock.sendall(struct.pack(">I", len(msg_pickle)))
	sock.sendall(msg_pickle)
	logger.debug(msg[0]+'sent to'+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

def recv_msg(sock, expect_msg_type=None):
	msg_len = struct.unpack(">I", sock.recv(4))[0]
	msg = sock.recv(msg_len, socket.MSG_WAITALL)
	msg = pickle.loads(msg)
	logger.debug(msg[0]+'received from'+str(sock.getpeername()[0])+':'+str(sock.getpeername()[1]))

	if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
		raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
	return msg

def split_weights_client(weights,cweights):
	for key in cweights:
		assert cweights[key].size() == weights[key].size()
		cweights[key] = weights[key]
	return cweights

def split_weights_server(weights,cweights,sweights):
	ckeys = list(cweights)
	skeys = list(sweights)
	keys = list(weights)

	for i in range(len(skeys)):
		assert sweights[skeys[i]].size() == weights[keys[i + len(ckeys)]].size()
		sweights[skeys[i]] = weights[keys[i + len(ckeys)]]

	return sweights

def concat_weights(weights,cweights,sweights):
	concat_dict = collections.OrderedDict()

	ckeys = list(cweights)
	skeys = list(sweights)
	keys = list(weights)

	for i in range(len(ckeys)):
		concat_dict[keys[i]] = cweights[ckeys[i]]

	for i in range(len(skeys)):
		concat_dict[keys[i + len(ckeys)]] = sweights[skeys[i]]

	return concat_dict



def zero_init(net):
	for m in net.modules():
		if isinstance(m, nn.Conv2d):
			init.zeros_(m.weight)
			if m.bias is not None:
				init.zeros_(m.bias)
		elif isinstance(m, nn.BatchNorm2d):
			init.zeros_(m.weight)
			init.zeros_(m.bias)
			init.zeros_(m.running_mean)
			init.zeros_(m.running_var)
		elif isinstance(m, nn.Linear):
			init.zeros_(m.weight)
			if m.bias is not None:
				init.zeros_(m.bias)
	return net

def fed_avg(zero_model, w_local_list, totoal_data_size):
	keys = w_local_list[0][0].keys()
	for k in keys:
		for w in w_local_list:
			beta = float(w[1]) / float(totoal_data_size)
			if 'num_batches_tracked' in k:
				zero_model[k] = w[0][k]
			else:
				zero_model[k] += (w[0][k] * beta)

	return zero_model

#int(x[0:2]) * 3600 + int(x[3:5]) * 60 + int(x[6:8])
def get_nearest(time, list_of_time):

    if time[0:2][-1] == ":":
        time = "0" + time
    for x in range(len(list_of_time)):
        if list_of_time[x][0:2][-1] == ":":
            list_of_time[x] = "0" + list_of_time[x]
    min_time = list_of_time[min(range(len(list_of_time)), key = lambda i: abs(int(list_of_time[i][0:2]) * 3600 + int(list_of_time[i][3:5]) * 60 + int(list_of_time[i][6:8])
										- ( int(time[0:2]) * 3600 + int(time[3:5]) * 60 + int(time[6:8]) ) ))]
    return min_time

def get_mem_usage(start_time, end_time):
    with open("memout.txt", 'r') as memdet:
        data = memdet.readlines()
        util_dict = {}
        for i in range(3,len(data)):
            #item = data[i].split(" ")
            item = re.sub(" +", " ", data[i]).lstrip().split(" ")
            #print(item)
            util_dict[item[0]] = item[3]
        if start_time not in util_dict.keys():
            start_time = get_nearest(start_time, list(util_dict))
        if end_time not in util_dict.keys():
            end_time = get_nearest(end_time, list(util_dict))

        utils = 0
        mem_utils = list(util_dict.values())[list(util_dict).index(start_time):list(util_dict).index(end_time) + 1]
        for util in mem_utils:
            utils += float(util)
        MEM_USAGE_FINE.append(mem_utils)
        num_utils = len(mem_utils)
        if num_utils == 0:
            num_utils = 1
        return utils / num_utils



#MEM_USAGE_FINE = []

    #print(start_time)
    #print(end_time)
#    print(util_dict[end_time])

    #print(util_dict)
    #print(list(util_dict.values()))
    #print("Average Memory Utilization  = ", utils / len(mem_utils))




def get_utilisation(cpu_start, cpu_end):
    with open("cpuout.txt", 'r') as cpudet:
        data = cpudet.readlines()
        util_dict = {}
        for i in range(3,len(data)):
            #item = data[i].split("    ")
            item = re.sub(" +", " ", data[i]).lstrip().split(" ")
            util_dict[item[0][0:8]] = item[2]
        #print(util_dict)
        if cpu_start not in util_dict.keys():
            cpu_start = get_nearest(cpu_start, list(util_dict))
        if cpu_end not in util_dict.keys():
            cpu_end = get_nearest(cpu_end, list(util_dict))

        utils = 0
        cpu_utils = list(util_dict.values())[list(util_dict).index(cpu_start):list(util_dict).index(cpu_end) + 1]
        for util in cpu_utils:
            utils += float(util)
        #print(cpu_utils)
        CPU_UTILS_FINE.append(cpu_utils)
        # print("Average CPU Utilization  = ", utils / len(cpu_utils))
        num_utils = len(cpu_utils)
        if num_utils == 0:
            num_utils = 1
        return utils / num_utils


def norm_list(alist):	
	return [l / sum(alist) for l in alist]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#print(get_nearest("8:13:15", ["08:13:16", "09:13:10"]))
