import os
import subprocess
import signal
pro = subprocess.Popen("sar -u 1 > cpuout.txt", stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
pro_mem = subprocess.Popen("sar -r 1 > memout.txt", stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)

ip_add = subprocess.check_output("ifconfig | head -n 2", shell=True)
num_cpu = subprocess.check_output("lscpu | tail -n +4 | head -n 2 | grep CPU", shell=True)
mem_max = subprocess.check_output("free -m | head -n 2", shell=True)
net_script = subprocess.check_output("cat ~/Desktop/netscript2.sh", shell=True)

import time
time.sleep(2)
mem_start = time.strftime("%I:%M:%S")

import torch
import socket
import multiprocessing
import argparse

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from Client import Client
import config
import utils

parser=argparse.ArgumentParser()
parser.add_argument('--offload', help='FedAdapt or classic FL mode', type= utils.str2bool, default= False)
args=parser.parse_args()

ip_address = '192.168.10.47'
index = 0
datalen = config.N / config.K
split_layer = config.split_layer[index]
LR = config.LR

logger.info('Preparing Client')
client = Client(index, ip_address, config.SERVER_ADDR, config.SERVER_PORT, datalen, 'VGG5', split_layer)

offload = args.offload
first = True # First initializaiton control
first_init_start = time.time()
client.initialize(split_layer, offload, first, LR)
first_init_end = time.time()
print("First Init took ", first_init_end - first_init_start)
first = False

logger.info('Preparing Data.')
cpu_count = multiprocessing.cpu_count()
trainloader, classes= utils.get_local_dataloader(index, cpu_count)

if offload:
	logger.info('FedAdapt Training')
else:
	logger.info('Classic FL Training')

first_util = True
ROUND_TIMES = []
RE_INIT_TIMES = []
TRAIN_TIMES = []
AGG_SEND_TIMES = []

NET_UTILS = []
INIT_NET_UTILS = []
INIT_NET_UTILS.append(client.net_util)
split_layer_schemes = []

timestamps = []

for r in range(config.R):
	logger.info('====================================>')
	logger.info('ROUND: {} START'.format(r))
	client.recv_msg(client.sock, "START")
	round_start = time.time()
	training_time, comm_tuple = client.train(trainloader)
	round_train = time.time()

	logger.info('ROUND: {} END'.format(r))

	logger.info('==> Waiting for aggregration')
	client.upload()
	agg_send_time = time.time()

	logger.info('==> Reinitialization for Round : {:}'.format(r + 1))

	if offload:
		received_split = client.recv_msg(client.sock)
		config.split_layer = received_split[0][1]
		client.net_util += received_split[1]
		split_layer_schemes.append(config.split_layer)

	if r > 49:
		LR = config.LR * 0.1

	NET_UTILS.append(client.net_util)
	client.reinitialize(config.split_layer[index], offload, first, LR)

	round_end = time.time()

	INIT_NET_UTILS.append(client.net_util)
	print("Time for entire round is ", round_end - round_start)
	ROUND_TIMES.append(round_end - round_start)
	TRAIN_TIMES.append(round_train - round_start)
	AGG_SEND_TIMES.append(agg_send_time - round_train)
	RE_INIT_TIMES.append(round_end - agg_send_time)
	config.SEND_RECV_TRAIN_TIME.append(comm_tuple)

	if first_util:
		first_util = False
		timestamps.append([mem_start, time.strftime("%I:%M:%S", time.localtime(round_start)), time.strftime("%I:%M:%S", time.localtime(round_train)),
                                time.strftime("%I:%M:%S", time.localtime(agg_send_time)), time.strftime("%I:%M:%S", time.localtime(round_end))])
	else:
		timestamps.append([time.strftime("%I:%M:%S", time.localtime(round_start)), time.strftime("%I:%M:%S", time.localtime(round_train)),
                                time.strftime("%I:%M:%S", time.localtime(agg_send_time)), time.strftime("%I:%M:%S", time.localtime(round_end))])

	logger.info('==> Reinitialization Finish')

	print("Device details are : ", index, ip_address)
	print("IP address string is ", ip_address)
	print("Device index is ", index)
	print("ifconfig IP address is ", ip_add)
	print("Number of CPU is ", num_cpu)
	print("Amount of Memory is ", mem_max)
	print("Network Script Output is ", net_script)
	print("First init took ", first_init_end - first_init_start)
	print("Total Round Execution time ", ROUND_TIMES)
	print("Initialisation Times are ", RE_INIT_TIMES)
	print("Train times are ", TRAIN_TIMES)
	print("Time for aggregation is ", AGG_SEND_TIMES)
	print("Network Speed per round", config.NETWORK_SPEEDS)
	print("The Network Utilisations are ", NET_UTILS)
	print("The initialisation net utils are ", INIT_NET_UTILS)
	print(timestamps)
	print("Offloading Points are ", split_layer_schemes)
	print("Mean, Sum, Stdev of Communication Time per round is ", config.SEND_RECV_TRAIN_TIME)

os.system("pkill -x sar")
