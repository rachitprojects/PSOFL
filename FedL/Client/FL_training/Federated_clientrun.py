import os
import subprocess
import signal
pro = subprocess.Popen("sar -u 1 > cpuout.txt", stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
pro_mem = subprocess.Popen("sar -r 1 > memout.txt", stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
ip_add = subprocess.check_output("ifconfig | head -n 2 | grep inet", shell=True)
num_cpu = subprocess.check_output("lscpu | tail -n +4 | head -n 2 | grep CPU", shell=True)
mem_max = subprocess.check_output("free -m | head -n 2 | grep Mem", shell=True)
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

ip_address = '192.168.10.47'
index = 0
LR = config.LR

logger.info('Preparing Client')
client = Client(index, ip_address, config.SERVER_ADDR, config.SERVER_PORT, 'VGG5')

first = True

logger.info('Preparing Data.')
cpu_count = multiprocessing.cpu_count()
trainloader, classes= utils.get_local_dataloader(index, cpu_count)

logger.info('Classic FL Training')

INIT_TIMES = []
TRAIN_TIMES = []
AGG_SEND_TIMES = []
ROUND_TIMES = []
NET_UTILS = []

timestamps = []

for r in range(config.R):
	logger.info('====================================>')
	logger.info('ROUND: {} START'.format(r))
	client.recv_msg(client.sock, "START")
	client.nettest()
	round_start = time.time()
	client.initialize(first, LR)
	round_init = time.time()
	training_time = client.train(trainloader)
	round_train = time.time()
	logger.info('ROUND: {} END'.format(r))
	logger.info('==> Waiting for aggregration')
	client.upload()
	logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
	round_end = time.time()

	if r > 49:
		LR = config.LR * 0.1

	print("The net utilisations are ", client.net_util)
	NET_UTILS.append(client.net_util)
	print("Time for initialisation is ", round_init - round_start)
	print("Time for training is ", round_train - round_init)
	print("Time for aggregation is ", round_end - round_train)
	print("Time for entire round is ", round_end - round_start)
	ROUND_TIMES.append(round_end - round_start)
	INIT_TIMES.append(round_init - round_start)
	TRAIN_TIMES.append(round_train - round_init)
	AGG_SEND_TIMES.append(round_end - round_train)

	if first:
		timestamps.append([mem_start, time.strftime("%I:%M:%S", time.localtime(round_start)), time.strftime("%I:%M:%S", time.localtime(round_init)),
                                time.strftime("%I:%M:%S", time.localtime(round_train)), time.strftime("%I:%M:%S", time.localtime(round_end))])
	else:
		timestamps.append([time.strftime("%I:%M:%S", time.localtime(round_start)), time.strftime("%I:%M:%S", time.localtime(round_init)),
                                time.strftime("%I:%M:%S", time.localtime(round_train)), time.strftime("%I:%M:%S", time.localtime(round_end))])

	first = False

	print("IP address string is : ", ip_address)
	print("Device Index is ", index)
	print("ifconfig IP address is ", ip_add)
	print("Number of CPU is ", num_cpu)
	print("Amount of Memory is ", mem_max)
	print("Network Script output is ", net_script)
	print("Total Round Execution time ", ROUND_TIMES)
	print("Initialisation Times are ", INIT_TIMES)
	print("Train times are ", TRAIN_TIMES)
	print("Time for aggregation is ", AGG_SEND_TIMES)
	print("Network Speed per round", config.NETWORK_SPEEDS)
	print("The Network Utilisations are ", NET_UTILS)
	print(timestamps)

os.system("pkill -x sar")

