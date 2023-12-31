import subprocess
import os
import signal
pro = subprocess.Popen("sar -u 1 > cpuout.txt", stdout=subprocess.PIPE, shell=True, preexec_fn = os.setsid)
ip_add = subprocess.check_output("ifconfig | head -n 2", shell=True)
num_cpu = subprocess.check_output("lscpu | tail -n +4 | head -n 2 | grep CPU", shell=True)
mem_max = subprocess.check_output("free -m | head -n 2", shell=True)
net_script = subprocess.check_output("cat ~/Desktop/netscript2.sh ", shell=True)

import time
time.sleep(2)
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
import subprocess

ip_address = '192.168.10.23'
index = 1
split_layer = config.split_layer[index]
LR = config.LR

logger.info('Preparing Client')
client = Client(index, ip_address, config.SERVER_ADDR, config.SERVER_PORT, 'VGG5', split_layer)

logger.info('Preparing Data.')
cpu_count = multiprocessing.cpu_count()
trainloader, classes= utils.get_local_dataloader(index, cpu_count)

ROUND_TIMES = []
CPU_UTILS = []

split_layer = client.recv_msg(client.sock, "SPLIT_LAYER")[0][1][index]

for r in range(config.R):
	print(split_layer)
	logger.info('====================================>')
	logger.info('ROUND: {} START'.format(r))

	client.recv_msg(client.sock, "START")
	client.nettest()
	cpu_start = time.strftime("%I:%M:%S")
	round_start = time.time()
	client.initialize(split_layer, LR)
	training_time = client.train(trainloader)

	logger.info('==> Waiting for aggregration')
	client.upload()
	logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
	round_end = time.time()
	cpu_end = time.strftime("%I:%M:%S")

	if r > 49:
		LR = config.LR * 0.1

	print("Time for entire round is ", round_end - round_start)
	ROUND_TIMES.append(round_end - round_start)

	print(cpu_start, cpu_end)
	cpu_util = utils.get_utilisation(cpu_start, cpu_end)
	print("CPU Utilisation is ", cpu_util)
	CPU_UTILS.append(cpu_util)
	client.send_msg(client.sock, ["MSG_OPTIMIZATION_DATA", round_end - round_start, cpu_util])


	config.split_layer = client.recv_msg(client.sock)[0][1]
	split_layer = config.split_layer[index]

	print("The Network Speeds per round are ", config.NETWORK_SPEEDS)
	print("Device Details are ", index, ip_address)
	print("IP address string is ", ip_address)
	print("ifconfig IP address is ", ip_add)
	print("Number of CPU is ", num_cpu)
	print("Amount of Memory is ", mem_max)
	print("Network Script Output is ", net_script)
	print("Total Round Execution time ", ROUND_TIMES)
	print("Network Speed per round", config.NETWORK_SPEEDS)
	print("The CPU utilisations are ", CPU_UTILS)

os.system("pkill -x sar")





