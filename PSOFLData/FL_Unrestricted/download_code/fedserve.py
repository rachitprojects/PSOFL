import os
import subprocess
import signal
pro = subprocess.Popen("sar -u 1 > cpuout.txt", stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
pro_mem = subprocess.Popen("sar -r 1 > memout.txt", stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)

import time
time.sleep(2)
mem_start = time.strftime("%I:%M:%S")

import torch
import pickle
import argparse

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from Server import Server
import config
import utils

LR = config.LR
first = True # First initializaiton control

logger.info('Preparing Sever.')
server = Server(0, config.SERVER_ADDR, config.SERVER_PORT, 'VGG5')

round_times = []

INIT_TIMES = []
TRAINING_TIMES = []
AGG_TIMES = []

timestamps = []

for r in range(config.R):
	logger.info('====================================>')
	logger.info('==> Round {:} Start'.format(r))
	server.scatter(["START", True])
	server.nettest(config.CLIENTS_LIST)
	s_time = time.time()
	server.initialize(config.CLIENTS_LIST, first, LR)
	init_time = time.time()
	server.train(thread_number= config.K, client_ips= config.CLIENTS_LIST)
	train_time = time.time()
	aggregrated_model = server.aggregate(config.CLIENTS_LIST)
	e_time = time.time()
	server.save()
	round_time = e_time - s_time

	INIT_TIMES.append(init_time - s_time)
	TRAINING_TIMES.append(train_time - init_time)
	AGG_TIMES.append(e_time - train_time)
	if first:
		timestamps.append([mem_start, time.strftime("%H:%M:%S", time.localtime(s_time)), time.strftime("%H:%M:%S", time.localtime(init_time)),
				time.strftime("%H:%M:%S", time.localtime(train_time)), time.strftime("%H:%M:%S", time.localtime(e_time))])
	else:
		timestamps.append([time.strftime("%H:%M:%S", time.localtime(s_time)), time.strftime("%H:%M:%S", time.localtime(init_time)),
			time.strftime("%H:%M:%S", time.localtime(train_time)), time.strftime("%H:%M:%S", time.localtime(e_time))])

	test_acc = server.test(r)

	logger.info('Round Finish')
	logger.info('==> Round Training Time: {:}'.format(round_time))
	round_times.append(round_time)

	logger.info('==> Reinitialization for Round : {:}'.format(r + 1))

	if r > 49:
		LR = config.LR * 0.1

	first = False

	print("Round times are ", round_times)
	print("Initialisation times are ", INIT_TIMES)
	print("Training Times are ", TRAINING_TIMES)
	print("Aggregation times are ", AGG_TIMES)
	print("Testing Times are ", config.TESTING_TIMES)
	print("Accuracies are ", config.ACCURACIES)

	print(timestamps)

os.system("pkill -x sar")
