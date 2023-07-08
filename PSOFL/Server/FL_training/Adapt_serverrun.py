import time
import os
import signal
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
from statistics import mean

LR = config.LR

logger.info('Preparing Sever.')
server = Server(0, config.SERVER_ADDR, config.SERVER_PORT, 'VGG5')

training_times = []

config.split_layer = server.gen_layer(0)
server.scatter(["SPLIT_LAYER", config.split_layer])

for r in range(config.R):
	logger.info('====================================>')
	logger.info('==> Round {:} Start'.format(r))
	server.scatter(["START", True])
	server.nettest(config.CLIENTS_LIST)

	s_time = time.time()
	server.initialize(config.split_layer, LR, r)
	server.train(thread_number= config.K, client_ips= config.CLIENTS_LIST)
	aggregrated_model = server.aggregate(config.CLIENTS_LIST)
	e_time = time.time()
	cpu_end = time.strftime("%H:%M:%S")
	trianing_time = e_time - s_time

	test_acc = server.test(r)
	logger.info('Round Finish')
	logger.info('==> Round Training Time: {:}'.format(trianing_time))
	training_times.append(trianing_time)

	logger.info('==> Reinitialization for Round : {:}'.format(r + 1))

	server.gather(config.CLIENTS_LIST)
	split_layers = server.adaptive_offload(r)

	if r > 49:
		LR = config.LR * 0.1

	print("Round times are ", training_times)
	print("Average Round times are ", mean(training_times))
	print("Testing Times are ", config.TESTING_TIMES)
	print("Average Testing Times are ", mean(config.TESTING_TIMES))
	print("Accuracies are ", config.ACCURACIES)
	print("Round Fitnesses are ", server.round_fitness)

os.system("pkill -x sar")
