import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
import numpy as np
import sys

sys.path.append('../')
import config
import utils
from statistics import mean, stdev
from Communicator import *

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)

class Client(Communicator):
	def __init__(self, index, ip_address, server_addr, server_port, datalen, model_name, split_layer):
		super(Client, self).__init__(index, ip_address)
		self.datalen = datalen
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model_name = model_name
		self.uninet = utils.get_model('Unit', self.model_name, config.model_len-1, self.device, config.model_cfg)

		logger.info('Connecting to Server.')
		self.sock.connect((server_addr,server_port))

	def initialize(self, split_layer, offload, first, LR):
		self.net_util = 0
		if offload or first:
			self.split_layer = split_layer

			logger.debug('Building Model.')
			self.net = utils.get_model('Client', self.model_name, self.split_layer, self.device, config.model_cfg)
			logger.debug(self.net)
			self.criterion = nn.CrossEntropyLoss()

		self.optimizer = optim.SGD(self.net.parameters(), lr=LR,
					  momentum=0.9)
		logger.debug('Receiving Global Weights..')
		received = self.recv_msg(self.sock)
		weights = received[0][1]
		self.net_util += received[1]
		if self.split_layer == (config.model_len -1):
			self.net.load_state_dict(weights)
		else:
			pweights = utils.split_weights_client(weights,self.net.state_dict())
			self.net.load_state_dict(pweights)
		logger.debug('Initialize Finished')

	def train(self, trainloader):
		# Network speed test
		net_test_util_send = 0
		net_test_util_recv = 0
		net_util_train_send = 0
		net_util_train_recv = 0
		net_util_send_iter = 0
		network_time_start = time.time()
		msg = ['MSG_TEST_NETWORK', self.uninet.cpu().state_dict()]
		net_test_util_send = self.send_msg(self.sock, msg)
		msg_total = self.recv_msg(self.sock,'MSG_TEST_NETWORK')
		network_time_end = time.time()
		msg = msg_total[0][1]
		net_test_util_recv = msg_total[1]
		network_speed = (2 * config.model_size * 8) / (network_time_end - network_time_start) #Mbit/s 
		config.NETWORK_SPEEDS.append(network_speed)
		logger.info('Network speed is {:}'.format(network_speed))
		msg = ['MSG_TEST_NETWORK', self.ip, network_speed]
		self.net_util += self.send_msg(self.sock, msg)
		self.net_util += net_test_util_send + net_test_util_recv

		# Training start
		s_time_total = time.time()
		time_training_c = 0
		wait_time = []
		self.net.to(self.device)
		self.net.train()
		if self.split_layer == (config.model_len -1): # No offloading training
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				self.optimizer.zero_grad()
				outputs = self.net(inputs)
				loss = self.criterion(outputs, targets)
				loss.backward()
				self.optimizer.step()
		else: # Offloading training
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				self.optimizer.zero_grad()
				outputs = self.net(inputs)

				msg = ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', outputs.cpu(), targets.cpu()]
				send_start = time.time()
				net_util_train_send = self.send_msg(self.sock, msg)
				self.net_util += net_util_train_send

				# Wait receiving server gradients
				received_grad = self.recv_msg(self.sock)
				recv_end = time.time()
				gradients = received_grad[0][1].to(self.device)
				net_util_train_recv = received_grad[1]
				self.net_util += net_util_train_recv

				outputs.backward(gradients)
				self.optimizer.step()
				wait_time.append(recv_end - send_start)
		e_time_total = time.time()
		logger.info('Total time: ' + str(e_time_total - s_time_total))
		#config.TRAINING_TIMES.append(e_time_total - s_time_total)

		training_time_pr = (e_time_total - s_time_total) / int((config.N / (config.K * config.B)))
		logger.info('training_time_per_iteration: ' + str(training_time_pr))

		msg = ['MSG_TRAINING_TIME_PER_ITERATION', self.ip, training_time_pr]
		self.net_util += self.send_msg(self.sock, msg)

		print(wait_time)
		mean_comm_time = 0
		sum_comm_time = 0
		stdev_comm_time = 0
		if wait_time:
			mean_comm_time = mean(wait_time)
			sum_comm_time = sum(wait_time)
			stdev_comm_time = stdev(wait_time)

		return e_time_total - s_time_total, (mean_comm_time, sum_comm_time, stdev_comm_time)

	def upload(self):
		msg = ['MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER', self.net.cpu().state_dict()] 
		agg_model_util = self.send_msg(self.sock, msg)
		print("Aggregation Size is ", agg_model_util)
		self.net_util += agg_model_util

	def reinitialize(self, split_layers, offload, first, LR):
		self.initialize(split_layers, offload, first, LR)

