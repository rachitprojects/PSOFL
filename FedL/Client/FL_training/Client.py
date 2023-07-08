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
from Communicator import *

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)

class Client(Communicator):
	def __init__(self, index, ip_address, server_addr, server_port, model_name):
		super(Client, self).__init__(index, ip_address)
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model_name = model_name
		self.uninet = utils.get_model('Unit', model_name, config.model_len - 1, self.device, config.model_cfg)

		logger.info('Connecting to Server.')
		self.sock.connect((server_addr,server_port))

	def initialize(self, first, LR):
		self.net_util = 0
		if first:

			logger.debug('Building Model.')
			self.net = utils.get_model('Client', self.model_name, 6, self.device, config.model_cfg)
			logger.debug(self.net)
			self.criterion = nn.CrossEntropyLoss()

		self.optimizer = optim.SGD(self.net.parameters(), lr=LR,
					  momentum=0.9)
		logger.debug('Receiving Global Weights..')
		received = self.recv_msg(self.sock)
		weights = received[0][1]
		self.net_util += received[1]
		print("Size of Received weights is ", received[1])
		self.net.load_state_dict(weights)
		logger.debug('Initialize Finished')

	def nettest(self):
                print("Started Network Speed Testing")
                network_time_start = time.time()
                msg = ['MSG_TEST_NETWORK', self.uninet.cpu().state_dict()]
                self.send_msg(self.sock, msg)
                msg_total = self.recv_msg(self.sock,'MSG_TEST_NETWORK')
                network_time_end = time.time()
                network_speed = (2 * config.model_size  * 8) / (network_time_end - network_time_start) #Mbit/s
                config.NETWORK_SPEEDS.append(network_speed)
                logger.info('Network speed is {:}'.format(network_speed))

	def train(self, trainloader):
		self.net.to(self.device)
		self.net.train()
		for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
			inputs, targets = inputs.to(self.device), targets.to(self.device)
			self.optimizer.zero_grad()
			outputs = self.net(inputs)
			loss = self.criterion(outputs, targets)
			loss.backward()
			self.optimizer.step()

	def upload(self):
		msg = ['MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER', self.net.cpu().state_dict()]
		agg_model_util = self.send_msg(self.sock, msg)
		print("Aggregation Size is ", agg_model_util)
		self.net_util += agg_model_util
