
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import threading
import tqdm
import random
import numpy as np

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from Communicator import *
import utils
import config
import time

np.random.seed(0)
torch.manual_seed(0)

class Server(Communicator):
	def __init__(self, index, ip_address, server_port, model_name):
		super(Server, self).__init__(index, ip_address)
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model_name = model_name
		self.port = server_port
		self.sock.bind((self.ip, self.port))
		self.client_socks = {}

		while len(self.client_socks) < config.K:
			self.sock.listen(5)
			logger.info("Waiting Incoming Connections.")
			(client_sock, (ip, port)) = self.sock.accept()
			logger.info('Got connection from ' + str(ip))
			logger.info(client_sock)
			self.client_socks[str(ip)] = client_sock
			config.CLIENTS_LIST.append(str(ip))
		self.uninet = utils.get_model('Unit', self.model_name, config.model_len-1, self.device, config.model_cfg)

		self.transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
		self.testset = torchvision.datasets.CIFAR10(root=config.dataset_path, train=False, download=True, transform=self.transform_test)
		self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=4)
		#uninet_weights = torch.load("fedlearn.pth")['model_state_dict']
		#self.uninet.load_state_dict(uninet_weights)

	def _thread_init(self, sock, msg):
		self.send_msg(sock, msg)

	def initialize(self, client_ips, first, LR):
		if first:
			self.nets = {}
			for client_ip in client_ips:
				self.nets[client_ip] = utils.get_model('Server', self.model_name, config.model_len - 1, self.device, config.model_cfg)
			self.criterion = nn.CrossEntropyLoss()

		msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', self.uninet.state_dict()]

		self.init_threads = {}
		for i in self.client_socks:
			self.init_threads[i] = threading.Thread(target=self._thread_init, args=(self.client_socks[i], msg))
			self.init_threads[i].start()

		for x in self.init_threads:
			self.init_threads[x].join()


	def nettest(self, client_ips):
                self.net_threads = {}
                for i in range(len(client_ips)):
                        self.net_threads[client_ips[i]] = threading.Thread(target=self._thread_network_testing, args=(client_ips[i],))
                        self.net_threads[client_ips[i]].start()

                for i in range(len(client_ips)):
                        self.net_threads[client_ips[i]].join()

	def train(self, thread_number, client_ips):
		self.threads = {}
		for i in range(len(client_ips)):
			self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_no_offloading, args=(client_ips[i],))
			logger.info(str(client_ips[i]) + ' no offloading training start')
			self.threads[client_ips[i]].start()

		for i in range(len(client_ips)):
			self.threads[client_ips[i]].join()

	def _thread_network_testing(self, client_ip):
		msg = self.recv_msg(self.client_socks[client_ip], 'MSG_TEST_NETWORK')
		msg_net = ['MSG_TEST_NETWORK', self.uninet.cpu().state_dict()]
		self.send_msg(self.client_socks[client_ip], msg_net)

	def _thread_training_no_offloading(self, client_ip):
		pass

	def aggregate(self, client_ips):
		w_local_list =[]
		for i in range(len(client_ips)):
			msg = self.recv_msg(self.client_socks[client_ips[i]], 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER')
			w_local = (msg[1],config.N / config.K)
			w_local_list.append(w_local)
		zero_model = utils.zero_init(self.uninet).state_dict()
		aggregrated_model = utils.fed_avg(zero_model, w_local_list, config.N)
		self.uninet.load_state_dict(aggregrated_model)

		return aggregrated_model

	def save(self):
		torch.save({'model_state_dict' : self.uninet.state_dict()}, './fedlearn.pth')
		print("Model Saved")

	def test(self, r):
		self.uninet.eval()
		test_loss = 0
		correct = 0
		total = 0
		start = time.time()
		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.testloader)):
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs = self.uninet(inputs)
				loss = self.criterion(outputs, targets)

				test_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()

		acc = 100.*correct/total
		end = time.time()
		logger.info('Test Accuracy: {}'.format(acc))
		print("Time to test is ", end - start)
		config.TESTING_TIMES.append(end - start)
		config.ACCURACIES.append(acc)

		return acc

	def scatter(self, msg):
		for i in self.client_socks:
			self.send_msg(self.client_socks[i], msg)
