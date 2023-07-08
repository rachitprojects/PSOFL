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
		self.port = server_port
		self.model_name = model_name
		self.sock.bind((self.ip, self.port))
		self.client_socks = {}

		while len(self.client_socks) < config.K:
			self.sock.listen(5)
			logger.info("Waiting Incoming Connections.")
			(client_sock, (ip, port)) = self.sock.accept()
			logger.info('Got connection from ' + str(ip))
			logger.info(client_sock)
			self.client_socks[str(ip) + str(port)] = client_sock
			config.CLIENTS_LIST.append(str(ip) + str(port))
		self.uninet = utils.get_model('Unit', self.model_name, config.model_len-1, self.device, config.model_cfg)

		self.transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
		self.testset = torchvision.datasets.CIFAR10(root=config.dataset_path, train=False, download=True, transform=self.transform_test)
		self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=4)

		self.train_time_store = np.array([[0.0 for x in range(config.model_len)] for y in range(config.K)])
		self.cpu_store = np.array([[0.0 for x in range(config.model_len)] for y in range(config.K)])
		self.round_fitness = []

	#def encode(self, n, k):
	#	for x in range(n):
	#		if x == 0:
	#			lower = 0
	#		elif k >= 1:
	#			return n - 1
	#		else:
	#			lower = (1/n) * x
	#		higher = (1/n) * (x + 1)

	#		if k >= lower and k < higher:
        #    			return x

	def encode(self, n, k):
		return sorted([1/n * x for x in range(0, n)] + [k]).index(k)

	def initialize(self, split_layers, LR, r):
		self.split_layers = split_layers
		print(split_layers)
		self.nets = {}
		self.optimizers= {}
		for i in range(len(split_layers)):
			client_ip = config.CLIENTS_LIST[i]
			if split_layers[i] < len(config.model_cfg[self.model_name]) -1: # Only offloading client need initialize optimizer in server
				print("Offloading Initialisation", split_layers[i])
				self.nets[client_ip] = utils.get_model('Server', self.model_name, split_layers[i], self.device, config.model_cfg)

				#offloading weight in server also need to be initialized from the same global weight
				cweights = utils.get_model('Client', self.model_name, split_layers[i], self.device, config.model_cfg).state_dict()
				pweights = utils.split_weights_server(self.uninet.state_dict(),cweights,self.nets[client_ip].state_dict())
				self.nets[client_ip].load_state_dict(pweights)

				self.optimizers[client_ip] = optim.SGD(self.nets[client_ip].parameters(), lr=LR,
				  momentum=0.9)
			else:
				self.nets[client_ip] = utils.get_model('Server', self.model_name, split_layers[i], self.device, config.model_cfg)
		self.criterion = nn.CrossEntropyLoss()

		msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', self.uninet.state_dict()]

		for i in self.client_socks:
			self.send_msg(self.client_socks[i], msg)

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
			print(i)
			if config.split_layer[i] == (config.model_len -1):
				self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_no_offloading, args=(client_ips[i],))
				logger.info(str(client_ips[i]) + ' no offloading training start')
				self.threads[client_ips[i]].start()
			else:
				logger.info(str(client_ips[i]))
				self.threads[client_ips[i]] = threading.Thread(target=self._thread_training_offloading, args=(client_ips[i],))
				logger.info(str(client_ips[i]) + ' offloading training start')
				self.threads[client_ips[i]].start()

		for i in range(len(client_ips)):
			self.threads[client_ips[i]].join()

	def _thread_network_testing(self, client_ip):
		msg = self.recv_msg(self.client_socks[client_ip], 'MSG_TEST_NETWORK')
		msg = ['MSG_TEST_NETWORK', self.uninet.cpu().state_dict()]
		self.send_msg(self.client_socks[client_ip], msg)

	def _thread_training_no_offloading(self, client_ip):
		pass

	def _thread_training_offloading(self, client_ip):
		iteration = int((config.N / (config.K * config.B)))
		print("Number of iterations is ", iteration, " client_ip ", client_ip)
		for i in range(iteration):
			msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
			smashed_layers = msg[1]
			labels = msg[2]

			inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
			self.optimizers[client_ip].zero_grad()
			outputs = self.nets[client_ip](inputs)
			loss = self.criterion(outputs, targets)
			loss.backward()
			self.optimizers[client_ip].step()

			msg = ['MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_'+str(client_ip), inputs.grad]
			self.send_msg(self.client_socks[client_ip], msg)
		return 'Finish'

	def aggregate(self, client_ips):
		w_local_list =[]
		for i in range(len(client_ips)):
			msg = self.recv_msg(self.client_socks[client_ips[i]], 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER')
			if config.split_layer[i] != (config.model_len -1):
				w_local = (utils.concat_weights(self.uninet.state_dict(),msg[1],self.nets[client_ips[i]].state_dict()),config.N / config.K)
				w_local_list.append(w_local)
			else:
				w_local = (msg[1],config.N / config.K)
				w_local_list.append(w_local)
		zero_model = utils.zero_init(self.uninet).state_dict()
		aggregrated_model = utils.fed_avg(zero_model, w_local_list, config.N)
		
		self.uninet.load_state_dict(aggregrated_model)
		return aggregrated_model

	def gather(self, client_ips):
		self.optim_data = []
		for i in range(len(client_ips)):
			msg = self.recv_msg(self.client_socks[client_ips[i]], 'MSG_OPTIMIZATION_DATA')
			self.optim_data.append([msg[1], msg[2]])

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
		logger.info('Test Accuracy: {}'.format(acc))
		end = time.time()
		print("Time to test is ", end - start)
		config.TESTING_TIMES.append(end - start)
		config.ACCURACIES.append(acc)
		# Save checkpoint.
		torch.save(self.uninet.state_dict(), './'+ config.model_name +'.pth')

		return acc

	def obj_func(self, split_layers):
		obj_val = 0
		max_train = 0
		for i in range(len(split_layers)):
			train_time = self.train_time_store[i, split_layers[i]]
			try:
				if train_time > max_train:
					max_train = train_time
			except:
				print("Value Error, skipping")
				print(train_time, max_train, i, split_layers[i])
			obj_val += 0.05 * self.cpu_store[i, split_layers[i]]
		obj_val += 0.5 * max_train

		return obj_val

	def multi_dim_obj_func(self, a, b, c, d, e, f, g, h, i, j):
		a = np.array([self.encode(6, x) for x in a])
		b = np.array([self.encode(6, x) for x in b])
		c = np.array([self.encode(6, x) for x in c])
		d = np.array([self.encode(6, x) for x in d])
		e = np.array([self.encode(6, x) for x in e])
		f = np.array([self.encode(6, x) for x in f])
		g = np.array([self.encode(6, x) for x in g])
		h = np.array([self.encode(6, x) for x in h])
		i = np.array([self.encode(6, x) for x in i])
		j = np.array([self.encode(6, x) for x in j])

		objectives = []
		for k in range(len(a)):
			objectives.append(self.obj_func([a[k], b[k], c[k], d[k], e[k], f[k], g[k], h[k], i[k], j[k]]))

		return np.array(objectives)


	def pso_update(self):
		n_particles = 10
		X = np.random.rand(config.K, n_particles)
		V = np.random.rand(config.K, n_particles) * 0.1
		c1 = 0.1
		c2 = 0.1
		w = 0.8
		pbest = X
		pbest_obj = self.multi_dim_obj_func(X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8], X[9])

		gbest = pbest[:, pbest_obj.argmin()]
		gbest_obj = pbest_obj.min()

		for i in range(1000):
			r1, r2 = np.random.rand(2)
			V = w * V + c1*r1*(pbest - X) + c2*r2*(gbest.reshape(-1,1)-X)
			X = X + V
			obj = self.multi_dim_obj_func(X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8], X[9])
			pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]
			pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
			gbest = pbest[:, pbest_obj.argmin()]
			gbest_obj = pbest_obj.min()

		return [self.encode(6, x) for x in gbest]


	def gen_layer(self, r):
		if r < 7:
			return [r for x in range(config.K)]
		else:
			return self.pso_update()

	def adaptive_offload(self, r):
		print("Inside Adaptive Offload ", self.optim_data)
		fitness_val = 0
		max_train = 0
		# Change to max
		for x in self.optim_data:
			fitness_val += 0.05 * x[1]
			if x[0] > max_train:
				max_train = x[0]
		fitness_val += 0.5 * max_train

		self.round_fitness.append(fitness_val)

		for y in range(len(self.optim_data)):
			split_layer_curr = config.split_layer[y]
			self.train_time_store[y, split_layer_curr] = self.optim_data[y][0]
			self.cpu_store[y, split_layer_curr] = self.optim_data[y][1]

		print("Train Time Store ", self.train_time_store)
		print("CPU Store ", self.cpu_store)
		config.split_layer = self.gen_layer(r + 1)
		#config.split_layer = self.pso_update(r + 1)
		#config.split_layer = [6, 6, 4, 6, 6, 6, 6, 6, 6, 6]
		logger.info('Next Round OPs: ' + str(config.split_layer))

		msg = ['SPLIT_LAYERS',config.split_layer]
		self.scatter(msg)
		return config.split_layer

	def scatter(self, msg):
		for i in self.client_socks:
			self.send_msg(self.client_socks[i], msg)
