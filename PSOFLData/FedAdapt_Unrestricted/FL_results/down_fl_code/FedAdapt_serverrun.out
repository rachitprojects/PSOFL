import os
import subprocess
import signal
pro = subprocess.Popen("sar -u 1 > cpuout.txt", stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
pro_mem = subprocess.Popen("sar -r 1 > memout.txt", stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)

import time
time.sleep(2)
mem_start = time.strftime("%H:%M:%S")


import torch
import pickle
import argparse

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from Sever import Sever
import config
import utils
import PPO
from statistics import mean

parser=argparse.ArgumentParser()
parser.add_argument('--offload', help='FedAdapt or classic FL mode', type= utils.str2bool, default= False)
args=parser.parse_args()

LR = config.LR
offload = args.offload
first = True # First initializaiton control

logger.info('Preparing Sever.')
sever = Sever(0, config.SERVER_ADDR, config.SERVER_PORT, 'VGG5')
first_init = time.time()
sever.initialize(config.split_layer, offload, first, LR)
first = False

first_util = True
state_dim = 2*config.G
action_dim = config.G

model_path_infer = "./PPOunres22062906.pth"
if offload:
	#Initialize trained RL agent 
	agent = PPO.PPO(state_dim, action_dim, config.action_std, config.rl_lr, config.rl_betas, config.rl_gamma, config.K_epochs, config.eps_clip)
	agent.policy.load_state_dict(torch.load(model_path_infer))

end_first_init = time.time()

if offload:
	logger.info('FedAdapt Training')
else:
	logger.info('Classic FL Training')

res = {}
res['trianing_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

training_times = []
ROUND_TIME = []
TRAIN_TIME = []
AGG_TIME = []
RE_INIT_TIME = []
timestamps = []

for r in range(config.R):
	logger.info('====================================>')
	logger.info('==> Round {:} Start'.format(r))
	sever.scatter(["START", True])

	s_time = time.time()
	state, bandwidth = sever.train(thread_number= config.K, client_ips= config.CLIENTS_LIST, round_number=r)
	train_time = time.time()
	aggregrated_model = sever.aggregate(config.CLIENTS_LIST)
	e_time = time.time()

	trianing_time = e_time - s_time


	res['trianing_time'].append(trianing_time)
	res['bandwidth_record'].append(bandwidth)

	test_acc = sever.test(r)
	res['test_acc_record'].append(test_acc)

	with open(config.home + '/results/FedAdapt_res.pkl','wb') as f:
				pickle.dump(res,f)

	logger.info('Round Finish')
	logger.info('==> Round Training Time: {:}'.format(trianing_time))
	training_times.append(trianing_time)

	logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
	if offload:
		split_layers = sever.adaptive_offload(agent, state)
	else:
		split_layers = config.split_layer

	if r > 49:
		LR = config.LR * 0.1

	sever.reinitialize(split_layers, offload, first, LR)
	round_end = time.time()

	sever.save()
	if first_util:
		first_util = False
		timestamps.append([mem_start, time.strftime("%H:%M:%S", time.localtime(s_time)), time.strftime("%H:%M:%S", time.localtime(train_time)), time.strftime("%H:%M:%S", time.localtime(e_time)), time.strftime("%H:%M:%S", time.localtime(round_end))])
	else:
		timestamps.append([time.strftime("%H:%M:%S", time.localtime(s_time)), time.strftime("%H:%M:%S", time.localtime(train_time)), time.strftime("%H:%M:%S", time.localtime(e_time)), time.strftime("%H:%M:%S", time.localtime(round_end))])

	ROUND_TIME.append(round_end - s_time)
	TRAIN_TIME.append(train_time - s_time)
	AGG_TIME.append(e_time - train_time)
	RE_INIT_TIME.append(round_end - e_time)
	logger.info('==> Reinitialization Finish')

	print("First Init is ", end_first_init - first_init)
	print("The model path for inference is ", model_path_infer)
	print("Round times are ", ROUND_TIME)
	print("Training Times are ", TRAIN_TIME)
	print("Aggregation times are ", AGG_TIME)
	print("Initialisation Times are ", RE_INIT_TIME)
	print("Testing Times are ", config.TESTING_TIMES)
	print("Accuracies are ", config.ACCURACIES)
	print("Average Transfer Times are ", config.SERVER_CALC_TIMES)

	print("Timestamps are ", str(timestamps))

os.system("pkill -x sar")
