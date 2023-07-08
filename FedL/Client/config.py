import sys

# Network configration
SERVER_ADDR= '10.10.0.4'
SERVER_PORT = 51000

K = 10 # Number of devices

# Dataset configration
dataset_name = 'CIFAR10'
dataset_path = "./data"
N = 50000 # data length


# Model configration
model_cfg = {
	# (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
	'VGG5' : [('C', 3, 32, 3, 32*32*32, 32*32*32*3*3*3), ('M', 32, 32, 2, 32*16*16, 0), 
	('C', 32, 64, 3, 64*16*16, 64*16*16*3*3*32), ('M', 64, 64, 2, 64*8*8, 0), 
	('C', 64, 64, 3, 64*8*8, 64*8*8*3*3*64), 
	('D', 8*8*64, 128, 1, 64, 128*8*8*64), 
	('D', 128, 10, 1, 10, 128*10)]
}
model_name = 'VGG5'
model_size = 2.23
split_layer = [6,6,6,6,6,6,6,6,6,6] #Initial split layers
model_len = 7
NETWORK_SPEEDS = []

# FL training configration
R = 100 # FL rounds
LR = 0.01 # Learning rate
B = 100 # Batch size


